#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))


# ==================================================================
# 0. Constants (fixed, do not modify — changes break comparability with historical results)
# ==================================================================

SEED = 2026

# Label normalization constants
SALT_MEAN, SALT_STD = 34.745396, 0.6393247
TEMP_MEAN, TEMP_STD = 9.403517, 8.000061
MEAN_STD = {'salt': (SALT_MEAN, SALT_STD), 'temp': (TEMP_MEAN, TEMP_STD)}
STD = {'salt': SALT_STD, 'temp': TEMP_STD}
VALID_VARS = ('salt', 'temp')

VARS_ALL = ['uo_mean', 'uo_std', 'vo_mean', 'vo_std', 'mlotst_mean', 'mlotst_std',
            'so_mean', 'so_std', 'thetao_mean', 'thetao_std', 'zos_mean', 'zos_std']
DATA_MEAN_ALL = [0.00090828876, 0.050593812, 0.0047905985, 0.047351096, 47.800728, 10.757117,
                 34.009117, 0.21114606, 13.841141, 0.18191428, -0.18183799, 0.033111636]
DATA_STD_ALL = [0.1346134, 0.04672826, 0.0801891, 0.045290526, 52.75789, 24.374735,
                2.3664289, 0.3743279, 11.660583, 0.17114641, 0.7058132, 0.032751158]


DROP_GROUPS = {
    'mla': ('mlotst_mean', 'mlotst_std'),
    'uv': ('uo_mean', 'uo_std', 'vo_mean', 'vo_std'),
    'ssh': ('zos_mean', 'zos_std'),
}
VARS = list(VARS_ALL)
DATA_MEAN = list(DATA_MEAN_ALL)
DATA_STD = list(DATA_STD_ALL)
N_STATIC = 3                       # deptho / lon / lat

GRID = (170, 360)                  # lat -79.5~89.5 × lon 0.5~359.5
N_LEVELS = 26                       # Argo LEVEL slice(1, 27)
LEVELS_M = np.array([
    5., 10., 20., 30., 50., 75., 100., 125., 150., 200., 250., 300., 400., 500.,
    600., 700., 800., 900., 1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000.])

BANDS = (('full_5_2000', 0.0, 2001.0), ('swin_30_1000', 30.0, 1000.0),
         ('upper_0_150', 0.0, 150.0), ('mid_150_500', 150.0, 500.0),
         ('deep_500_2000', 500.0, 2001.0))

TRAIN_RANGE = (1, 144)             # 143 months
VAL_RANGE = (144, 156)             # 12 months
TEST_RANGE = (156, 180)            # 24 months

CMEMS_FILE = ('cmems_mod_glo_phy-mnstd_my_0.25deg_P1M-m_multi-vars_'
              '180.00W-179.75E_80.00S-90.00N_0.51m_2004-12-01-2020-04-01.nc')
ARGO_FILE = 'argo_2005-2020_grd_decode_time.nc'
STATIC_FILE = 'cmems_mod_glo_phy_my_0.083deg_static_1775750304049.nc'


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_drop(drop_spec):
    global VARS, DATA_MEAN, DATA_STD
    if not drop_spec:
        return []
    drop = set()
    for tok in drop_spec.replace(' ', '').split(','):
        if not tok:
            continue
        if tok in DROP_GROUPS:
            drop.update(DROP_GROUPS[tok])
        elif tok in VARS_ALL:
            drop.add(tok)
        else:
            raise SystemExit(
                f"[fatal] --drop unrecognized: {tok!r}; available groups: {sorted(DROP_GROUPS)}, "
                f"or individual variable names: {VARS_ALL}")
    if len(drop) >= len(VARS_ALL):
        raise SystemExit("[fatal] --drop removed all input variables—model cannot train")
    keep = [i for i, v in enumerate(VARS_ALL) if v not in drop]
    VARS = [VARS_ALL[i] for i in keep]
    DATA_MEAN = [DATA_MEAN_ALL[i] for i in keep]
    DATA_STD = [DATA_STD_ALL[i] for i in keep]
    return sorted(drop)


def make_slices(active):
    return {v: slice(i * N_LEVELS, (i + 1) * N_LEVELS) for i, v in enumerate(active)}


def band_layers(lo, hi):
    return np.where((LEVELS_M >= lo) & (LEVELS_M <= hi))[0]


def find_data_root(explicit=None):
    cands = []
    if explicit:
        cands.append(explicit)
    d = _HERE
    for _ in range(4):
        cands.append(os.path.join(d, 'Data'))
        d = os.path.dirname(d)
    for c in cands:
        if os.path.isfile(os.path.join(c, CMEMS_FILE)):
            return os.path.abspath(c)
    raise SystemExit(
        f"[fatal] Data directory not found (containing {CMEMS_FILE}). Searched:\n  "
        + "\n  ".join(cands) + "\nPlease specify with --data_root.")


# ==================================================================
# 1. Model: FNO2D+1D shared encoder + per-variable private decoder
# ==================================================================

class SpectralConv2d(nn.Module):
    """2D spectral convolution over (H, W), truncated to m1 × m2 low-frequency modes."""

    def __init__(self, in_c, out_c, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        s = 1.0 / (in_c * out_c)
        self.w1 = nn.Parameter(s * torch.randn(in_c, out_c, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.randn(in_c, out_c, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        m1, m2 = min(self.m1, H // 2), min(self.m2, W // 2 + 1)
        xf = torch.fft.rfft2(x.float())
        of = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1,
                         device=x.device, dtype=torch.cfloat)
        of[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        of[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(of, s=(H, W)).to(x.dtype)


class SpectralConv1d(nn.Module):

    def __init__(self, in_c, out_c, modes):
        super().__init__()
        self.modes = modes
        s = 1.0 / (in_c * out_c)
        self.w1 = nn.Parameter(s * torch.randn(in_c, out_c, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.randn(in_c, out_c, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, L = x.shape
        m = min(self.modes, L // 2)
        xf = torch.fft.rfft(x.float(), dim=-1)
        of = torch.zeros(B, self.w1.shape[1], L // 2 + 1,
                         device=x.device, dtype=torch.cfloat)
        of[:, :, :m] = torch.einsum("bil,iol->bol", xf[:, :, :m], self.w1[:, :, :m])
        of[:, :, -m:] = torch.einsum("bil,iol->bol", xf[:, :, -m:], self.w2[:, :, :m])
        return torch.fft.irfft(of, n=L).to(x.dtype)


class FNOBlock2D(nn.Module):
    def __init__(self, width, modes, dropout=0.0):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes[0], modes[1])
        self.bypass = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.norm(self.spec(x) + self.bypass(x))))


class FNOBlock1D(nn.Module):
    def __init__(self, width, modes, dropout=0.0):
        super().__init__()
        self.spec = SpectralConv1d(width, width, modes)
        self.bypass = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.norm(self.spec(x) + self.bypass(x))))


class SharedFNOEncoder(nn.Module):
    def __init__(self, in_ch=15, width=128, width_1d=64, modes_2d=(16, 16), modes_1d=16,
                 n_layers_2d=4, n_layers_1d=2, dropout=0.1):
        super().__init__()
        self.in_ch, self.width = in_ch, width

        self.in_proj_2d = nn.Conv2d(in_ch, width, 1)
        self.blocks_2d = nn.ModuleList(
            [FNOBlock2D(width, modes_2d, dropout) for _ in range(n_layers_2d)])
        self.out_proj_2d = nn.Conv2d(width, width, 1)

        self.in_proj_1d = nn.Conv1d(1, width_1d, 1)
        self.blocks_1d = nn.ModuleList(
            [FNOBlock1D(width_1d, modes_1d, dropout) for _ in range(n_layers_1d)])
        self.out_proj_1d = nn.Conv1d(width_1d, 1, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(width + in_ch, width, 1), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(width, width, 1))
        self.fuse_norm = nn.GroupNorm(8, width)

    def forward(self, x):
        B, C, H, W = x.shape
        h2 = self.in_proj_2d(x)
        for blk in self.blocks_2d:
            h2 = blk(h2)
        h2 = self.out_proj_2d(h2)                        # [B, width, H, W]
        h1 = self.in_proj_1d(
            x.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, 1, C))
        for blk in self.blocks_1d:
            h1 = blk(h1)
        h1 = self.out_proj_1d(h1).reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.fuse_norm(self.fuse(torch.cat([h2, h1], dim=1)) + h2)


class DepthModulatedDecoder(nn.Module):

    def __init__(self, width, out_levels=26, dropout=0.1, vert_kernel=3):
        super().__init__()
        self.width, self.out_levels = width, out_levels

        self.depth_embed = nn.Embedding(out_levels, width)
        nn.init.normal_(self.depth_embed.weight, std=0.02)
        self.gamma, self.beta = nn.Linear(width, width), nn.Linear(width, width)
        # Initialize as identity modulation: γ=1, β=0
        nn.init.zeros_(self.gamma.weight); nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

        self.head = nn.Sequential(
            nn.Conv2d(width, max(width // 2, 8), 1), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(max(width // 2, 8), 1, 1))

        self.vert_smooth = nn.Conv1d(out_levels, out_levels, kernel_size=vert_kernel,
                                     padding=vert_kernel // 2, groups=out_levels)

    def forward(self, z):
        B, C, H, W = z.shape
        emb = self.depth_embed.weight
        g, b = self.gamma(emb), self.beta(emb)
        out = torch.cat([self.head(g[d].view(1, self.width, 1, 1) * z
                                   + b[d].view(1, self.width, 1, 1))
                         for d in range(self.out_levels)], dim=1)
        o = self.vert_smooth(out.reshape(B, self.out_levels, H * W))
        return o.reshape(B, self.out_levels, H, W)




class FNODM(nn.Module):

    def __init__(self, in_ch=15, out_levels=26, width=128,
                 width_1d=64, modes_2d=(16, 16), modes_1d=16, n_layers_2d=4, n_layers_1d=2,
                 dropout=0.1,vert_kernel=3,
                 active=('salt', 'temp')):
        super().__init__()
        active = tuple(active)
        if not active or any(v not in VALID_VARS for v in active):
            raise ValueError(f"active must be a non-empty subset of {VALID_VARS}, got {active!r}")
        self.in_ch, self.out_levels, self.active = in_ch, out_levels, active

        self.encoder = SharedFNOEncoder(
            in_ch=in_ch, width=width, width_1d=width_1d, modes_2d=modes_2d,
            modes_1d=modes_1d, n_layers_2d=n_layers_2d, n_layers_1d=n_layers_1d,
            dropout=dropout)
        Dec = DepthModulatedDecoder

        self._decoders = nn.ModuleDict({
            v: Dec(width, out_levels, dropout, vert_kernel) for v in active})

    def encode(self, x):
        if x.shape[2:] != GRID:
            x = F.interpolate(x, size=GRID, mode='bilinear', align_corners=False)
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        outs = [self._decoders[v](z) for v in self.active]
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]

    def forward_split(self, x):
        z = self.encode(x)
        return {v: self._decoders[v](z) for v in self.active}

    def describe(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = {v: sum(p.numel() for p in self._decoders[v].parameters()) for v in self.active}
        s = " | ".join([f"Encoder {enc/1e6:.3f} M (shared)"]
                       + [f"{v} Decoder {n/1e6:.3f} M" for v, n in dec.items()])
        return s + f" | Total {(enc + sum(dec.values()))/1e6:.3f} M"


# ==================================================================
# 2. Loss: standardized spatial MSE + Kendall uncertainty weighting
# ==================================================================

class JointLoss(nn.Module):
    """L = Σ_v [0.5·exp(-s_v)·L_v + 0.5·s_v], L_v computed in standardized space.

    SALT/TEMP differ in magnitude by more than one order (σ 0.64 psu vs 8.0 °C).
    Adding them directly in physical space makes the temperature term ~30× the
    salinity term, leaving salinity barely trained. We therefore compute the loss
    in normalized space first.
    """

    def __init__(self, active=('salt', 'temp')):
        super().__init__()
        self.active = tuple(active)
        self.logvar_salt = nn.Parameter(torch.zeros(1))
        self.logvar_temp = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def weights(self):
        return (0.5 * math.exp(-self.logvar_salt.item()) if 'salt' in self.active else 0.0,
                0.5 * math.exp(-self.logvar_temp.item()) if 'temp' in self.active else 0.0)

    def _mse(self, pred, tar, msk):
        n = msk.sum() + 1e-8
        return ((pred - tar) ** 2 * msk).sum() / n

    def forward(self, pred, target, mask):
        sl = make_slices(self.active)
        total = pred.new_zeros(())
        log = {}
        for v in self.active:
            lv = self._mse(pred[:, sl[v]], target[:, sl[v]], mask[:, sl[v]])
            log[f'loss_{v}'] = float(lv.detach())
            s = self.logvar_salt if v == 'salt' else self.logvar_temp
            total = total + 0.5 * torch.exp(-s) * lv + 0.5 * s
        return total, log


# ==================================================================
# 3. Dataset
# ==================================================================

class STDataset(torch.utils.data.Dataset):
    """Input (surface 12 channels/month + 3 static), labels (Argo 26 levels × active).

    Argo and CMEMS are offset by 1 month: argo[i] corresponds to cmems[i+1],
    so single-month input uses offset (+1,). Data are preloaded into memory once;
    nc file handles are closed immediately afterwards (HDF5 global lock causes
    silent deadlocks across processes).
    """

    def __init__(self, indices, active=('salt', 'temp'), data_root='../../Data',
                 input_months=1):
        try:
            import xarray as xr
        except ImportError as e:
            raise SystemExit("Reading nc requires xarray: pip install xarray netCDF4\n" + str(e))
        self.active = tuple(active)
        self.input_months = int(input_months)
        self.offsets = (1,) if self.input_months == 1 else (-1, 0, 1)
        self.in_ch = len(VARS) * self.input_months + N_STATIC

        cmems = xr.open_dataset(os.path.join(data_root, CMEMS_FILE)).isel(depth=0)
        argo = xr.open_dataset(os.path.join(data_root, ARGO_FILE))

        deptho = torch.from_numpy(
            xr.open_dataset(os.path.join(data_root, STATIC_FILE)).deptho.data)
        deptho = F.interpolate(deptho.unsqueeze(0).unsqueeze(0), (681, 1440),
                                mode='bilinear').squeeze().numpy()
        deptho = (deptho - np.nanmin(deptho)) / (np.nanmax(deptho) - np.nanmin(deptho))
        lon_2d, lat_2d = np.meshgrid(cmems.longitude.values, cmems.latitude.values)
        self.static = np.stack([deptho, 2 * lon_2d / 360 - 1, lat_2d / 90], axis=0)

        self.depth_levels = argo['LEVEL'].values[1:1 + N_LEVELS].astype(np.float32)
        try:
            self.time_labels = [str(np.datetime64(t[0]))[:10]
                               for t in np.asarray(argo['TIME'].values)[:, None]]
        except Exception:
            self.time_labels = None

        self.indices = [int(i) for i in indices]
        self.buf = [self._build(k, cmems, argo) for k in range(len(self.indices))]
        print(f"[Data] {len(self.indices)} groups | input {self.in_ch} channels "
              f"| labels {N_LEVELS * len(self.active)} channels | active={self.active}")

        # Close file handles immediately: HDF5 global lock causes silent deadlocks across processes
        for ds in (cmems, argo):
            try:
                ds.close()
            except Exception:
                pass


    def _build_input(self, actual_index, cmems):
        steps = [cmems.isel(time=int(actual_index + off)) for off in self.offsets]
        # Channel order var-major / time-minor, identical to historical models
        chunks = []
        for v, m, s in zip(VARS, DATA_MEAN, DATA_STD):
            chunks.append(np.stack([(st[v].data - m) / s for st in steps], axis=0))
        data = np.concatenate(chunks + [self.static], axis=0)
        data = np.nan_to_num(data, nan=0.0)
        t = torch.from_numpy(data).float().unsqueeze(0)
        return F.interpolate(t, size=GRID, mode='bilinear', align_corners=True).squeeze(0)

    def _build_label(self, actual_index, argo, var):
        mean, std = MEAN_STD[var]
        raw = argo[var.upper()].isel(TIME=actual_index, LEVEL=slice(1, N_LEVELS + 1)).sel(
            LONGITUDE=slice(0, 360), LATITUDE=slice(-80, 90)).data
        tar = (np.nan_to_num(raw, nan=0.0) - mean) / std
        msk = np.logical_not(np.isnan(raw)).astype(np.float32)
        return tar.astype(np.float32), msk

    def _build(self, index, cmems, argo):
        i = self.indices[index]
        inp = self._build_input(i, cmems)
        tars, msks = [], []
        for v in self.active:                     # Order must match make_slices
            t, m = self._build_label(i, argo, v)
            tars.append(t)
            msks.append(m)
        return (inp,
                torch.from_numpy(np.concatenate(tars, axis=0)).float(),
                torch.from_numpy(np.concatenate(msks, axis=0)).float())

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, i):
        return self.buf[i]


# ==================================================================
# 4. Evaluation
# ==================================================================

def _metrics(acc, sel):
    n = float(acc['cnt'][sel].sum())
    if n <= 0:
        return None
    st, st2 = float(acc['st'][sel].sum()), float(acc['st2'][sel].sum())
    ss_res = float(acc['se'][sel].sum())
    ss_tot = st2 - st * st / n
    return dict(rmse=math.sqrt(ss_res / n),
                r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
                mae=float(acc['sae'][sel].sum()) / n,
                bias=float(acc['sb'][sel].sum()) / n,
                n=int(n))


@torch.no_grad()
def evaluate(model, loader, device, active):
    """Single-pass evaluation: global / per-depth-band / per-layer accuracy (physical units)."""
    model.eval()
    sl = make_slices(active)
    acc = {v: dict(st=np.zeros(N_LEVELS), st2=np.zeros(N_LEVELS), cnt=np.zeros(N_LEVELS),
                    se=np.zeros(N_LEVELS), sae=np.zeros(N_LEVELS), sb=np.zeros(N_LEVELS))
           for v in active}
    for inp, tar, msk in loader:
        pred = model(inp.to(device))
        for v in active:
            mean, std = MEAN_STD[v]
            s = sl[v]
            p = (pred[:, s] * std + mean).float().cpu().numpy().astype(np.float64)
            t = tar[:, s].numpy().astype(np.float64) * std + mean
            m = msk[:, s].numpy().astype(np.float64)
            e = (p - t) * m
            a = acc[v]
            a['st'] += (t * m).sum(axis=(0, 2, 3))
            a['st2'] += ((t * m) ** 2).sum(axis=(0, 2, 3))
            a['cnt'] += m.sum(axis=(0, 2, 3))
            a['se'] += (e * e).sum(axis=(0, 2, 3))
            a['sae'] += np.abs(e).sum(axis=(0, 2, 3))
            a['sb'] += e.sum(axis=(0, 2, 3))

    out = {'overall': {v: _metrics(acc[v], np.arange(N_LEVELS)) for v in active},
           'bands': {}, 'profile': []}
    for name, lo, hi in BANDS:
        sel = band_layers(lo, hi)
        out['bands'][name] = dict(
            depth_lo=float(LEVELS_M[sel].min()), depth_hi=float(LEVELS_M[sel].max()),
            n_layers=int(sel.size),
            metrics={v: _metrics(acc[v], sel) for v in active})
    for k in range(N_LEVELS):
        row = {'depth_m': float(LEVELS_M[k])}
        for v in active:
            m = _metrics(acc[v], np.array([k]))
            if m:
                row.update({f'{v}_rmse': m['rmse'], f'{v}_r2': m['r2'],
                            f'{v}_bias': m['bias'], f'{v}_n': m['n']})
        out['profile'].append(row)
    return out


@torch.no_grad()
def export_predictions(model, loader, device, active, path,
                       time_labels=None, time_indices=None):
    """Export test-set predictions (physical units, with depth_levels / time_labels metadata)."""
    model.eval()
    sl = make_slices(active)
    P, T, M = {v: [] for v in active}, {v: [] for v in active}, {v: [] for v in active}
    for inp, tar, msk in loader:
        parts = model.forward_split(inp.to(device))
        for v in active:
            mean, std = MEAN_STD[v]
            s = sl[v]
            P[v].append((parts[v] * std + mean).float().cpu().numpy())
            T[v].append((tar[:, s] * std + mean).numpy())
            M[v].append(msk[:, s].numpy())
    payload = {}
    for v in active:
        payload[f'{v}_predictions'] = np.concatenate(P[v], axis=0).astype(np.float32)
        payload[f'{v}_targets'] = np.concatenate(T[v], axis=0).astype(np.float32)
        payload[f'{v}_masks'] = np.concatenate(M[v], axis=0).astype(np.float32)
    payload['active'] = np.array(list(active), dtype='U8')
    payload['input_months'] = np.int32(loader.dataset.input_months)
    payload['salt_mean'] = np.float32(SALT_MEAN)
    payload['salt_std'] = np.float32(SALT_STD)
    payload['temp_mean'] = np.float32(TEMP_MEAN)
    payload['temp_std'] = np.float32(TEMP_STD)
    payload['depth_levels'] = np.asarray(loader.dataset.depth_levels, np.float32)
    if time_labels is not None:
        payload['time_labels'] = np.array([str(x) for x in time_labels], dtype='U32')
    if time_indices is not None:
        payload['time_indices'] = np.asarray(time_indices, np.int32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"[Export] Test-set predictions -> {path}")


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    tot, n = 0.0, 0
    for inp, tar, msk in loader:
        inp, tar, msk = inp.to(device), tar.to(device), msk.to(device)
        loss, _ = criterion(model(inp), tar, msk)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        tot += float(loss.item())
        n += 1
    return tot / max(n, 1)


# ==================================================================
# 5. Main
# ==================================================================

def build_parser():

    task_name = os.path.basename(__file__)[:-3]
    p = argparse.ArgumentParser(
        description="FNO-DM single-file training script (final configuration, with 1D branch, no downsampling)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ---- Optimal configuration (defaults are the final values; generally no need to change) ----
    p.add_argument('--active', default='joint', choices=['joint', 'salt', 'temp'])
    p.add_argument('--input_months', type=int, default=1, choices=[1, 3])
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--width_1d', type=int, default=64)
    p.add_argument('--modes_2d', type=int, nargs=2, default=[16, 16])
    p.add_argument('--modes_1d', type=int, default=16)
    p.add_argument('--n_layers_2d', type=int, default=4)
    p.add_argument('--n_layers_1d', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--epoch', type=int, default=500)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--lr_logvar_scale', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--drop', default='',
                   help='Input ablation: mla / uv / ssh, comma-separated or variable names; empty = full 12 channels')


    p.add_argument('--task_name', type=str, default=task_name)

    # ---- Runtime ----
    p.add_argument('--data_root', default=None)
    p.add_argument('--out_dir', default=None,
                   help='Default ./<task_name>_<decoder_type><vars>/')
    p.add_argument('--device', default='cuda')
    p.add_argument('--tf32', type=int, default=0,
                   help='1 = faster, but alters numerical trajectory; not comparable with fp32 historical results')
    p.add_argument('--max_samples', type=int, default=0, help='>0 = use only first N months (smoke test)')
    p.add_argument('--save_every', type=int, default=25, help='Save checkpoint every N epochs')
    p.add_argument('--result_csv', default=os.path.join(_HERE, 'result_all.csv'),
                   help='Shared CSV for appending all experiment results (empty = skip)')
    p.add_argument('--fresh', action='store_true', help='Ignore existing checkpoint, train from scratch')
    return p


def append_result_csv(res, path):
    """Append a single experiment result row to the shared CSV.

    Joint and single-variable runs have different fields, so columns are merged
    via union: if the file already exists, the old header is read back; new fields
    trigger a header rewrite with empty values filled for historical rows; missing
    fields are always written as empty strings.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(res.keys())
    if os.path.exists(path):
        with open(path, 'r', newline='', encoding='utf-8-sig') as f:
            old = csv.reader(f)
            header = next(old, [])
            rows = list(old)
        for k in fieldnames:
            if k not in header:
                header.append(k)
        for r in rows:
            r += [''] * (len(header) - len(r))
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    else:
        header = fieldnames
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(header)
    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow([res.get(k, '') for k in header])


def main():
    args = build_parser().parse_args()
    tag = f"_drop-{args.drop.replace(',', '+').replace(' ', '')}" if args.drop else ''
    args.task_name = (f'{args.task_name}_im{args.input_months}_{args.active}'
                      f'_s{args.seed}{tag}')
    if args.out_dir is None:
        args.out_dir = os.path.join(_HERE, args.task_name)

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    active = {'joint': ('salt', 'temp'), 'salt': ('salt',), 'temp': ('temp',)}[args.active]
    dropped = apply_drop(args.drop)
    data_root = find_data_root(args.data_root)
    in_ch = len(VARS) * args.input_months + N_STATIC
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 78)
    print(f"FNO-DM  |  active={active}  |  input {in_ch} channels "
          f"({args.input_months} months \u00d7 {len(VARS)} vars + {N_STATIC} static) ")
    if dropped:
        print(f"[Ablation] Removed {len(dropped)} channels: {', '.join(dropped)}")
        print(f"          Keeping {len(VARS)} channels: {', '.join(VARS)}")
    print(f"Data {data_root}   |   Device {device}   |   seed {args.seed}")
    print("=" * 78)


    # ---------- Data ----------
    def idxs(rng):
        lo, hi = rng
        v = list(range(lo, hi))
        return v[:args.max_samples] if args.max_samples else v

    tr_idx, va_idx, te_idx = idxs(TRAIN_RANGE), idxs(VAL_RANGE), idxs(TEST_RANGE)
    ds_kw = dict(active=active, data_root=data_root, input_months=args.input_months)
    train_ds = STDataset(tr_idx, **ds_kw)
    val_ds = STDataset(va_idx, **ds_kw)
    test_ds = STDataset(te_idx, **ds_kw)
    # num_workers=0: data already in memory; multiprocessing would only fork extra copies
    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch,
                                            shuffle=True, num_workers=0)
    va_loader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    te_loader = torch.utils.data.DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    # ---------- Model / Optimizer ----------
    model = FNODM(in_ch=in_ch, out_levels=N_LEVELS, width=args.width, width_1d=args.width_1d,
                  modes_2d=tuple(args.modes_2d), modes_1d=args.modes_1d,
                  n_layers_2d=args.n_layers_2d, n_layers_1d=args.n_layers_1d,
                  dropout=args.dropout,
                  active=active).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[Model] {model.describe()}")

    criterion = JointLoss(active=active).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': [criterion.logvar_salt, criterion.logvar_temp],
         'lr': args.lr * args.lr_logvar_scale},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epoch, eta_min=1e-5)

    checkpoint_path = os.path.join(args.out_dir, f'fnodm_{args.task_name}.pth')
    best_score, best_epoch, wait = float('inf'), 0, 0
    best_state = None

    def save_checkpoint(ep):
        torch.save({'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'logvar_salt': criterion.logvar_salt.detach().cpu(),
                    'logvar_temp': criterion.logvar_temp.detach().cpu(),
                    'best_model': best_state, 'best_score': best_score,
                    'best_epoch': best_epoch, 'wait': wait, 'epoch': ep,
                    'active': active, 'in_ch': in_ch}, checkpoint_path)

    start_epoch = 1
    if os.path.exists(checkpoint_path) and not args.fresh:
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['net'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        try:
            scheduler.load_state_dict(ck['scheduler'])
        except Exception:
            pass
        criterion.logvar_salt.data.copy_(ck['logvar_salt'])
        criterion.logvar_temp.data.copy_(ck['logvar_temp'])
        start_epoch = ck['epoch'] + 1
        best_score, best_epoch, wait = ck['best_score'], ck['best_epoch'], ck['wait']
        best_state = ck.get('best_model')
        print(f"[Resume] Continuing from {checkpoint_path} epoch {ck['epoch']}"
              f" (best {best_score:.5f}@{best_epoch})")

    t0 = time.time()
    epoch = start_epoch - 1
    print(f"[Train] Start @ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    for epoch in range(start_epoch, args.epoch + 1):
        tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion, device)
        va = evaluate(model, va_loader, device, active)
        scheduler.step()

        # Composite score: average RMSE / per-variable std (dimensionless)
        score = sum(va['overall'][v]['rmse'] / STD[v] for v in active) / len(active)
        if score < best_score:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if epoch % 5 == 0 or epoch == 1 or wait == 0:
            ws, wt = criterion.weights()
            parts = " | ".join(
                f"{v.upper()} {va['overall'][v]['rmse']:.5f}" for v in active)
            print(f"epoch {epoch:3d} | loss {tr_loss:.5f} | val {parts} "
                  f"| score {score:.5f} | w=({ws:.3f},{wt:.3f}) "
                  f"| best {best_score:.5f}@{best_epoch} "
                  f"| {(time.time()-t0)/60:.1f} min", flush=True)

        if epoch % args.save_every == 0:
            save_checkpoint(epoch)

        if wait >= args.patience:
            print(f"[Early stop] No improvement for {args.patience} epochs (best @ epoch {best_epoch})")
            break

    # ---------- Final evaluation: use validation-set best weights ----------
    if best_state is not None:
        model.load_state_dict(best_state)
        save_checkpoint(epoch)
        print(f"[Eval] Loaded best weights (epoch {best_epoch}) -> {checkpoint_path}")
    else:
        print("[Warning] No best_state (resumed with no further improvement), evaluating with current weights")

    meta_labels = getattr(test_ds, 'time_labels', None)
    labels = ([meta_labels[i] for i in te_idx] if meta_labels else None)
    export_predictions(model, te_loader, device, active,
                       os.path.join(args.out_dir, f'fnodm_{args.task_name}.npz'),
                       # f'result_{}.json'
                       time_labels=labels, time_indices=te_idx)
    ev = evaluate(model, te_loader, device, active)

    # ---------- Per-layer profile ----------
    prof_path = os.path.join(args.out_dir, f'fnodm_{args.task_name}.csv')
    with open(prof_path, 'w', newline='', encoding='utf-8-sig') as f:
        cols = ['depth_m'] + [f'{v}_{k}' for v in active
                              for k in ('rmse', 'r2', 'bias', 'n')]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in ev['profile']:
            w.writerow(row)

    # ---------- Results ----------
    print("=" * 78)
    print(f"Test set {len(test_ds)} time points: "
          f"{labels[0] if labels else te_idx[0]} ~ {labels[-1] if labels else te_idx[-1]}")
    res = {'best_epoch': best_epoch, 'params': n_param, 'seed': args.seed,
           'active': list(active), 'input_months': args.input_months,
           'drop': args.drop, 'n_input_ch': len(VARS),
           'input_vars': ','.join(VARS),
       }
    for v in active:
        m = ev['overall'][v]
        res[f'{v}_rmse'] = m['rmse']; res[f'{v}_r2'] = m['r2']
        res[f'{v}_mae'] = m['mae']; res[f'{v}_bias'] = m['bias']
        print(f"  {v.upper():4s}  5-2000 m   RMSE {m['rmse']:.5f}   R2 {m['r2']:.5f}   "
              f"MAE {m['mae']:.5f}   bias {m['bias']:+.5f}")
    b = ev['bands']['swin_30_1000']
    for v in active:
        m = b['metrics'][v]
        if m:
            res[f'{v}_rmse_30_1000'] = m['rmse']; res[f'{v}_r2_30_1000'] = m['r2']
            print(f"  {v.upper():4s} 30-1000 m   RMSE {m['rmse']:.5f}   R2 {m['r2']:.5f}"
                  f"   ({b['n_layers']} layers, SwO-3D caliber)")
    print("=" * 78)
    total_run_time = f'{(time.time()-t0)/60:.2f} min'
    #
    res['Begin_time']=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))
    res['End_time']=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    res['total_run_time'] = total_run_time
    with open(os.path.join(args.out_dir, f'result_{args.task_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[Result] total_run_time {total_run_time} min -> "
          f"{os.path.join(args.out_dir, f'result_{args.task_name}.json')}")

    # ---------- Append to shared CSV ----------
    if args.result_csv:
        res = {'task_name': args.task_name, 'best_score': best_score, **res}
        try:
            append_result_csv(res, args.result_csv)
            print(f"[Result] Appended one row -> {args.result_csv}")
        except Exception as e:
            print(f"[Warning] Write to {args.result_csv} failed: {e}")


if __name__ == '__main__':
    main()