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
# 0. 常量（固定，勿改 —— 改了与历史数字不可比）
# ==================================================================

SEED = 2026

# 标签归一化常数
SALT_MEAN, SALT_STD = 34.745396, 0.6393247
TEMP_MEAN, TEMP_STD = 9.403517, 8.000061
MEAN_STD = {'salt': (SALT_MEAN, SALT_STD), 'temp': (TEMP_MEAN, TEMP_STD)}
STD = {'salt': SALT_STD, 'temp': TEMP_STD}
VALID_VARS = ('salt', 'temp')

# 输入变量（6 个 × mean/std = 12 通道/月），顺序不可变
VARS = ['uo_mean', 'uo_std', 'vo_mean', 'vo_std', 'mlotst_mean', 'mlotst_std',
        'so_mean', 'so_std', 'thetao_mean', 'thetao_std', 'zos_mean', 'zos_std']
DATA_MEAN = [0.00090828876, 0.050593812, 0.0047905985, 0.047351096, 47.800728, 10.757117,
             34.009117, 0.21114606, 13.841141, 0.18191428, -0.18183799, 0.033111636]
DATA_STD = [0.1346134, 0.04672826, 0.0801891, 0.045290526, 52.75789, 24.374735,
            2.3664289, 0.3743279, 11.660583, 0.17114641, 0.7058132, 0.032751158]
N_STATIC = 3                       # deptho / lon / lat

GRID = (170, 360)                  # lat -79.5~89.5 × lon 0.5~359.5
N_LEVELS = 26                       # Argo LEVEL 的 slice(1, 27)
LEVELS_M = np.array([
    5., 10., 20., 30., 50., 75., 100., 125., 150., 200., 250., 300., 400., 500.,
    600., 700., 800., 900., 1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000.])

# 深度分档（闭区间；30-1000 m 恰好 16 层，与 SwinOcean3D 同口径）
BANDS = (('full_5_2000', 0.0, 2001.0), ('swin_30_1000', 30.0, 1000.0),
         ('upper_0_150', 0.0, 150.0), ('mid_150_500', 150.0, 500.0),
         ('deep_500_2000', 500.0, 2001.0))

TRAIN_RANGE = (1, 144)             # 143 个月
VAL_RANGE = (144, 156)             # 12
TEST_RANGE = (156, 180)            # 24

CMEMS_FILE = ('cmems_mod_glo_phy-mnstd_my_0.25deg_P1M-m_multi-vars_'
              '180.00W-179.75E_80.00S-90.00N_0.51m_2004-12-01-2020-04-01.nc')
ARGO_FILE = 'argo_2005-2020_grd_decode_time.nc'
STATIC_FILE = 'cmems_mod_glo_phy_my_0.083deg_static_1775750304049.nc'


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_slices(active):
    """active 顺序 → 通道切片。active=('salt','temp') → [0:26]=SALT、[26:52]=TEMP。"""
    return {v: slice(i * N_LEVELS, (i + 1) * N_LEVELS) for i, v in enumerate(active)}


def band_layers(lo, hi):
    return np.where((LEVELS_M >= lo) & (LEVELS_M <= hi))[0]


def find_data_root(explicit=None):
    """从 --data_root 或本文件位置向上找含 cmems 文件的 Data 目录。"""
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
        f"[fatal] 未找到数据目录（含 {CMEMS_FILE}）。已尝试：\n  "
        + "\n  ".join(cands) + "\n请用 --data_root 指定。")


# ==================================================================
# 1. 模型：FNO2D+1D 共享编码器 + 每变量私有解码器
# ==================================================================

class SpectralConv2d(nn.Module):
    """沿 (H, W) 的 2D 谱卷积，截断到 m1 × m2 个低频模态。"""

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
    """沿通道栈方向的 1D 谱卷积。"""

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
    """单变量的深度调制解码器（FNO-DM 的 DM 组件）。

    z [B, width, H, W] ──(逐层 FiLM 调制 + 共享 1×1 头)──> [B, D, H, W]
                       ──(深度可分离 1D 卷积垂向平滑)────> [B, D, H, W]

    参数量 O(width²)，与层数 D 无关。**必须每个变量私有**：共享 γ/β 时
    盐度误差会恶化 20% 以上（历史 11-salt_temp_union 结论）。
    """

    def __init__(self, width, out_levels=26, dropout=0.1, vert_kernel=3):
        super().__init__()
        self.width, self.out_levels = width, out_levels

        self.depth_embed = nn.Embedding(out_levels, width)
        nn.init.normal_(self.depth_embed.weight, std=0.02)
        self.gamma, self.beta = nn.Linear(width, width), nn.Linear(width, width)
        # 初始化为恒等调制 γ=1、β=0
        nn.init.zeros_(self.gamma.weight); nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

        self.head = nn.Sequential(
            nn.Conv2d(width, max(width // 2, 8), 1), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(max(width // 2, 8), 1, 1))

        self.vert_smooth = nn.Conv1d(out_levels, out_levels, kernel_size=vert_kernel,
                                     padding=vert_kernel // 2, groups=out_levels)

    def forward(self, z):
        B, C, H, W = z.shape
        # emb = self.depth_embed.weight
        # g, b = self.gamma(emb), self.beta(emb)

        out = torch.cat([self.head(z)
                         for d in range(self.out_levels)], dim=1)
        o = self.vert_smooth(out.reshape(B, self.out_levels, H * W))
        return o.reshape(B, self.out_levels, H, W)




class FNODM(nn.Module):
    """共享编码器 + 按 active 构建的私有解码器。

    active=('salt','temp') → 联合模型；active=('salt',) → 独立基线。
    两者编码器逐参数一致，单变量模式不会实例化多余解码器。
    """

    def __init__(self, in_ch=15, out_levels=26, width=128,
                 width_1d=64, modes_2d=(16, 16), modes_1d=16, n_layers_2d=4, n_layers_1d=2,
                 dropout=0.1,vert_kernel=3,
                 active=('salt', 'temp')):
        super().__init__()
        active = tuple(active)
        if not active or any(v not in VALID_VARS for v in active):
            raise ValueError(f"active 只能是 {VALID_VARS} 的非空子集，收到 {active!r}")
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
        s = " | ".join([f"编码器 {enc/1e6:.3f} M（共享,）"]
                       + [f"{v} 解码器 {n/1e6:.3f} M" for v, n in dec.items()])
        return s + f" | 合计 {(enc + sum(dec.values()))/1e6:.3f} M"


# ==================================================================
# 2. 损失：标准化空间 MSE + Kendall 不确定性加权
# ==================================================================

class JointLoss(nn.Module):
    """L = Σ_v [0.5·exp(-s_v)·L_v + 0.5·s_v]，L_v 在标准化空间算。

    SALT/TEMP 量纲差一个数量级以上（σ 0.64 psu vs 8.0 °C），物理空间直接
    相加会让温度项量级约为盐度项的 30 倍，盐度基本学不动，故先在归一空间算。
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
# 3. 数据集
# ==================================================================

class STDataset(torch.utils.data.Dataset):
    """输入（海表 12 通道/月 + 3 静态）、标签（Argo 26 层 × active）。

    Argo 与 CMEMS 相差 1 个月：argo[i] 对应 cmems[i+1]，故单月输入取偏移 (+1,)。
    数据一次性预载进内存；读完后立即关闭 nc 句柄（HDF5 全局锁跨进程会静默死锁）。
    """

    def __init__(self, indices, active=('salt', 'temp'), data_root='../../Data',
                 input_months=1):
        try:
            import xarray as xr
        except ImportError as e:
            raise SystemExit("读取 nc 需要 xarray：pip install xarray netCDF4\n" + str(e))
        self.active = tuple(active)
        self.input_months = int(input_months)
        self.offsets = (1,) if self.input_months == 1 else (-1, 0, 1)
        self.in_ch = 12 * self.input_months + N_STATIC

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
        print(f"[数据] {len(self.indices)} 组 | 输入 {self.in_ch} 通道 "
              f"| 标签 {N_LEVELS * len(self.active)} 通道 | active={self.active}")

        # 读完立刻关句柄：HDF5 全局锁跨进程使用会静默死锁
        for ds in (cmems, argo):
            try:
                ds.close()
            except Exception:
                pass

    # ---------- 构造 ----------

    def _build_input(self, actual_index, cmems):
        steps = [cmems.isel(time=int(actual_index + off)) for off in self.offsets]
        # 通道顺序 var-major / time-minor，与历史模型完全一致
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
        for v in self.active:                     # 顺序必须与 make_slices 一致
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
# 4. 评估
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
    """一次扫描给出：全局 / 分深度档 / 逐层 精度（物理量纲）。"""
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
    """导出测试集预测（物理量纲，带 depth_levels / time_labels 元信息）。"""
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
    print(f"[导出] 测试集预测 -> {path}")


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    tot, n = 0.0, 0
    for inp, tar, msk in loader:
        inp, tar, msk = inp.to(device), tar.to(device), msk.to(device)
        loss, _ = criterion(model(inp), tar, msk)
        optimizer.zero_grad()
        loss.backward()
        # 只裁模型参数：logvar 需要更大的有效步长，一起裁会把加权机制废掉
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        tot += float(loss.item())
        n += 1
    return tot / max(n, 1)


# ==================================================================
# 5. 主流程
# ==================================================================

def build_parser():

    task_name = os.path.basename(__file__)[:-3]
    p = argparse.ArgumentParser(
        description="FNO-DM 最优配置单文件训练脚本（支持 1D 支路不下采样）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ---- 最优配置（默认值即定稿值，一般不用改）----
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


    p.add_argument('--task_name', type=str, default=task_name)

    # ---- 运行相关 ----
    p.add_argument('--data_root', default=None)
    p.add_argument('--out_dir', default=None,
                   help='默认 ./<task_name>_<decoder_type><vars>/')
    p.add_argument('--device', default='cuda')
    p.add_argument('--tf32', type=int, default=0,
                   help='1 可提速，但会改变数值轨迹，与 fp32 历史结果不可比')
    p.add_argument('--max_samples', type=int, default=0, help='>0 只取前 N 个月，冒烟用')
    p.add_argument('--save_every', type=int, default=25, help='每 N 个 epoch 落一次盘')
    p.add_argument('--result_csv', default=os.path.join(_HERE, 'result_all.csv'),
                   help='所有实验结果追加写入的共享 CSV（空串则不写）')
    p.add_argument('--fresh', action='store_true', help='忽略已有 checkpoint，从头训练')
    return p


def append_result_csv(res, path):
    """把单次实验结果追加写入共享 CSV。

    joint 与单变量跑的字段不同，故列集合取并集：文件已存在时读回旧表头，
    出现新字段则重写表头并为历史行补空值；缺失字段一律写空串。
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
    args.task_name = f'{args.task_name}_im{args.input_months}_{args.active}_s{args.seed}'
    if args.out_dir is None:
        args.out_dir = os.path.join(_HERE, args.task_name)

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    active = {'joint': ('salt', 'temp'), 'salt': ('salt',), 'temp': ('temp',)}[args.active]
    data_root = find_data_root(args.data_root)
    in_ch = 12 * args.input_months + N_STATIC
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 78)
    print(f"FNO-DM  |  active={active}  |  输入 {in_ch} 通道 "
          f"（{args.input_months} 个月） ")
    print(f"数据 {data_root}   |   设备 {device}   |   seed {args.seed}")
    print("=" * 78)


    # ---------- 数据 ----------
    def idxs(rng):
        lo, hi = rng
        v = list(range(lo, hi))
        return v[:args.max_samples] if args.max_samples else v

    tr_idx, va_idx, te_idx = idxs(TRAIN_RANGE), idxs(VAL_RANGE), idxs(TEST_RANGE)
    ds_kw = dict(active=active, data_root=data_root, input_months=args.input_months)
    train_ds = STDataset(tr_idx, **ds_kw)
    val_ds = STDataset(va_idx, **ds_kw)
    test_ds = STDataset(te_idx, **ds_kw)
    # num_workers=0：数据已在内存，多进程只会 fork 出多余副本
    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch,
                                            shuffle=True, num_workers=0)
    va_loader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    te_loader = torch.utils.data.DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    # ---------- 模型 / 优化器 ----------
    model = FNODM(in_ch=in_ch, out_levels=N_LEVELS, width=args.width, width_1d=args.width_1d,
                  modes_2d=tuple(args.modes_2d), modes_1d=args.modes_1d,
                  n_layers_2d=args.n_layers_2d, n_layers_1d=args.n_layers_1d,
                  dropout=args.dropout,
                  active=active).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[模型] {model.describe()}")

    criterion = JointLoss(active=active).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': [criterion.logvar_salt, criterion.logvar_temp],
         'lr': args.lr * args.lr_logvar_scale},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epoch, eta_min=1e-5)

    checkpoint_path = os.path.join(args.out_dir, f'fnodm_best_{args.task_name}.pth')
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

    # 断点续跑：已有 checkpoint 就接着跑（--fresh 强制从头）
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
        print(f"[续训] 从 {checkpoint_path} 的 epoch {ck['epoch']} 继续"
              f"（best {best_score:.5f}@{best_epoch}）")

    t0 = time.time()
    epoch = start_epoch - 1
    print(f"[训练] 开始 @ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    for epoch in range(start_epoch, args.epoch + 1):
        tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion, device)
        va = evaluate(model, va_loader, device, active)
        scheduler.step()

        # 综合分：各变量 RMSE / 自身 std 后取平均（无量纲）
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
            print(f"[早停] {args.patience} 个 epoch 无提升（最佳 @ epoch {best_epoch}）")
            break

    # ---------- 最终评估：用验证集最优权重 ----------
    if best_state is not None:
        model.load_state_dict(best_state)
        save_checkpoint(epoch)
        print(f"[评估] 载入最优权重（epoch {best_epoch}）-> {checkpoint_path}")
    else:
        print("[警告] 没有 best_state（续训且未再提升），用当前权重评估")

    meta_labels = getattr(test_ds, 'time_labels', None)
    labels = ([meta_labels[i] for i in te_idx] if meta_labels else None)
    export_predictions(model, te_loader, device, active,
                       os.path.join(args.out_dir, 'fnodm_best_predictions_Heads_flat_linear.npz'),
                       time_labels=labels, time_indices=te_idx)
    ev = evaluate(model, te_loader, device, active)

    # ---------- 逐层剖面 ----------
    prof_path = os.path.join(args.out_dir, 'fnodm_best_profile_Heads_flat_linear.csv')
    with open(prof_path, 'w', newline='', encoding='utf-8-sig') as f:
        cols = ['depth_m'] + [f'{v}_{k}' for v in active
                              for k in ('rmse', 'r2', 'bias', 'n')]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in ev['profile']:
            w.writerow(row)

    # ---------- 结果 ----------
    print("=" * 78)
    print(f"测试集 {len(test_ds)} 个时间点："
          f"{labels[0] if labels else te_idx[0]} ~ {labels[-1] if labels else te_idx[-1]}")
    res = {'best_epoch': best_epoch, 'params': n_param, 'seed': args.seed,
           'active': list(active), 'input_months': args.input_months,
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
                  f"   （{b['n_layers']} 层，SwO-3D 同口径）")
    print("=" * 78)
    total_run_time = f'{(time.time()-t0)/60:.2f} min'
    #
    res['Begin_time']=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))
    res['End_time']=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    res['total_run_time'] = total_run_time
    with open(os.path.join(args.out_dir, 'result_Heads_flat_linear.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[结果] total_run_time {total_run_time} min -> "
          f"{os.path.join(args.out_dir, 'result_Heads_flat_linear.json')}")

    # ---------- 追加写入共享 CSV ----------
    if args.result_csv:
        res = {'task_name': args.task_name, 'best_score': best_score, **res}
        try:
            append_result_csv(res, args.result_csv)
            print(f"[结果] 已追加一行 -> {args.result_csv}")
        except Exception as e:
            print(f"[警告] 写入 {args.result_csv} 失败：{e}")


if __name__ == '__main__':
    main()
'''
cd ~/cyx/G20260410_DR/project/Experiment_FNOModels
conda activate your_env

python fnodm.py --active=joint --input_months=1 --batch=4
python fnodm.py --active=joint --input_months=3 --batch=4
python fnodm.py --active=salt --input_months=1 --batch=4
python fnodm.py --active=salt --input_months=3 --batch=4
python fnodm.py --active=temp --input_months=1 --batch=4
python fnodm.py --active=temp --input_months=3 --batch=4

'''
