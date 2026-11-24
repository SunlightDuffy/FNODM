#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FNO-DM 业务化推理 —— CMEMS 月平均表层场 -> 26 层温盐三维产品

把 `fnodm_1028.py` 训练好的模型应用到 2020-04 之后新下载的 CMEMS 再分析月
平均场上，逐月输出 5–2000 m、26 层的盐度/温度三维网格，存成 CF 风格
NetCDF（xarray / Panoply / GIS 可直接读）。

与训练口径的对应关系（改任何一条都会让产品与论文数字不可比）
------------------------------------------------------------
1. 输入通道 = 12 个表层变量（VARS 顺序）+ 3 个静态通道（deptho / lon / lat），
   归一化常数 DATA_MEAN / DATA_STD 全部直接复用 `fnodm_1028`。
2. 输入先 nan_to_num(0)，再 bilinear 插到 170×360（align_corners=True）——
   与 `STDataset._build_input` 逐行一致。
3. 输出在标准化空间，逐变量 `pred*STD+MEAN` 还原（同 `evaluate()`）。
4. 时间口径：训练时 argo[i] 对应 cmems[i+1]（同一真实月份），所以推理直接用
   「当月」CMEMS 场；input_months=3 时取 [j-2, j-1, j]，边界用最近可用月填充。
5. 模型结构不在这里重写：直接 import `fnodm_1028.FNODM`，宽度 / 模态 / 层数
   从 checkpoint 的 state_dict 反推，避免手填超参与权重对不上。

用法
----
    # 默认：联合模型（project/Experiment_FNOModels/fnodm_1028_im1_joint_s2026 的 best 权重）
    #         + 2020-04~2025-12 全部 69 个月
    python fnodm_infer_product.py

    # 只跑一季度，并额外存一份 npz
    python fnodm_infer_product.py --start 2020-04 --end 2020-06 --npz

    # 冒烟：跑 2 个月，只打印统计不写文件
    python fnodm_infer_product.py --max-months 2 --dry-run

输出
----
    project/products/FNODM_so_thetao_202004-202512.nc
        so      (time, depth, latitude, longitude)   psu
        thetao  (time, depth, latitude, longitude)   degC

本文件只做推理与落盘，不训练、不改动任何 checkpoint。
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))          # -> project/Experiment_FNOModels
_ROOT = os.path.dirname(os.path.dirname(_HERE))             # -> 仓库根
sys.path.insert(0, _HERE)

from fnodm_1028 import (                                    # noqa: E402
    DATA_MEAN, DATA_STD, FNODM, GRID, LEVELS_M, MEAN_STD, N_LEVELS,
    N_STATIC, STATIC_FILE, VARS, make_slices)

# ---------------------------------------------------------------- 常量
# 模型输出网格（与训练时的标签网格一致：lat -79.5~89.5，lon 0.5~359.5）
LAT_OUT = np.linspace(-79.5, 89.5, GRID[0]).astype(np.float32)
LON_OUT = (0.5 + np.arange(GRID[1])).astype(np.float32)

SURFACE_NC = ('cmems_mod_glo_phy-mnstd_my_0.25deg_P1M-m_multi-vars_'
              '180.00W-179.75E_80.00S-90.00N_0.51m_2020-04-01-2025-12-01.nc')

# 变量名/单位映射（图件与论文里沿用 psu / °C 的叫法）
PROD_VAR = {'salt': 'so', 'temp': 'thetao'}
PROD_UNITS = {'salt': 'psu', 'temp': 'degC'}
PROD_LONG = {'salt': 'Sea water salinity',
             'temp': 'Sea water potential temperature'}
PROD_STD = {'salt': 'sea_water_salinity',
            'temp': 'sea_water_potential_temperature'}

# 默认权重：同目录下的训练输出目录（不同机器上可能多一层 checkpoints/，
# 文件也可能叫 fnodm_<task>.pth 而不是 fnodm_best_<task>.pth，故由
# resolve_ckpt() 按候选列表自动搜；要精确指定就用 --ckpt。
CKPT_DIR_NAME = 'fnodm_1028_im1_joint_s2026'
CKPT_FILE_NAME = f'fnodm_best_{CKPT_DIR_NAME}.pth'


def log(msg):
    print(msg, flush=True)


def resolve_ckpt(explicit=None):
    """定位权重文件：--ckpt > 脚本目录附近的候选目录 > 目录内任意 .pth。

    训练脚本落盘名是 fnodm_<task_name>.pth，而归档时常被重命名成
    fnodm_best_<task_name>.pth；不同机器上还可能多一层 checkpoints/。
    所以这里不写死单一路径，也不要求文件名完全一致。
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f'[fatal] --ckpt 指定的文件不存在：{explicit}')
        return explicit

    dirs = [
        os.path.join(_HERE, CKPT_DIR_NAME),                        # 本地仓库结构
        os.path.join(_HERE, 'checkpoints', CKPT_DIR_NAME),          # 服务器结构
        os.path.join(_HERE, CKPT_DIR_NAME, 'checkpoints'),
        os.path.join(_ROOT, 'History', 'Experiment_Joint_st', 'checkpoints'),
    ]
    for d in dirs:                                                # 先按常见文件名
        p = os.path.join(d, CKPT_FILE_NAME)
        if os.path.isfile(p):
            return p
    for d in dirs:                                                # 再按目录内任意 .pth
        if os.path.isdir(d):
            hits = sorted(glob.glob(os.path.join(d, '*.pth')))
            if hits:
                log(f'[提示] 未找到 {CKPT_FILE_NAME}，改用 {hits[0]}')
                return hits[0]
    raise SystemExit(
        '[fatal] 找不到权重文件，已尝试：\n  ' + '\n  '.join(dirs)
        + '\n请用 --ckpt 显式指定，例如：\n'
        + '  python fnodm_infer_product.py --ckpt ./checkpoints/xxx.pth')


# ---------------------------------------------------------------- 模型
def infer_arch(sd):
    """从 state_dict 反推超参，避免手填 width/modes 与权重对不上。"""
    width = int(sd['encoder.in_proj_2d.weight'].shape[0])
    width_1d = int(sd['encoder.in_proj_1d.weight'].shape[0])
    modes_2d = tuple(int(x) for x in sd['encoder.blocks_2d.0.spec.w1'].shape[2:])
    modes_1d = int(sd['encoder.blocks_1d.0.spec.w1'].shape[2])
    n_2d = len({k.split('.')[2] for k in sd if k.startswith('encoder.blocks_2d.')})
    n_1d = len({k.split('.')[2] for k in sd if k.startswith('encoder.blocks_1d.')})
    # 解码器：27 = 26 层自适应 + 26 层恒等；vert_smooth 的 kernel 就是垂向平滑窗
    any_dec = next(k for k in sd if k.startswith('_decoders.'))
    dec = any_dec.split('.')[1]
    out_levels = int(sd[f'_decoders.{dec}.depth_embed.weight'].shape[0])
    vert_kernel = int(sd[f'_decoders.{dec}.vert_smooth.weight'].shape[2])
    return dict(width=width, width_1d=width_1d, modes_2d=modes_2d,
                modes_1d=modes_1d, n_layers_2d=n_2d, n_layers_1d=n_1d,
                out_levels=out_levels, vert_kernel=vert_kernel)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    active = tuple(ck['active']) if 'active' in ck else ('salt', 'temp')
    in_ch = int(ck['in_ch'])
    state = ck.get('best_model')
    if state is None:                       # 续训未提升时退回当前权重
        state = ck['net']
    arch = infer_arch(state)
    model = FNODM(in_ch=in_ch, out_levels=arch['out_levels'], width=arch['width'],
                  width_1d=arch['width_1d'], modes_2d=arch['modes_2d'],
                  modes_1d=arch['modes_1d'], n_layers_2d=arch['n_layers_2d'],
                  n_layers_1d=arch['n_layers_1d'], dropout=0.0,
                  vert_kernel=arch['vert_kernel'], active=active)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    log(f"[模型] {ckpt_path}")
    log(f"       active={active}  in_ch={in_ch}  best_epoch={ck.get('best_epoch')}  "
        f"best_score={ck.get('best_score')}")
    log(f"       {model.describe()}")
    return model, active, in_ch


# ---------------------------------------------------------------- 输入场
def build_static(static_nc, sf_nc):
    """复刻 STDataset.__init__ 的 3 个静态通道（deptho / lon / lat）。"""
    import xarray as xr
    with xr.open_dataset(static_nc) as ds:
        deptho = torch.from_numpy(np.asarray(ds['deptho'].data))
    with xr.open_dataset(sf_nc) as ds:
        lon = np.asarray(ds['longitude'].values, dtype=np.float64)
        lat = np.asarray(ds['latitude'].values, dtype=np.float64)
    # 训练时固定插到 681×1440（0.25° 全球），与经度/纬度数无关
    deptho = F.interpolate(deptho.unsqueeze(0).unsqueeze(0), (681, 1440),
                           mode='bilinear').squeeze().numpy()
    deptho = (deptho - np.nanmin(deptho)) / (np.nanmax(deptho) - np.nanmin(deptho))
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    static = np.stack([deptho, 2 * lon_2d / 360 - 1, lat_2d / 90], axis=0)
    # deptho 含 30% 陆地 NaN，bilinear 会把 NaN 扩散到海岸带；随后
    # build_input 里的 nan_to_num(nan=0) 会把这些点归零——与训练时完全相同。
    log(f"[静态] deptho {deptho.shape} 归一化后 "
        f"[{np.nanmin(deptho):.3f}, {np.nanmax(deptho):.3f}] "
        f"有效 {np.isfinite(deptho).mean() * 100:.1f}% | 网格 {lon.size}x{lat.size}")
    return static.astype(np.float32)


def build_input(ds, j, offsets, static):
    """第 j 月的模型输入：(15, 170, 360) float32。逐行对齐 _build_input。"""
    ntime = ds.sizes['time']
    chunks = []
    for v, m, s in zip(VARS, DATA_MEAN, DATA_STD):
        steps = []
        for off in offsets:
            jj = min(max(j + off, 0), ntime - 1)      # 边界用最近可用月填充
            steps.append((np.asarray(ds[v].isel(time=jj).values, dtype=np.float32) - m) / s)
        chunks.append(np.stack(steps, axis=0))        # var-major / time-minor
    data = np.concatenate(chunks + [static], axis=0)
    data = np.nan_to_num(data, nan=0.0)
    t = torch.from_numpy(np.ascontiguousarray(data)).float().unsqueeze(0)
    return F.interpolate(t, size=GRID, mode='bilinear', align_corners=True).squeeze(0)


# ---------------------------------------------------------------- 落盘
def write_netcdf(path, times, cubes, product_mask=None):
    """写成 CF 风格 NetCDF：dims=(time, depth, latitude, longitude)。"""
    import xarray as xr
    data_vars = {}
    for v, cube in cubes.items():
        if product_mask is not None:
            cube = np.where(product_mask, cube, np.nan).astype(np.float32)
        data_vars[PROD_VAR[v]] = (
            ('time', 'depth', 'latitude', 'longitude'), cube,
            {'units': PROD_UNITS[v], 'long_name': PROD_LONG[v],
             'standard_name': PROD_STD[v]})
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': ('time', np.asarray(times, dtype='datetime64[ns]')),
            'depth': ('depth', LEVELS_M.astype(np.float32),
                      {'units': 'm', 'positive': 'down', 'long_name': 'depth'}),
            'latitude': ('latitude', LAT_OUT,
                         {'units': 'degrees_north', 'long_name': 'latitude'}),
            'longitude': ('longitude', LON_OUT,
                          {'units': 'degrees_east', 'long_name': 'longitude'}),
        },
        attrs={
            'title': 'FNO-DM reconstructed ocean temperature and salinity',
            'summary': ('Three-dimensional (26 levels, 5-2000 m) salinity and '
                        'potential temperature reconstructed from CMEMS monthly '
                        'surface fields by the FNO-DM model.'),
            'source': 'CMEMS GLORYS12V1 monthly means (surface fields)',
            'model': 'FNO-DM (shared 2D+1D spectral encoder + depth-modulated decoders)',
            'processing': 'project/Experiment_FNOModels/fnodm_infer_product.py',
            'Conventions': 'CF-1.8',
        })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    enc = {PROD_VAR[v]: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
           for v in cubes}
    ds.to_netcdf(path, encoding=enc)
    log(f"[写出] {path}  ({os.path.getsize(path) / 1e6:.0f} MB)")


# ---------------------------------------------------------------- 主流程
def main():
    p = argparse.ArgumentParser(
        description='FNO-DM 推理：CMEMS 月平均表层场 -> 26 层温盐三维产品',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt', default=None,
                   help=f'权重路径；默认在 {CKPT_DIR_NAME}/ 下自动搜索')
    p.add_argument('--nc', default=os.path.join(_ROOT, 'Data', SURFACE_NC))
    p.add_argument('--static', default=os.path.join(_ROOT, 'Data', STATIC_FILE))
    p.add_argument('--out', default=None,
                   help='默认 project/products/FNODM_so_thetao_<start>-<end>.nc')
    p.add_argument('--start', default='2020-04', help='起始月 YYYY-MM')
    p.add_argument('--end', default='2025-12', help='结束月 YYYY-MM')
    p.add_argument('--stride', type=int, default=1, help='月抽样步长（冒烟用）')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--device', default='auto', help='auto / cpu / cuda')
    p.add_argument('--max-months', type=int, default=0, help='>0 只跑前 N 个月')
    p.add_argument('--mask-land', action='store_true',
                   help='用 deptho<=0 把陆地置 NaN（默认保留模型原始输出）')
    p.add_argument('--npz', action='store_true', help='额外存一份 npz')
    p.add_argument('--dry-run', action='store_true', help='只统计，不写文件')
    args = p.parse_args()

    import xarray as xr
    args.ckpt = resolve_ckpt(args.ckpt)
    for f in (args.nc, args.static):
        if not os.path.isfile(f):
            raise SystemExit(f'[fatal] 找不到文件：{f}')

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    log(f"[设备] {device}"
        + (f"  {torch.cuda.get_device_name(0)}" if device.type == 'cuda' else ''))

    model, active, in_ch = load_model(args.ckpt, device)
    input_months = in_ch // 12 if in_ch % 12 == N_STATIC else 1
    offsets = (0,) if input_months == 1 else (-2, -1, 0)
    log(f"[输入] {input_months} 个月窗口 -> {in_ch} 通道，offsets={offsets}")

    # ---------- 选时间 ----------
    ds = xr.open_dataset(args.nc).isel(depth=0)
    times_all = np.asarray(ds['time'].values)
    labels = np.array([str(np.datetime64(t, 'D'))[:7] for t in times_all])
    keep = [i for i, s in enumerate(labels) if args.start <= s <= args.end][::args.stride]
    if args.max_months:
        keep = keep[:args.max_months]
    if not keep:
        raise SystemExit(f'[fatal] {args.start}~{args.end} 在 {args.nc} 里没有月份')
    log(f"[时间] {labels[keep[0]]} ~ {labels[keep[-1]]} 共 {len(keep)} 个月"
        f"（文件内 {len(labels)} 个月）")

    static = build_static(args.static, args.nc)

    # ---------- 逐月推理 ----------
    cubes = {v: np.full((len(keep), N_LEVELS, GRID[0], GRID[1]), np.nan,
                        dtype=np.float32) for v in active}
    t0 = time.time()
    torch.set_grad_enabled(False)
    for s in range(0, len(keep), args.batch):
        idx = keep[s:s + args.batch]
        inp = torch.stack([build_input(ds, j, offsets, static) for j in idx]).to(device)
        parts = model.forward_split(inp)
        for v in active:
            mean, std = MEAN_STD[v]
            arr = (parts[v][:, :N_LEVELS] * std + mean).float().cpu().numpy()
            cubes[v][s:s + len(idx)] = arr.astype(np.float32)
        done = min(s + args.batch, len(keep))
        speed = (time.time() - t0) / done
        log(f"  [{done:3d}/{len(keep)}] {labels[idx[0]]} .. {labels[idx[-1]]}"
            f"  {speed:.2f} s/月  剩余 {speed * (len(keep) - done) / 60:.1f} min")

    ds.close()

    # ---------- 统计 ----------
    log("=" * 72)
    for v in active:
        a = cubes[v]
        fin = np.isfinite(a)
        log(f"  {PROD_VAR[v]:7s} min {np.nanmin(a):8.3f}  max {np.nanmax(a):8.3f}"
            f"  mean {np.nanmean(a):8.3f}  有效格点 {fin.sum():,}")
    log("=" * 72)

    # ---------- 落盘 ----------
    times = [np.datetime64(t, 'D') for t in times_all[keep]]
    if args.dry_run:
        log("[dry-run] 不写文件")
    else:
        if args.out is None:
            tag = f"{labels[keep[0]].replace('-', '')}-{labels[keep[-1]].replace('-', '')}"
            args.out = os.path.join(_ROOT, 'project', 'products',
                                    f'FNODM_so_thetao_{tag}.nc')
        mask = None
        if args.mask_land:
            mask = _land_mask(args.static)
        write_netcdf(args.out, times, cubes, mask)
        if args.npz:
            npz = os.path.splitext(args.out)[0] + '.npz'
            payload = {f'{v}_predictions': cubes[v] for v in active}
            payload.update({'time_labels': np.array(labels[keep], dtype='U7'),
                            'depth_levels': LEVELS_M.astype(np.float32),
                            'latitude': LAT_OUT, 'longitude': LON_OUT})
            np.savez_compressed(npz, **payload)
            log(f"[写出] {npz}")
        meta = {'ckpt': os.path.abspath(args.ckpt), 'nc': os.path.abspath(args.nc),
                'active': list(active), 'input_months': input_months,
                'n_months': len(keep), 'start': labels[keep[0]],
                'end': labels[keep[-1]], 'mask_land': bool(args.mask_land),
                'elapsed_min': round((time.time() - t0) / 60, 2)}
        with open(os.path.splitext(args.out)[0] + '.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[完成] 用时 {(time.time() - t0) / 60:.1f} min")


def _land_mask(static_nc):
    """deptho>0 为海洋；返回 (170,360) 的布尔掩膜（已插到模型网格）。"""
    import xarray as xr
    with xr.open_dataset(static_nc) as ds:
        deptho = np.asarray(ds['deptho'].data)
    t = torch.from_numpy(np.nan_to_num(deptho, nan=0.0)).float()
    t = F.interpolate(t.unsqueeze(0).unsqueeze(0), size=GRID,
                      mode='bilinear', align_corners=False).squeeze().numpy()
    return t > 0


if __name__ == '__main__':
    main()