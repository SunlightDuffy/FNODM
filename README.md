# FNO-DM: Global 3D Ocean Thermohaline Reconstruction

Code for the paper *"Global Three-Dimensional Ocean Thermohaline Reconstruction from Surface Reanalysis Fields Using FNO-DM"*.

## Repository Structure

```
fno-dm/
├── README.md
├── fnodm.py                  # Training / inference script (baseline)
├── fnodm_infer_product.py         # Batch inference for 2020-2025 products
├── summarize_results.py           # Aggregate metrics across seeds
├── ablations/
│   ├── fnodm_NoDM.py         # Ablation: remove depth modulation
│   ├── fnodm_NoFNO1D.py      # Ablation: remove 1D spectral branch
│   └── fnodm_NoSpectralCov.py# Ablation: remove spectral convolution
├── weights/                       # Pre-trained model checkpoints
│   ├── joints2026/                # Joint T+S, seed=2026, M=1  (Table 4 joint row)
│   ├── salt_s2026/                # Salt-only, seed=2026, M=1  (Table 4 independent row)
│   ├── temp_s2026/                # Temp-only, seed=2026, M=1  (Table 4 independent row)
│   ├── seeds2025/                 # Joint + Salt + Temp, seed=2025  (Table 5,6)
│   └── seeds2027/                 # Joint + Salt + Temp, seed=2027  (Table 5,6)
└── data/                          # Place downloaded .nc files here
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- numpy, netCDF4, xarray
- gsw (TEOS-10 Gibbs SeaWater toolbox)

```bash
pip install torch numpy netCDF4 xarray gsw
```

## Data Preparation

Download three NetCDF files and place them in `data/`:

| File | Source | DOI |
|---|---|---|
| `cmems_mod_glo_phy-mnstd_my_0.25deg_P1M-m_multi-vars_180.00W-179.75E_80.00S-90.00N_0.51m_2004-12-01-2020-04-01.nc` | CMEMS GLOBAL_MULTIYEAR_PHY_ENS_001_031 | `10.48670/moi-00024` |
| `argo_2005-2020_grd_decode_time.nc` | Argo gridded product | `10.17882/42182` |
| `cmems_mod_glo_phy_my_0.083deg_static_1775750304049.nc` | CMEMS static fields | Included with above |

Download: https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_ENS_001_031

## Reproducing Paper Results

### 1. Joint Temperature--Salinity Model (Table 4, joint row)

Train from scratch:
```bash
python fnodm.py --input_months 1 --active salt temp --seed 2026
```

Expected (~4.5 h on consumer GPU):
| Metric | Value |
|---|---|
| Salt RMSE | 0.0789 psu |
| Salt R² | 0.9851 |
| Temp RMSE | 0.4648 °C |
| Temp R² | 0.9967 |

Or use pre-trained weights:
```bash
python fnodm.py --eval_only --resume weights/joints2026/fnodm_best_fnodm_im1_joint_s2026.pth
```

### 2. Independent Salt Model (Table 4, independent row)

```bash
python fnodm.py --input_months 1 --active salt --seed 2026
```

Expected:
| Metric | Value |
|---|---|
| Salt RMSE | 0.0781 psu |
| Salt R² | 0.9854 |

Pre-trained: `weights/salt_s2026/fnodm_best_fnodm_im1_salt_s2026.pth`

### 3. Independent Temperature Model (Table 4, independent row)

```bash
python fnodm.py --input_months 1 --active temp --seed 2026
```

Expected:
| Metric | Value |
|---|---|
| Temp RMSE | 0.4518 °C |
| Temp R² | 0.9968 |

Pre-trained: `weights/temp_s2026/fnodm_best_fnodm_im1_temp_s2026.pth`

### 4. Multi-Seed Stability (Tables 5, 6)

```bash
# Seed 2025
python fnodm.py --input_months 1 --active salt temp --seed 2025
python fnodm.py --input_months 1 --active salt --seed 2025
python fnodm.py --input_months 1 --active temp --seed 2025

# Seed 2027
python fnodm.py --input_months 1 --active salt temp --seed 2027
python fnodm.py --input_months 1 --active salt --seed 2027
python fnodm.py --input_months 1 --active temp --seed 2027
```

Pre-trained weights in `weights/seeds2025/` and `weights/seeds2027/`.

### 5. Ablation Experiments (Table 9)

```bash
# Remove depth modulation
python ablations/fnodm_NoDM.py

# Remove spectral convolution (keep only 1x1 branch)
python ablations/fnodm_NoSpectralCov.py

# Remove 1D channel-mixing branch
python ablations/fnodm_NoFNO1D.py
```

## Key Hyperparameters

| Hyperparameter | Value |
|---|---|
| Hidden width w | 128 |
| Retained modes (m1, m2) | (16, 16) |
| 1D modes | 16 |
| Fourier layers (2D / 1D) | 4 / 2 |
| Input months M | 1 (15 channels) |
| Grid | 170×360 (1°×1°) |
| Depth levels | 26 (5–2000 m) |
| Optimizer | AdamW, lr=2e-4, weight_decay=1e-3 |
| LR schedule | Cosine annealing |
| Batch size / max epochs / patience | 8 / 500 / 30 |
| Loss | Standardized masked MSE + uncertainty weighting |

## Citation

```bibtex
@article{chen2026fnodm,
  title={Global Three-Dimensional Ocean Thermohaline Reconstruction from
         Surface Reanalysis Fields Using {FNO-DM}},
  author={Chen, Yongxiang and Zhang, Zhenchang and Jiang, Yuwu},
  year={2026},
  note={Preprint}
}
```

## License

MIT
