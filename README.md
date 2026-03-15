# CTDF: Cross-temporal downscaling and fusion for hourly 0.01° precipitation estimation

Code for the paper:
> Cross-temporal downscaling and fusion for hourly 0.01° precipitation estimation: A case study in Youxian District, China

---

## Requirements

- Python 3.9
- [autoxgb](https://github.com/abhishekkrthakur/autoxgb)
- xarray
- scipy
- scikit-learn
- pandas
- numpy

Install dependencies:
```bash
pip install autoxgb xarray scipy scikit-learn pandas numpy
```

---

## Repository Structure
```
CTDF_code/
├── GPM_downscaling/
│   ├── train.py          # Train spatial downscaling model (daily GPM at 0.01°)
│   └── predict.py        # Apply trained model to new data
├── Fusion/
│   ├── train.py          # Train fusion model (hourly fused precipitation)
│   └── predict.py        # Apply trained model and evaluate against gauge observations
└── Temporal_disaggregation/
    └── disaggregate.py   # Disaggregate downscaled daily data to hourly
```

---

## Workflow

The CTDF framework consists of three sequential steps:

**Step 1 — Spatial downscaling (GPM_downscaling)**

Train an XGBoost model to downscale daily GPM from 0.1° to 0.01° using auxiliary variables.

Input features: DEM, NDVI, LST, cloud properties
Target: daily GPM precipitation
```bash
python train.py --input train_data.csv --output model_output
python predict.py  # edit input_path and model_path in script
```

**Step 2 — Temporal disaggregation (Temporal_disaggregation)**

Disaggregate the downscaled daily 0.01° data to hourly using temporal ratios derived from coarse-resolution (0.1°) hourly GPM.
```bash
python disaggregate.py \
    --daily-fine    GPM_daily_0.01deg.nc \
    --daily-coarse  GPM_daily_0.1deg.nc \
    --hourly-coarse GPM_hourly_0.1deg.nc \
    --output        GPM_hourly_0.01deg.nc
```

**Step 3 — Fusion (Fusion)**

Train an XGBoost model to fuse the hourly downscaled GPM with ground gauge observations.

Input features: downscaled GPM, cloud properties
Target: gauge observations (ground truth)
```bash
python train.py --input fusion_input.csv --output model_output
python predict.py  # edit input_path and model_path in script
```
