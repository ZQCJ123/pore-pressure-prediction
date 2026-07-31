# A Soft-Sensing-Based Method for Ahead-of-Bit Pore Pressure Prediction Under Seismic Constraints

This repository contains the code accompanying the manuscript **"A Soft-Sensing-Based Method for Ahead-of-Bit Pore Pressure Prediction Under Seismic Constraints."** The study targets **3 m ahead-of-bit pore pressure prediction** in offshore drilling under limited logging availability.

The overall framework combines:

1. **Data preprocessing**, including outlier removal, interpolation, and signal smoothing for raw well logs and related measurements.
2. **Soft sensing of bottom-hole information**, using drilling and seismic-related inputs to estimate variables needed by the final predictor.
3. **Seismic-guided ahead-of-bit prediction**, where a Preformer-based sequence model predicts pore pressure while being constrained by a seismic-derived low-frequency pressure trend.

At a unified depth interval of **0.1524 m (0.5 ft)**, the prediction model uses a **40-sample input window** (about **6.1 m** behind the bit) to generate a **20-sample output window** (about **3.0 m** ahead of the bit).

## Repository Structure

```text
.
|-- preprocessing/
|   `-- data-analysis.py
|-- soft sensing/
|   `-- main.py
`-- prediction/
    |-- run.py
    |-- data/
    |   `-- ETT/
    |-- data_provider/
    |-- exp/
    |-- layers/
    |-- models/
    |-- pre model/
    |-- results/
    `-- utils/
```

## What Each Part Corresponds to in the Paper

### 1. `preprocessing/`

This folder contains the data preprocessing code used to clean and smooth raw well data before soft sensing and final pore pressure prediction.

- `data-analysis.py`: preprocessing and analysis script.
- The active preprocessing pipeline in this script performs:
  - global outlier removal using a three-sigma rule,
  - local window-based outlier screening,
  - linear interpolation for missing points introduced during filtering,
  - wavelet decomposition and reconstruction (`db4`, level 2),
  - Savitzky-Golay smoothing on the low-frequency approximation component.

The script also contains additional commented research utilities for:

- file encoding detection,
- depth matching and interpolation,
- cross-plot analysis,
- Eaton-method pore pressure calculation,
- normal compaction trend analysis,
- lithology-depth matching.

### 2. `soft sensing/`

This folder contains the soft-sensing component used in the paper pipeline.

- `main.py`: main script for the hybrid trend-residual soft-sensing workflow.
- The provided implementation uses a combination of:
  - trend-domain experts,
  - residual-domain experts,
  - adaptive reliability-based fusion,
  - depth-wise smoothing and fusion strategies.
- The script is organized around the idea described in the paper: reconstructing missing or hard-to-measure bottom-hole information from available drilling and seismic-related variables.

In the current codebase, the soft-sensing script includes models such as `RidgeCV`, `GradientBoostingRegressor`, `XGBRegressor`, and sequence-aware experts. It also saves prediction curves and serialized model bundles for later use.

### 3. `prediction/`

This folder contains the final ahead-of-bit pore pressure prediction module.

- `run.py`: experiment entry point for training/testing.
- `models/Preformer.py`: the Preformer backbone used in the paper.
- `exp/exp_main.py`: training, validation, testing, and hybrid-loss logic.
- `data_provider/data_loader.py`: data loading, scaling, and well-based sample construction.
- `data/ETT/`: processed data files used by the prediction experiments.
- `pre model/`: saved auxiliary or pretrained models used in the study workflow.
- `results/`: saved prediction outputs.

The prediction model follows the paper design: it uses multivariate drilling sequences together with soft-sensed variables, and it introduces a **hybrid loss** that combines:

- pressure prediction error, and
- consistency with a seismic-derived pressure trend.

## Method Overview

The paper workflow can be summarized as follows:

1. **Data preprocessing**
   - Align drilling, logging, and seismic-derived attributes by depth.
   - Remove abnormal values with global and local statistical filtering.
   - Recover filtered samples by interpolation.
   - Apply wavelet-based smoothing to suppress high-frequency noise while preserving depth-dependent trends.

2. **Soft sensing of bottom-hole information**
   - Estimate bottom-hole variables from available measurements.
   - Use trend-residual decomposition and expert fusion to improve robustness under cross-well distribution shift.

3. **Ahead-of-bit pore pressure prediction**
   - Use the original Preformer architecture as the prediction backbone.
   - Augment the predictor with soft-sensed variables.
   - Introduce seismic-trend consistency as an auxiliary constraint during training.

## Data

The study uses offshore well data from the **Western Australian Petroleum and Geothermal Information Management System (WAPIMS)**. According to the manuscript, five wells (A-E) are used in the study.

Raw or intermediate well-log preprocessing can be handled through:

```text
preprocessing/data-analysis.py
```

Processed example data for the prediction module are provided under:

```text
prediction/data/ETT/
```

These files contain depth-aligned features used by the final prediction model. In the current implementation, the last column is treated as the target pore pressure (`PP`), and the preceding columns are used as model inputs.

## Environment

The code is research code and was not packaged as a single installable library. A typical environment includes:

- Python 3.8+
- NumPy
- pandas
- SciPy
- chardet
- PyWavelets
- scikit-learn
- XGBoost
- matplotlib
- joblib
- PyTorch
- TensorFlow / Keras

## How to Run

### Data preprocessing

The preprocessing entry point is:

```text
preprocessing/data-analysis.py
```

This script currently expects a local input CSV such as:

```text
nimblefoot.csv
```

and processes each log channel column-by-column using outlier filtering, interpolation, and wavelet-based smoothing.

Before running it, please check and adjust:

- the input CSV filename,
- any plot export path such as `filepath`,
- the final CSV export line if you want to save the preprocessed dataset.

Then run:

```bash
cd preprocessing
python data-analysis.py
```

### Soft sensing

Before running the soft-sensing script, please update the local file paths in:

```text
soft sensing/main.py
```

In particular, check:

- `A_PATH`
- `B_PATH`
- `C_PATH`
- `out_dir`

Then run:

```bash
cd "soft sensing"
python main.py
```

### Pore pressure prediction

The prediction module is driven from:

```text
prediction/run.py
```

Typical usage:

```bash
cd prediction
python run.py --is_training 1 --model Preformer --data custom --root_path ./data/ETT/ --data_path wellA_pr_allB.csv --features MS --target PP
```

After training, testing can be run with:

```bash
python run.py --is_training 0 --model Preformer --data custom --root_path ./data/ETT/ --data_path wellA_pr_allB.csv --features MS --target PP
```

Please note:

- the preprocessing script is still organized as a research script and contains several commented experimental utilities in addition to the active preprocessing pipeline;
- `run.py` contains default arguments that may need to be adjusted for your local data file, checkpoint path, and experiment setting.
- Some parts of `exp/exp_main.py` retain commented experimental code paths from the original study.

## Main Results Reported in the Paper

For the proposed method, the manuscript reports the following ahead-of-bit prediction performance on testing wells:

| Well | MAE | RMSE | MAPE (%) |
|------|-----|------|----------|
| B | 0.1658 | 0.2565 | 0.6438 |
| C | 0.1945 | 0.2385 | 0.6200 |
| D | 0.2787 | 0.3870 | 0.8867 |
| E | 0.1138 | 0.1515 | 0.4946 |

The manuscript further states that, relative to the strongest baseline (`Transformer`), the proposed framework achieves substantial reductions in prediction error across all reported testing wells.

## Notes

- This repository is organized to reflect the three main technical parts of the paper workflow: **data preprocessing**, **soft sensing**, and **ahead-of-bit pore pressure prediction**.
- The current code includes research artifacts such as intermediate results, pretrained models, and local-path settings from the original experimental environment.
- If this repository is uploaded for peer review or reproducibility, it is recommended to keep this `README.md` together with the preprocessing script, processed data files, and the two modeling code folders.

## Citation

If you use this code or build on this implementation, please cite the corresponding manuscript:

```text
A Soft-Sensing-Based Method for Ahead-of-Bit Pore Pressure Prediction Under Seismic Constraints
```
