# Surrogate Modeling Tool

A locally-deployed web application for aerodynamic and CFD engineers to upload tabular datasets, train surrogate models (Linear Regression or Gaussian Process Regression), and interactively explore results — all on-device with no cloud dependencies and no data transmission.

## Features

- **4-step guided wizard**: upload data → configure model → train & validate → predict & export
- **Gaussian Process Regression** with ARD kernels (RBF, Matérn) and automatic hyperparameter optimization; per-feature length-scale importance visualization
- **Linear Regression** with normalized coefficient importance
- **One surrogate per output column**: train CL, CD, and CM simultaneously, each with its own metrics and downloadable model
- **Diagnostic plots**: parity plot, residuals, feature importance, 1D sensitivity with GPR uncertainty band
- **Prediction modes**: manual single-point form with training range hints, or batch CSV upload
- **Export**: download trained models as `.joblib` files; download predictions as CSV

## Requirements

- Python 3.8+ (Anaconda/Miniconda recommended)
- All dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda install --file requirements.txt -c conda-forge
```

## Launch

**From Anaconda Prompt (recommended):**
```bash
conda activate base
python run_surrogate_tool.py
```

**From standard terminal:**
```bash
python run_surrogate_tool.py
```

The browser opens automatically at `http://localhost:5000`.

> **Note for Anaconda users:** Run from the Anaconda Prompt or after running `conda activate base` to ensure MKL/numpy DLLs are loaded correctly.

## Usage

1. **Step 1 — Data Explorer**: Upload a CSV file. The tool displays a summary table (column types, min/max/mean, null counts) and a scatter matrix. Select which columns are input features and which are output targets, then click "Confirm".

2. **Step 2 — Model Configuration**: Choose between Linear Regression and GPR. For GPR, set the kernel (RBF or Matérn), initial length scale, and noise level (alpha). Adjust the train/test split and optionally enable k-fold cross-validation.

3. **Step 3 — Training & Validation**: Click "Train Model". Results appear for each target column: RMSE, R², MAE on both train and test sets; parity plot; residuals; feature importance chart. For GPR, the optimized kernel hyperparameters are displayed. Use the sensitivity dropdown to plot how any input affects a target while holding others at their median.

4. **Step 4 — Predict & Export**: Download the trained model(s) as `.joblib` files. Enter a single design point manually or upload a new CSV to get batch predictions. Download predictions as a CSV.

## Notes

- GPR is recommended for datasets up to ~1,000 rows. A warning is shown for datasets larger than 2,000 rows.
- All data remains on your local machine. No external network calls are made.
- Uploaded CSVs are stored temporarily in `uploads/`; trained models are saved in `models/`.
