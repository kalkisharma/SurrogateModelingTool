# Surrogate Modeling Tool

A locally-deployed web application for aerodynamic and CFD engineers to upload tabular datasets, train surrogate models, and interactively explore results — all on-device with no cloud dependencies and no data transmission.

## Features

- **4-step guided wizard**: upload & explore data → configure model → train & validate → predict & export
- **Three model types**: Linear Regression (fast baseline), Gaussian Process Regression (GPR, best for small datasets with uncertainty estimates), and Random Forest (best for medium-to-large datasets)
- **Per-target model selection**: each output column can use a different model type
- **Data Explorer** (Step 1): four pre-training diagnostics — (F1) input correlation heatmap, (F2) feature–output scatter with linearity check, (F3) custom two-variable scatter with range filters, (F4) unusual run detector using Isolation Forest
- **Outlier detection**: IQR-based flagging with adjustable sensitivity (1.5× or 3.0×); optionally exclude flagged rows before training
- **GPR** with ARD kernels (RBF, Matérn) and automatic hyperparameter optimization; per-feature length-scale importance
- **One surrogate per output column**: train CL, CD, and CM simultaneously, each with its own model, metrics, and downloadable pipeline
- **Training history**: compare the last 10 training runs side-by-side; ★ marks the best R² per output
- **Diagnostic plots**: parity plot, residuals, feature importance, 1D sensitivity with ±1σ uncertainty band, 2D response surface, on-demand learning curve
- **Prediction modes**: single-point form with training-range hints and confidence badges, or batch CSV upload with preview table
- **Prediction Explorer**: tracks up to 20 predictions as SVG line plots against the training envelope
- **Export**: download trained pipelines as `.joblib` files, predictions as CSV, and full config + metrics as JSON

## Requirements

- Python 3.8+ (Anaconda/Miniconda strongly recommended — see Troubleshooting)
- All dependencies pinned in `requirements.txt`

## Installation

**From Anaconda Prompt (recommended on Windows):**
```bash
conda activate base
pip install -r requirements.txt
```

**From standard terminal (Linux/macOS):**
```bash
pip install -r requirements.txt
```

> **Windows users:** The pinned NumPy and scikit-learn versions require MKL DLLs that are only available after running `conda activate base`. Installing from a standard terminal or Git Bash will succeed but the app will crash on startup. See [Troubleshooting](#troubleshooting).

## Launch

**From Anaconda Prompt (required on Windows):**
```bash
conda activate base
python run_surrogate_tool.py
```

**From standard terminal (Linux/macOS):**
```bash
python run_surrogate_tool.py
```

The browser opens automatically at `http://localhost:5000`.

## Usage

### Step 1 — Data Explorer

Upload a CSV file with one row per simulation run. The tool displays a summary table (column types, min/max/mean/std, skew distribution badges, coverage tiers, null counts).

**Column selection:** Choose which columns are input features (design parameters you control) and which are output targets (quantities you want to predict). Only numeric columns can be targets. Confirm your selection to clean the data and generate a scatter matrix.

**Outlier detection:** After confirming columns, the tool flags rows outside 1.5× IQR per column. Use the sensitivity selector to switch to 3.0× for data that intentionally spans extreme conditions (e.g. post-stall drag). Check "Exclude flagged rows" to remove them from training data.

**Data Explorer** (expand the "Explore Data Relationships" panel):
- **(F1) Input Correlations** — Pearson heatmap; flags pairs with |r| ≥ 0.92 as potentially redundant
- **(F2) What Your Model Will Learn** — each input vs. each output with a linear trend line; flags non-linear features where GPR or Random Forest will outperform Linear Regression
- **(F3) Investigate Any Two Variables** — custom scatter with per-column range filters, colour-by-third-variable, and log-scale axes
- **(F4) Unusual Run Detector** — Isolation Forest anomaly scores; red dots (score > 0.6) are worth reviewing for setup errors or unexpected operating conditions

### Step 2 — Model Configuration

Choose a model type. The tool recommends one based on your dataset size:

| Model | Best for | Training time | Uncertainty |
|---|---|---|---|
| Linear Regression | Quick baseline, any size | Instant | ✓ (bootstrap resampling) |
| GPR | Small, clean datasets (30–500 rows) | Slow above 500 rows | ✓ analytical |
| Random Forest | Medium–large datasets (100–2000 rows) | Fast | ✓ (tree variance) |

Use **Advanced: use a different model for each output** to assign different models per output column.

**GPR options:** Kernel (RBF for smooth responses, Matérn for near-stall or noisy data), initial length scale (leave at 1.0 — auto-optimized), and alpha (leave at 1e-6 for deterministic CFD solvers).

**RF options:** Max tree depth (5 is recommended) and min samples per leaf.

**Split and validation:** Adjust the train/test split (default 20% held out), optionally enable k-fold cross-validation, and toggle input normalization (recommended for GPR and Linear).

### Step 3 — Train & Validate

Click **Train Model**. Results appear per output column:

- **Health banner** — flags overfitting (train/test R² gap > 0.1) and low accuracy (R² < 0.7)
- **Metrics cards** — RMSE, R², MAE on both train set and held-out test set
- **Training history table** — last 10 runs; ★ marks the best R² per output; download comparison CSV
- **Parity plot** — predicted vs. actual on test runs; points should cluster on the diagonal
- **Residuals plot** — error vs. predicted; random scatter around zero is good; patterns indicate missed trends
- **Feature importance** — normalized influence of each input on this output
- **1D Sensitivity** — sweep one input across its training range while holding others at a reference point; ±1σ band shows model uncertainty
- **2D Response Surface** — colour map over a 30×30 grid of two inputs; GPR adds an uncertainty panel
- **Learning Curve** (on demand) — training and validation R² vs. dataset size; diagnoses overfitting and underfitting

For GPR, the optimized kernel string and per-feature length scales are shown after training. A shorter length scale for a feature means the model found it more relevant.

### Step 4 — Predict & Export

**Download:** Per-target `.joblib` pipeline files, full config + metrics JSON, and predictions CSV.

**Single-point prediction:** Enter one design point. The form auto-fills feature medians, shows training-range hints per input, and flags extrapolation if any input is outside the training data range. Predictions display with confidence badges:
- **High** (green) — model uncertainty < 2% of predicted value
- **Medium** (amber) — 2–5%
- **Low** (red) — > 5%

**Prediction Explorer:** Each prediction is added to a running SVG plot per output, showing the training data envelope (grey band), the training mean (dashed line), and your predictions with ±1σ bars.

**Batch prediction:** Upload a CSV with only the feature columns to get predictions for every row. A preview table shows the first 5 rows; download the full results as CSV.

## Understanding Model Uncertainty (±σ)

All three models display a ±σ uncertainty estimate, but the underlying methods differ:

| Model | ±σ method | What it represents |
|---|---|---|
| Linear Regression | 100 bootstrap resamples | How sensitive predictions are to which rows were in the training set — not a noise model |
| GPR | Posterior standard deviation (analytical) | True probabilistic uncertainty; grows naturally away from training data; the most statistically interpretable ±σ |
| Random Forest | Standard deviation across 200 tree predictions | Ensemble disagreement; a proxy for uncertainty but not a formal statistical confidence interval |

The confidence badge thresholds (<2% / 2–5% / >5%) are based on relative uncertainty (σ / |ŷ|) and apply the same scale to all three methods. GPR uncertainty is the only one with formal probabilistic meaning. For design decisions in regions with sparse training data, run a CFD verification regardless of the confidence level shown.

## Data Storage & Privacy

All data stays on your local machine. The tool makes no network calls.

**Files written during a session:**
- `uploads/` — a copy of each CSV you upload. Files are **not** deleted automatically between sessions.
- `models/` — one `.joblib` pipeline file per trained output per session. Files are **not** deleted automatically.
- `surrogate_tool.log` — rotating log file in the project root (1 MB max, 2 backups).

**Manual cleanup:** To remove all uploaded data and trained models, delete the contents of `uploads/` and `models/`. The directories themselves will be recreated on next launch.

**Export-controlled data (ITAR/EAR/CUI):** This tool does not apply encryption or access control beyond your operating system's file permissions. If you are working with export-controlled or proprietary CFD data, ensure the machine meets your organisation's data-handling requirements before uploading.

## Troubleshooting

**`ImportError`, DLL error, or crash on startup (Windows)**
The pinned NumPy and scikit-learn versions depend on Intel MKL DLLs that are only available after activating the Anaconda environment. Run the tool from **Anaconda Prompt** using `conda activate base && python run_surrogate_tool.py`. Running from Git Bash, standard PowerShell, or VS Code's integrated terminal will fail unless the Anaconda base environment has been activated in that session.

**Port 5000 already in use**
Another process is using port 5000 (common on macOS where AirPlay receiver uses it). Open `run_surrogate_tool.py` and change `port=5000` on the last line to any free port (e.g. `port=5001`). Then navigate to `http://localhost:5001`.

**Browser doesn't open automatically**
The auto-open uses Python's `threading.Timer` and `webbrowser.open`, which may be blocked by some antivirus tools or locked-down environments. If no browser window appears, navigate manually to `http://localhost:5000` after the terminal shows `Running on http://127.0.0.1:5000`.

## Sample Datasets

Five ready-to-use datasets are provided in `sample_data/`. Run the corresponding generator script to (re)produce each CSV:

| File | Rows | Inputs | Outputs | Demonstrates |
|---|---|---|---|---|
| `naca0012_airfoil.csv` | 96 | alpha, Mach, Re | CL, CD, CM | Baseline subsonic airfoil; good first dataset |
| `transonic_naca0012.csv` | 96 | alpha, Mach, Re | CL, CD, CM | Cubic drag divergence — Linear Reg fails, GPR captures it |
| `naca4digit_family.csv` | 150 | camber, camber pos, thickness, alpha, Mach | CL, CD, CM | ARD reveals camber→CL/CM, thickness→CD |
| `wing_design_space.csv` | 200 | AR, sweep, taper, t/c, CL_design, Mach | CD_induced, CD_wave, CD_profile, L/D | 6-input, 4-output; distinct feature sensitivities per output |
| `rocket_nozzle.csv` | 120 | chamber pressure, area ratio, ambient pressure, chamber temp, gamma | Cf, Isp, Ve | Propulsion domain; isentropic physics; log-uniform ambient pressure |

Regenerate any CSV from the project root:
```bash
conda activate base
python sample_data/generate_transonic.py
```

## Notes

- GPR training time scales as O(n³). A warning is shown at 500 rows; for datasets above 2,000 rows Random Forest is a better choice.
- All data remains on your local machine. No external network calls are made.
- The `uploads/` and `models/` directories accumulate files across sessions — see [Data Storage & Privacy](#data-storage--privacy) for cleanup instructions.
