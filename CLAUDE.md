# CLAUDE.md — Surrogate Modeling Tool

Read this file before making any changes.

---

## What This Project Is

A locally-deployed Python/Flask web application for aerodynamic and CFD engineers.
Engineers upload a CSV dataset, train a surrogate model (Linear Regression, Gaussian Process Regression,
or Random Forest), and explore results interactively — all on-device with zero cloud dependencies.

**Launch command (from Anaconda Prompt):**
```
conda activate base
python run_surrogate_tool.py
```
Browser opens automatically at `http://localhost:5000`.

---

## Project Structure

```
surrogate_tool/
├── run_surrogate_tool.py        # entry point: opens browser thread + starts Flask
├── app/
│   ├── __init__.py              # create_app() factory; sets matplotlib Agg backend; inits APP_STATE
│   ├── routes.py                # all Flask routes — request/response only, zero ML logic
│   ├── ml_engine.py             # all ML logic: kernels, pipelines, training, all plots
│   ├── data_utils.py            # CSV loading, validation, summary, cleaning, pairplot, extrapolation check
│   └── templates/
│       └── index.html           # single-page 4-step wizard; inline CSS + JS, no frameworks
├── uploads/                     # gitignored — temp CSV files saved on upload
├── models/                      # gitignored — .joblib files saved after training
├── requirements.txt
├── README.md
└── CLAUDE.md                    # this file
```

---

## Key Design Decisions

| Topic | Decision | Reason |
|---|---|---|
| Data domain | CFD/Aerodynamic tabular CSV | Primary user base; 50–2,000 rows typical |
| Multi-output | One sklearn Pipeline per target column | CL/CD have different response surfaces |
| GPR kernels | RBF and Matérn only (NO RationalQuadratic) | RQ doesn't support ARD in sklearn |
| ARD kernels | `length_scale=np.ones(n_features)` | Per-feature length scales for feature importance |
| Session state | Global `app.config['STATE']` dict | Single-user local tool; no Flask-Session needed |
| GPR hyperparams | Expose kernel, initial length scale, alpha; n_restarts=5 auto | Engineer-friendly controls + auto-optimize |
| RF hyperparams | Expose `max_depth` + `min_samples_leaf`; n_estimators=200 fixed | Bias-variance controls only |
| GPR warning | Warn (don't block) above 2,000 rows | O(n³) but engineer's choice |
| UI layout | 4-step gated wizard | Can't proceed without completing current step |
| Extrapolation warning | Warn (don't block) when prediction inputs are outside training [min, max] | Silent extrapolation is dangerous |
| Sensitivity plot | 1D sweep with ±1σ uncertainty band; custom reference point | Most-used aerodynamic surrogate diagnostic |
| 2D surface plot | Contourf over 30×30 grid; GPR shows mean+σ side-by-side | AoA×Mach surfaces are standard CFD visualization |
| Feature importance | Linear: normalized coefficients; GPR: 1/length_scale; RF: MDI | Model-specific, maximally informative |
| Bootstrap uncertainty | 100-pipeline bootstrap for Linear; tree-variance for RF; native for GPR | All 3 model types expose ±σ |
| Jargon policy | Engineering meaning first, ML term in parentheses — tool-wide | Target user stalls at Step 2 without plain-English labels |
| Model auto-recommendation | Recommends model based on row count + feature count | Engineer should not have to understand GPR vs RF from scratch |
| `?` help cards | Every plot, metric, and panel header has an inline expandable explanation | Self-teaching tool |
| Config export | `GET /api/download/config` → surrogate_config.json | Reproducibility |
| UI style | Clean light theme, accent `#2563EB` | Professional engineering tool aesthetic |
| Data Explorer | 4 panels: F1 correlation heatmap, F2 feature-target scatter, F3 custom scatter, F4 unusual runs | Pre-training data quality checks |
| Per-target model selection | Each output can independently use Linear/GPR/RF via optional `per_target_config` dict; global hyperparams (GPR alpha/kernel, RF depth) are shared across all targets using that model type | Different outputs may suit different models |
| Training lock | `training_in_progress` flag in STATE; `/api/train` returns 409 if already training | Prevents concurrent training in single-user app |
| Extrapolation badge | Confidence badge turns purple when any prediction input is outside training [min, max] | Silent extrapolation is the top accuracy risk |
| Download filename prefix | All downloads use `Path(upload_filename).stem` as filename prefix | Traceability when engineer has multiple datasets |
| Per-target health banners | Step 3 shows overfitting/underfitting/RMSE%-of-range banners per target after training | Key diagnostic without ML jargon |

---

## Environment

- **Python**: Anaconda base environment at `C:\Users\kalki\anaconda3\python.exe`
- **Flask version**: 1.1.2 — use `attachment_filename=` in `send_file`, NOT `download_name=`
- **NumPy**: 1.21.5 — DLL issue when run from Git Bash/PowerShell; always use Anaconda Prompt or `conda run -n base python ...`
- **sklearn**: 1.0.2
- **Port**: 5000 (not 5001)

---

## Critical Constraints

**Flask 1.x `send_file`**: 4 calls in `routes.py` use `attachment_filename=`. If upgrading to Flask 2.x, change all to `download_name=`.

**GPR std bypass**: `sklearn.pipeline.Pipeline.predict()` does NOT forward `return_std=True` to the underlying GPR. Always extract the estimator directly:
```python
gpr = pipeline.named_steps['model']
X_scaled = pipeline.named_steps['scaler'].transform(X_input)
y_pred, y_std = gpr.predict(X_scaled, return_std=True)
```
Used in `/api/predict`, `get_sensitivity_plot_b64()`, and `get_surface_plot_b64()`.

**ARD length scale key**: After GPR fit: `gpr.kernel_.get_params()['k2__length_scale']` (k2 = base kernel inside `ConstantKernel * base`).

**CV**: Run on `np.vstack([X_train, X_test])` with a fresh pipeline clone — not the fitted pipeline. Intentional to avoid data leakage.

**`_reset_downstream(state, level)`**: Must be called on every new upload (`level='upload'`) and column re-confirm (`level='columns'`). Clears all downstream STATE including `de_*` cache keys.

**RF + normalize**: StandardScaler has no effect on RF (tree splits are scale-invariant). Kept for pipeline uniformity.
