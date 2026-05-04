# CLAUDE.md — Surrogate Modeling Tool

Read this file before making any changes.

---

## What This Project Is

A locally-deployed Python/Flask web application for aerodynamic and CFD engineers.
Engineers upload a CSV dataset, train a surrogate model (Linear Regression, GPR, or Random Forest),
and explore results — all on-device, no cloud.

**Launch:** `conda activate base && python run_surrogate_tool.py` → opens `http://localhost:5000`

---

## Project Structure

```
app/__init__.py       # create_app(); matplotlib Agg backend; STATE init
app/routes.py         # Flask routes — request/response only, no ML logic
app/ml_engine.py      # ML: kernels, pipelines, training, all plots
app/data_utils.py     # CSV loading, summary, cleaning, pairplot, DE plots
app/templates/index.html  # single-page 4-step wizard; inline CSS + JS
```

---

## Key Design Decisions

| Topic | Decision |
|---|---|
| Multi-output | One sklearn Pipeline per target column |
| GPR kernels | RBF and Matérn only (NO RationalQuadratic — no ARD in sklearn) |
| ARD kernels | `length_scale=np.ones(n_features)` for per-feature importance |
| Session state | Global `app.config['STATE']` dict — single-user local tool |
| Jargon policy | Engineering meaning first, ML term in parentheses |
| Data Explorer | 4 panels: F1 correlation heatmap, F2 feature-target scatter, F3 custom scatter, F4 unusual runs |
| Per-target model | Each output can use Linear/GPR/RF via `per_target_config` dict |
| Plot font sizes | `PLOT_LABEL_SIZE=10`, `PLOT_TICK_SIZE=9`, `PLOT_ANNOT_SIZE=9`, `PLOT_TITLE_SIZE=12`, `PLOT_TIGHT_PAD=1.5` (constants in both `ml_engine.py` and `data_utils.py`) |
| Coverage tiers | CV ≥5% ✓ green, 1–5% ! amber, <1% ✗ red; mean≈0 → "—" with tooltip |
| History cells | Stacked: R² on top + RMSE% below in muted text per target |

---

## Environment

- **Python**: `C:\Users\kalki\anaconda3\python.exe`
- **Flask**: 1.1.2 — use `attachment_filename=` in `send_file` (not `download_name=`)
- **NumPy**: 1.21.5 — DLL issue from Git Bash/PowerShell; always use Anaconda Prompt
- **sklearn**: 1.0.2 · **Port**: 5000

---

## Critical Constraints

**Flask 1.x `send_file`**: 4 calls in `routes.py` use `attachment_filename=`.

**GPR std bypass**: `Pipeline.predict()` does NOT forward `return_std=True`. Extract estimator directly:
```python
gpr = pipeline.named_steps['model']
X_scaled = pipeline.named_steps['scaler'].transform(X_input)
y_pred, y_std = gpr.predict(X_scaled, return_std=True)
```
Used in `/api/predict`, `get_sensitivity_plot_b64()`, `get_surface_plot_b64()`.

**ARD length scale key**: `gpr.kernel_.get_params()['k2__length_scale']`

**CV**: Run on `np.vstack([X_train, X_test])` with a fresh pipeline clone — intentional to avoid leakage.

**`_reset_downstream(state, level)`**: Call on every new upload (`level='upload'`) and column re-confirm (`level='columns'`). Clears `de_*` cache keys.

**STATE rollback**: `/api/train` snapshots `results`, `trained`, `last_predictions`, `train_history` before training; restores in all except branches.

**Predict route**: Wrapped in `_predict_impl()` + outer try/except. Returns `model_types` dict (per-target), not `model_type` string.

**Data Explorer cache**: `include_targets=true` bypasses the server-side cache and regenerates heatmap only; `include_targets=false` (default) uses/sets cache.
