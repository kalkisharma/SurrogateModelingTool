# CLAUDE.md — Surrogate Modeling Tool

This file is the full context document for continuing development on this project.
Read it entirely before making any changes.

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
Browser opens automatically at `http://localhost:5001`.

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
├── requirements.txt             # flask, scikit-learn, pandas, numpy, matplotlib, joblib
├── README.md                    # user-facing install and launch instructions
└── CLAUDE.md                    # this file
```

---

## Key Design Decisions

| Topic | Decision | Reason |
|---|---|---|
| Data domain | CFD/Aerodynamic tabular CSV | Primary user base; 50–2,000 rows typical |
| Multi-output | One sklearn Pipeline per target column | CL/CD have different response surfaces; more flexible |
| GPR kernels | RBF and Matérn only (NO RationalQuadratic) | RQ doesn't support ARD in sklearn — see below |
| ARD kernels | `length_scale=np.ones(n_features)` | Per-feature length scales for feature importance |
| Session state | Global `app.config['STATE']` dict | Single-user local tool; no Flask-Session needed |
| GPR hyperparams | Expose kernel, initial length scale, alpha; n_restarts=5 auto | Engineer-friendly: key controls + auto-optimize |
| RF hyperparams | Expose `max_depth` + `min_samples_leaf`; n_estimators=200 fixed | Bias-variance controls only; tree count irrelevant to engineer |
| Post-train display | Show optimized kernel values (str + length scale table) | Engineers need to know what was actually used |
| GPR warning | Warn (don't block) above 2,000 rows | O(n³) but engineer's choice; suggest RF instead |
| UI layout | 4-step gated wizard | Can't proceed without completing current step |
| Prediction modes | Manual single-point form + CSV batch upload | Both interactive and automated workflows |
| Extrapolation warning | Warn (don't block) when prediction inputs are outside training [min, max] | Silent extrapolation is dangerous for design decisions |
| Sensitivity plot | 1D sweep with GPR ±1σ uncertainty band; custom reference point via Step 4 form | Most-used aerodynamic surrogate diagnostic |
| 2D surface plot | Contourf over 30×30 grid; GPR shows mean+σ side-by-side | AoA×Mach surfaces are the standard CFD visualization |
| Feature importance | Linear: normalized coefficients; GPR: 1/length_scale; RF: MDI | Model-specific, maximally informative |
| Model comparison | Lightweight metrics history (last 3 runs) in STATE | Engineers always ask "which model is better for my data?" |
| Feature relevance warning | Alert when GPR length scale ≥ 500 (near upper bound 1e3) | Signals near-irrelevant features that waste model capacity |
| Config export | `GET /api/download/config` → surrogate_config.json | Reproducibility — engineers share configs and use them in pipelines |
| UI style | Clean light theme, accent `#2563EB` | Professional engineering tool aesthetic |
| Flask compat | `attachment_filename=` in `send_file` | Flask 1.1.2 installed on this machine (NOT Flask 2.x `download_name=`) |
| Bootstrap uncertainty | 100-pipeline bootstrap for Linear; tree-variance for RF; native for GPR | All 3 model types expose ±σ in predictions and sensitivity plots |
| Jargon policy | Engineering meaning first, ML term in parentheses — tool-wide | Target user (A/B knowledge) stalls at Step 2 without plain-English labels |
| Model auto-recommendation | Recommends model based on row count + feature count; pre-selects and explains | Engineer should not have to understand GPR vs RF trade-offs from scratch |
| `?` help cards | Every plot, metric, and panel header has an inline expandable explanation | Self-teaching tool — no external documentation needed |
| Plot captions | Static caption on parity/residuals/importance; dynamic on learning curve + health banner | Engineers need immediate reading of each chart, not just labels |
| Prediction explorer | Client-side SVG chart accumulating up to 20 predictions vs training envelope | Visual design space exploration without additional CFD runs |
| Exploration download | Client-side CSV blob from `appState.explorationHistory` — no server route | All 20 entries downloadable; no server state change needed |

---

## Environment

- **Python**: Anaconda base environment at `C:\Users\kalki\anaconda3\python.exe`
- **Flask version**: 1.1.2 — use `attachment_filename=` in `send_file`, NOT `download_name=`
- **NumPy**: 1.21.5 — has DLL issue when run from Git Bash/PowerShell directly; always run via Anaconda Prompt or `conda run -n base python ...`
- **sklearn**: 1.0.2 — `RandomForestRegressor` available in `sklearn.ensemble`
- **Key conda run command for testing**: `conda run -n base python <script.py>`

---

## APP_STATE Schema

`app.config['STATE']` is a plain Python dict initialized in `create_app()`.
It is the single source of truth for all session data. All routes access it via `current_app.config['STATE']`.
Reset on new upload via `_reset_downstream()` in `routes.py`.

```python
{
    # ---- Step 1: Data ----
    'df_raw': None,              # pd.DataFrame | None — original uploaded CSV
    'df_clean': None,            # pd.DataFrame | None — after dropna on selected cols
    'upload_filename': '',       # str — original filename for display
    'summary': None,             # dict — from get_summary(); JSON-serializable
    'pairplot_b64': None,        # str | None — base64 PNG, computed after set_columns
    'feature_cols': [],          # list[str] — selected input columns
    'target_cols': [],           # list[str] — selected output columns
    'n_dropped': 0,              # int — rows dropped by clean_data()

    # ---- Step 2: Config (stored when /api/train is called) ----
    'train_config': {
        'model_type': 'linear',     # 'linear' | 'gpr' | 'rf'
        'kernel_type': 'rbf',       # 'rbf' | 'matern' (GPR only)
        'length_scale': 1.0,        # float — initial; optimizer refines it (GPR only)
        'alpha': 1e-6,              # float — GPR noise regularization
        'test_size': 0.2,           # float — fraction for test split
        'use_cv': False,            # bool
        'cv_k': 5,                  # int — folds
        'normalize': True,          # bool — whether StandardScaler is in pipeline
        'feature_cols': [],         # list[str] — copy stored here for ml_engine access
        'max_depth': None,          # int | None — RF max tree depth (None = unlimited)
        'min_samples_leaf': 1,      # int — RF minimum samples per leaf
    },

    # ---- Step 3: Results ----
    'trained': False,
    'gpr_warning': None,         # str | None
    'results': {
        # keyed by target column name, e.g. 'CL', 'CD'
        # target_name: {
        #     'pipeline': fitted sklearn Pipeline (in RAM — NOT JSON serializable),
        #     'model_path': str — abs path to .joblib on disk,
        #     'metrics_train': {'rmse': float, 'r2': float, 'mae': float},
        #     'metrics_test':  {'rmse': float, 'r2': float, 'mae': float},
        #     'cv_score': float | None,
        #     'parity_b64': str,
        #     'residuals_b64': str,
        #     'feat_importance_b64': str,
        #     'optimized_kernel_str': str | None,              # str(gpr.kernel_) after fit
        #     'optimized_length_scales': list[float] | None,  # one per feature (GPR only)
        #     'irrelevant_feature_warnings': list[str],        # GPR: length_scale >= 500
        # }
    },
    'train_history': [],         # list of last 3 {model_type, kernel_type, timestamp, metrics}

    # ---- Data Explorer cache (cleared by _reset_downstream('columns')) ----
    'de_corr_b64': None,         # str | None — cached correlation heatmap PNG
    'de_corr_pairs': None,       # list | None — [{col_a, col_b, r}] pairs with |r| >= 0.92
    'de_ft_b64': None,           # str | None — cached feature-target scatter grid PNG
    'de_nonlinear_cols': None,   # list | None — feature names with linear R² < 0.7
    'de_unusual_scores': None,   # dict | None — {row_idx: score} for ALL rows; set by /api/unusual_runs; used by /api/data_scatter

    # ---- Step 4 ----
    'last_predictions': None,    # pd.DataFrame | None — for download
}
```

---

## API Routes Reference (`routes.py`)

All routes are on the `main` Blueprint, registered with no URL prefix.

| Method | Path | Input | Returns |
|---|---|---|---|
| GET | `/` | — | `render_template('index.html')` |
| POST | `/api/upload` | `multipart` field `file` (CSV) | `{summary, filename}` or `{error}` |
| POST | `/api/set_columns` | JSON `{feature_cols, target_cols}` | `{n_rows, n_dropped, pairplot_b64}` |
| POST | `/api/train` | JSON config dict | `{trained, gpr_warning, model_type, feature_cols, target_cols, results, train_history}` |
| GET | `/api/download/model/<target>` | URL param | Binary `.joblib` stream |
| GET | `/api/download/predictions` | — | CSV stream |
| GET | `/api/download/config` | — | JSON stream (`surrogate_config.json`) |
| POST | `/api/predict` | JSON body (single) OR multipart file (batch) | `{predictions, feature_cols, target_cols, model_type, extrapolation_warnings}` |
| GET | `/api/sensitivity` | `?feature=X&target=Y[&ref_<col>=val...]` | `{plot_b64, feature, target}` |
| GET | `/api/surface` | `?feature_x=X&feature_y=Y&target=Z` | `{plot_b64, feature_x, feature_y, target}` |
| GET | `/api/learning_curve` | `?target=Y` | `{plot_b64, target, final_train_r2, final_val_r2, val_still_rising}` |
| GET | `/api/data_explorer` | — | `{corr_heatmap_b64, feat_target_grid_b64, high_corr_pairs, nonlinear_hint_cols, n_plotted, n_total_features}` |
| GET | `/api/unusual_runs` | `?doe_type=grid\|lhs\|random` | `{plot_b64, top_runs, doe_caveat, n_total, flagged_detail, feature_bounds, all_scores}` |
| GET | `/api/data_scatter` | `?x_col=&y_col=&x_min=&x_max=&y_min=&y_max=&color_col=&x_log=0&y_log=0` | `{plot_b64, n_filtered, n_plotted, n_color_missing, log_warning}` |

### Critical route implementation notes

**`/api/upload`**: saves to `uploads/{timestamp}_{secure_filename}`. Calls `_reset_downstream(state, 'upload')` to wipe all downstream state on every new upload. Also resets `train_history`.

**`/api/set_columns`**: validates no column appears in both lists; rejects non-numeric targets; calls `clean_data()` then `get_pairplot_b64()`; calls `_reset_downstream(state, 'columns')` to clear training results. Accepts optional `exclude_outliers: bool` to exclude IQR-flagged rows before pairplot. Returns `outlier_info` dict (from `get_outlier_flags()`), `n_outliers_excluded`, `n_rows` (post-clean row count).

**`/api/train`**: calls `train_all()` from `ml_engine`. GPR warning set if `n_rows > 2000`. Response strips the `pipeline` object (not JSON-serializable) — the pipeline stays in `STATE['results'][target]['pipeline']` in RAM. Appends to `STATE['train_history']` (capped at 3 entries). RF params `max_depth` and `min_samples_leaf` are read from the request body; `max_depth=0` is converted to `None` (unlimited). Response also includes `feature_medians: {col: float}` used to pre-fill sensitivity reference inputs.

**`/api/predict`**: detects mode by `request.content_type`. Always runs `check_extrapolation()` and returns `extrapolation_warnings` list. Returns `{target}_std` fields for each target when available: GPR uses native std, RF uses tree-level variance (`_rf_std()`), Linear uses bootstrap std (`_bootstrap_std()`). For GPR std output, bypasses the Pipeline to call `gpr.predict(X_scaled, return_std=True)` directly — sklearn Pipeline does NOT forward `return_std=True`. Pattern:
```python
gpr = pipeline.named_steps['model']
if 'scaler' in pipeline.named_steps:
    X_scaled = pipeline.named_steps['scaler'].transform(X_input)
else:
    X_scaled = X_input
y_pred, y_std = gpr.predict(X_scaled, return_std=True)
```

**`/api/sensitivity`**: Builds `X_ref` from training data medians, then overrides individual features using optional `ref_<col>=value` query params (populated from Step 4 form inputs in the frontend). `x_sweep` is 100 points from `col_min - 10%` to `col_max + 10%`.

**`/api/surface`**: Builds 30×30 meshgrid over ±5% of each feature's training range. Calls `get_surface_plot_b64()`. For GPR: returns side-by-side mean + σ contours. For others: mean only.

**`/api/download/config`**: Serializes `STATE['train_config']` to JSON. Also injects `target_cols` (not stored in `train_config` directly).

**`/api/download/model/<target>`**: uses `attachment_filename=` (Flask 1.x). If upgraded to Flask 2.x, change to `download_name=`.

**`/api/data_explorer`**: Returns correlation heatmap and feature-target scatter. Passes `max_feat_cols=min(n_features, 6)` to `get_feat_target_grid_b64()`. Results cached in `STATE['de_*']` on first call; returned directly on subsequent calls. Returns `n_plotted` and `n_total_features` in both cached and fresh paths so the client caption ("Showing N of M") always works. Cache invalidated by `_reset_downstream(state, 'columns')`.

**`/api/unusual_runs`**: Runs Isolation Forest on `df_clean[feature_cols].dropna()`. Returns lollipop chart, top-10 `top_runs`, `doe_caveat` for grid DoE, plus: `n_total` (total clean rows), `flagged_detail` (feature values for rows scoring >0.6), `feature_bounds` (IQR lo/hi per column for cell highlighting), and `all_scores` (scores for ALL rows as `[{row_idx, score}]`). Also writes `STATE['de_unusual_scores'] = {row_idx: score}` for use by `/api/data_scatter`. Not cached — user-triggered on demand.

**`/api/data_scatter`**: Custom scatter for any two numeric columns from the selected feature+target set. Applies range filters first, then random-samples to 500 (`filter → sample` order). Supports three colour modes: none (blue `#2563EB`), third column (viridis + colorbar; NaN rows in grey `#94a3b8`), unusualness (`STATE['de_unusual_scores']`, three-tier red/orange/blue). Log scale applied per axis; returns `log_warning` if column has non-positive values. Returns `n_filtered` (pre-sample) and `n_plotted` (post-sample) for client caption.

**`_reset_downstream(state, level)`**: On `level='columns'`, clears training results and all five `de_*` cache keys including `de_unusual_scores`. On `level='upload'`, same via the columns clear path.

---

## ML Engine Reference (`ml_engine.py`)

### Kernel Construction (`build_kernel`)
Always uses **ARD (Automatic Relevance Determination)**:
```python
ls_vec = np.ones(n_features) * length_scale   # one length scale per feature
RBF(length_scale=ls_vec, length_scale_bounds=(1e-3, 1e3))
# or
Matern(length_scale=ls_vec, length_scale_bounds=(1e-3, 1e3), nu=1.5)
# wrapped with:
ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * base_kernel
```
After fitting, `gpr.kernel_.get_params()['k2__length_scale']` is a numpy array of shape `(n_features,)`.
Shorter length scale → feature has more influence on predictions.

**Why no RationalQuadratic?** sklearn's `RationalQuadratic` kernel has a single scalar length scale — it does not support ARD (vector length_scale). Since ARD is required for per-feature importance, RQ is excluded.

### Pipeline Construction (`build_pipeline(model_type, kernel, alpha, normalize, config=None)`)
```python
steps = [('scaler', StandardScaler())]  # if normalize=True
# Linear:
steps += [('model', LinearRegression())]
# GPR:
steps += [('model', GaussianProcessRegressor(
    kernel=kernel, alpha=alpha,
    normalize_y=True,          # always True
    n_restarts_optimizer=5,    # always 5
    random_state=42,
))]
# RF — config dict provides max_depth and min_samples_leaf:
steps += [('model', RandomForestRegressor(
    n_estimators=200,
    max_depth=config.get('max_depth'),   # None = unlimited
    min_samples_leaf=config.get('min_samples_leaf', 1),
    random_state=42,
))]
```
Note: `config` param is only used for RF. Pass `None` or `{}` for linear/GPR.

### Uncertainty Helpers
- `_gpr_std(pipeline, X)` — bypasses Pipeline, calls `gpr.predict(X_scaled, return_std=True)`
- `_rf_std(pipeline, X)` — tree-level variance: `np.std([t.predict(X) for t in rf.estimators_], axis=0)`; zero extra training
- `_bootstrap_std(bootstrap_models, X)` — std across 100 bootstrap-resampled linear pipelines stored in `results[target]['bootstrap_models']`
- `N_BOOTSTRAP = 100` constant controls linear bootstrap count

### `train_all()` — Main Training Function
- For each target column: builds pipeline, fits, computes metrics, generates plots, saves joblib
- CV: fits a **fresh** identical pipeline on `np.vstack([X_train, X_test])` to avoid data leakage
- Returns dict keyed by target name; the `pipeline` key holds the fitted object in RAM
- After GPR fit, checks `optimized_length_scales` for values ≥ 500 → populates `irrelevant_feature_warnings`
- For Linear model: also trains 100 bootstrap pipelines stored in `results[target]['bootstrap_models']`

### Plot Functions (all return base64 PNG str)
- `get_parity_plot_b64(y_true, y_pred, target_name)` — 5×5 in, scatter + diagonal line
- `get_residuals_plot_b64(y_true, y_pred, target_name)` — 10×4 in, two subplots
- `get_feature_importance_plot_b64(pipeline, feature_names, model_type, target_name)`:
  - Linear: `np.abs(coef) / sum` → horizontal bar chart
  - GPR: `1 / (length_scales + 1e-12)` normalized → horizontal bar chart
  - RF: `pipeline.named_steps['model'].feature_importances_` → horizontal bar chart (MDI)
- `get_learning_curve_plot_b64(train_sizes, train_scores, val_scores, target_name)` — blue train / red val R² vs dataset size
- `get_sensitivity_plot_b64(pipeline, X_ref, feature_names, feature_idx, target_name, model_type, x_sweep, bootstrap_models=None)`:
  - GPR: bypasses Pipeline for `return_std=True` → shaded ±1σ band
  - RF: `_rf_std()` → shaded ±1σ band
  - Linear: `_bootstrap_std(bootstrap_models, ...)` → shaded ±1σ band (pass `bootstrap_models` from `STATE['results']`)
  - Adds vertical dashed line at reference value
- `get_surface_plot_b64(pipeline, X_ref, feature_names, idx_x, idx_y, target_name, model_type, x_range, y_range, n_grid=30)`:
  - Builds 30×30 meshgrid, tiles X_ref, replaces idx_x and idx_y columns
  - GPR: `fig, (ax1, ax2)` — left=mean contourf, right=σ contourf (12×4 in)
  - Linear/RF: `fig, ax` — single mean contourf (6×4 in)
  - Uses same GPR std bypass pattern as sensitivity plot

### Model Saving (`save_model`)
```python
filepath = os.path.join(models_dir, f'model_{safe_name}.joblib')
joblib.dump(pipeline, filepath)
```
`safe_name` sanitizes spaces → `_` and `/` → `-`.

---

## Data Utilities Reference (`data_utils.py`)

### `validate_and_load_csv(filepath) → (df | None, error | None)`
- Catches all read errors
- Requires ≥ 2 numeric columns
- Strips whitespace from all column names

### `get_summary(df) → dict`
Returns: `{shape, columns, dtypes, null_counts, stats}` — all JSON-serializable.
`stats` populated for numeric columns: `{col: {min, max, mean, skew, cv}}`.
- `skew`: pandas `.skew()` — 0 = symmetric; >1 right-skewed, <-1 left-skewed
- `cv`: coefficient of variation = std / |mean|; very small (< 0.01) → "barely varies" badge in summary table

### `clean_data(df, feature_cols, target_cols) → (df_clean, n_dropped)`
Selects only the specified columns, drops rows with any NaN, resets index.

### `check_extrapolation(X_input, df_clean, feature_cols) → list[str]`
Checks each feature column in `X_input` (shape `[n_rows, n_features]`) against `[min, max]` of `df_clean`.
Returns one warning string per out-of-range feature. For batch input, counts total rows outside range.
Called in `/api/predict` before running model predictions. Never blocks — warnings only.

### `get_pairplot_b64(df, columns, max_cols=8, outlier_df=None) → str`
Caps at 8 columns. Uses `pandas.plotting.scatter_matrix` (no seaborn dependency).
Optional `outlier_df`: DataFrame of outlier rows to overlay as hollow orange circles on every
off-diagonal subplot. Overlay is drawn post-scatter_matrix by iterating `axes[i,j]` and calling
`ax.scatter(facecolors='none', edgecolors='#f97316', ...)`. Passed from `/api/set_columns` using
rows collected *before* the exclude_outliers step, so the ghost overlay always shows.
The pairplot for a test dataset with constant-value columns will emit matplotlib warnings
about "identical left == right" — these are harmless.

### `get_outlier_flags(df, cols) → dict`
IQR-based outlier detection per column. Returns dict `{col: {row_indices, lo, hi, values}}`.
Only columns with at least one outlier are included. Skips non-numeric and zero-IQR (constant) columns.
Called in `/api/set_columns`. Frontend renders results in the outlier panel with `renderOutlierPanel()`.

### `get_correlation_heatmap_b64(df, feature_cols, threshold=0.92) → (plot_b64, high_corr_pairs)`
Pearson correlation heatmap for feature columns. `RdBu_r` colormap, annotated with r values.
`high_corr_pairs`: list of `{col_a, col_b, r}` for pairs with |r| ≥ threshold.
Returns `('', [])` when fewer than 2 numeric feature columns.

### `get_feat_target_grid_b64(df, feature_cols, target_cols, max_feat_cols=3) → (plot_b64, nonlinear_hint_cols)`
Feature vs target scatter grid with linear trend line (polyfit degree 1). Default cap 3; route passes `min(n, 6)`.
`nonlinear_hint_cols`: features whose linear R² < 0.7 against at least one target — scans ALL features, not just the plotted ones (second scan loop over `feat_cols[ncols:]` after the plot loop).
Message shown to engineer: "Linear fit is weak — may be non-linear or noisy." (not "non-linear detected")
`suptitle` uses `y=0.98` (not 1.01) to prevent clipping under tight_layout.

### `get_unusual_runs_b64(df, feature_cols, top_n=10) → (plot_b64, top_runs, all_scores)`
Isolation Forest multivariate anomaly detection; lollipop chart of top-N unusual rows.
- Uses `df[cols].dropna()` (not `fillna(mean)`) — imputing mean distorts anomaly scores
- `contamination = max(0.05, min(0.1, 5 / n))` — avoids forcing false positives on small datasets
- Preserves original DataFrame indices in row labels and `top_runs` list
- Returns `('', [], [])` when fewer than 2 numeric features or fewer than 8 clean rows
- `all_scores`: `[{row_idx, score}]` for every row (not just top-N) — used by `/api/data_scatter` for colour-by-unusualness

### `get_scatter_plot_b64(df, x_col, y_col, x_min, x_max, y_min, y_max, color_col, unusual_scores, x_log, y_log, max_points=500) → (plot_b64, n_filtered, n_plotted, n_color_missing, log_warning)`
Custom scatter plot for any two numeric columns.
- Applies range filters first (correct `filter → sample` ordering), then random-samples to `max_points`
- `color_col='unusual'` + `unusual_scores` dict: three-tier `#dc2626`/`#f97316`/`#2563EB` (>0.6 / 0.4–0.6 / ≤0.4); draws legend
- `color_col=<col_name>`: viridis colormap + colorbar; NaN values coloured `#94a3b8` (grey); returns `n_color_missing`
- Default (no colour): `#2563EB`, `alpha=0.55`, `s=18` — matches F2 aesthetic
- Log scale guard: if column has non-positive values, skips log scale and returns `log_warning` string
- Returns `('', 0, 0, 0, None)` when no rows remain after filtering

---

## Frontend Architecture (`index.html`)

Single-file SPA — inline `<style>` and `<script>`, no external dependencies.

### CSS Custom Properties
```css
:root {
  --accent: #2563EB;      /* blue — primary color */
  --accent-hover: #1d4ed8;
  --bg: #ffffff;
  --panel: #f8f9fa;
  --border: #e2e8f0;
  --text: #1e293b;
  --text-muted: #64748b;
}
```

### Step Navigation and Gating
```javascript
const appState = {
    featureCols: [], targetCols: [], summary: null,
    modelType: 'linear', trained: false, currentStep: 1,
    featureMedians: {},        // {col: float} — pre-filled from /api/train response
    explorationHistory: [],    // client-side only, max 20 entries {inputs, predictions}
    step1Flags: [],            // [{type, label, detail}] — populated by populateStep1Flags()
    nonlinearFeatures: [],     // feature names flagged as weak linear fit by loadDataExplorer()
    recommendedModel: null,    // {model, name, reason} — set by applyModelRecommendation(); used by onModelTypeChange() to detect overrides
    unusualScores: {},         // {row_idx: score} — populated from /api/unusual_runs all_scores; used by F3 colour-by-unusual
};
const STEP_GATES = {
    2: () => appState.featureCols.length > 0,
    3: () => appState.featureCols.length > 0,
    4: () => appState.trained,
};
function goToStep(n) { /* checks gate, swaps .active class, updates progress bar */ }
```

**Important:** Step 4 calls `buildStep4()` on navigation to dynamically generate the prediction form and download links. `buildStep4()` is called by overriding `window.goToStep`. The override also handles Step 2: populates dataset summary and fires `applyModelRecommendation()`.

### Key DOM Element IDs
| ID | Purpose |
|---|---|
| `upload-zone` | Drag-and-drop upload area |
| `file-input` | Hidden file input (triggered by upload-zone click) |
| `summary-panel` | Data summary table panel (hidden until upload) |
| `summary-table-wrapper` | Inner div where summary HTML table is rendered |
| `column-selection-panel` | Column checkboxes panel |
| `feature-checkboxes` | Container for feature column checkboxes (class `.feature-cb`) |
| `target-checkboxes` | Container for target column checkboxes (class `.target-cb`) |
| `confirm-columns-btn` | Triggers `/api/set_columns` |
| `pairplot-img` | `<img>` for scatter matrix |
| `step1-next-btn` | Disabled until columns confirmed |
| `model-type` | `<select>` — `linear`, `gpr`, or `rf` |
| `gpr-panel` | GPR-specific config (hidden for linear/rf) — blue tint |
| `rf-panel` | RF-specific config (hidden for linear/gpr) — green tint |
| `kernel-type` | `<select>` — `rbf` or `matern` |
| `length-scale` | GPR initial length scale input |
| `alpha` | GPR noise level input |
| `rf-max-depth` | RF max tree depth `<select>` (0=unlimited, 3, 5, 10) |
| `rf-min-samples` | RF min samples per leaf `<select>` (1, 2, 5) |
| `test-size` | Range slider for train/test split |
| `use-cv` | Checkbox for k-fold CV |
| `normalize` | Checkbox for StandardScaler |
| `train-btn` | Triggers training in Step 3 |
| `loading-spinner` | Shown during training fetch |
| `gpr-warning-banner` | Yellow banner if n_rows > 2000 with GPR |
| `history-panel` | Training history comparison table (hidden until first train) |
| `history-table-wrapper` | Inner div for history table HTML |
| `results-panel` | Dynamically populated with target sections after training |
| `step3-next-btn` | Disabled until training completes |
| `model-summary-content` | Step 4 model summary cards |
| `download-grid` | Step 4 model download links |
| `single-point-form` | Dynamically generated per-feature inputs |
| `extrap-warning` | Yellow warning banner for single-point extrapolation |
| `batch-extrap-warning` | Yellow warning banner for batch extrapolation |
| `batch-file-input` | CSV file input for batch prediction |
| `download-preds-link` | Shown after batch prediction completes |
| `upload-quality-msg` | Row-count quality alert shown immediately after upload |
| `dataset-health-card` | Feature-to-run ratio card shown after column confirmation |
| `step2-dataset-summary` | Compact "N runs · F features" line at top of Step 2 |
| `step2-row-count` / `step2-feature-count` | Populated when navigating to Step 2 |
| `model-recommendation-banner` | Auto-recommendation alert with plain-English reason |
| `rec-model-name` / `rec-reason` | Spans inside recommendation banner |
| `exploration-plot-panel` | SVG prediction explorer panel in Step 4 |
| `exploration-x-feature` | `<select>` for X-axis of exploration chart |
| `exploration-charts` | Container for per-target SVG exploration charts |
| `download-exploration-btn` | Shown after first prediction; downloads exploration_history.csv |
| `lc-caption-{target}` | Dynamic caption text below learning curve plot label |
| `help-metrics-{target}` | Metric help card per target (R², RMSE, MAE explanations) |
| `help-plot-{type}-{target}` | Per-plot per-target help cards (parity/residuals/importance/sens/surface/lc) |
| `data-explorer-section` | `<details class="panel">` — collapsible Data Explorer (sibling of `column-selection-panel`, shown after column confirm) |
| `de-loading` / `de-error` | Loading spinner and error banner inside Data Explorer |
| `de-corr-img` / `de-corr-msg` | F1: correlation heatmap image + warning message div |
| `de-ft-img` / `de-ft-msg` / `de-ft-caption` | F2: feature-target scatter image, weak-linearity hint, "Showing N of M" caption |
| `de-sc-x` / `de-sc-y` | F3: X-axis and Y-axis column selector dropdowns |
| `de-sc-color` | F3: colour-by dropdown (None / col / "Unusualness score" after detector runs) |
| `de-sc-xmin` / `de-sc-xmax` / `de-sc-ymin` / `de-sc-ymax` | F3: range filter inputs (debounced 500ms, pre-filled with data min/max as placeholder) |
| `de-sc-xlog` / `de-sc-ylog` | F3: log-scale checkboxes |
| `de-sc-img` / `de-sc-caption` / `de-sc-error` / `de-sc-placeholder` | F3: scatter image, caption, error, empty-state text |
| `de-ur-img` / `de-ur-top` / `de-ur-caveat` / `de-ur-legend` | F4: lollipop image, top-runs list, DoE caveat, colour legend |
| `doe-type-select` | F4: DoE type dropdown (grid / lhs / random) |
| `de-prospective` | Prospective hint div shown after F1/F2 load |

### Key JS Functions Added in UX Upgrade
| Function | Purpose |
|---|---|
| `toggleHelp(id)` | Show/hide any `.help-card` div by ID |
| `getRowCountMessage(nRows)` | Returns `{type, text}` upload quality message |
| `renderDatasetHealthCard(nRows, nFeatures)` | Renders feature-to-run ratio card into `#dataset-health-card` |
| `recommendModel(nRows, nFeatures)` | Returns `{model, name, reason}`; uses `appState.nonlinearFeatures` from Data Explorer; fixes 501–2000 row gap |
| `applyModelRecommendation(nRows, nFeatures)` | Applies recommendation to banner + dropdown; stores result in `appState.recommendedModel` |
| `updateTestSizeWarning(pct)` | Warns when test set < 5 rows given current `pct` and `appState.summary.shape[0]` |
| `updateCvKWarning()` | Warns when cv-k > floor(nRows/2) |
| `validateStep2()` | Guards Step 3 navigation — checks alpha>0, length_scale>0, cv-k validity; shows `#step2-config-error` |
| `renderModelHealth(results, targetCols)` | Returns HTML string for colour-coded health banner |
| `confidenceBadge(value, std)` | Returns HTML badge: High/Moderate/Low confidence with ±σ |
| `learningCurveCaption(d)` | Returns 1-line diagnostic caption from `final_train_r2`, `final_val_r2`, `val_still_rising` |
| `renderExplorationPlots()` | Rebuilds all SVG exploration charts from `appState.explorationHistory` |
| `clearExplorationHistory()` | Empties `appState.explorationHistory` and re-renders |
| `downloadExplorationHistory()` | Client-side CSV blob download from `appState.explorationHistory` |
| `populateStep1Flags()` | Checks target |skew|>1.5 and feature cv<0.01; pushes `{type, label, detail}` entries to `appState.step1Flags` |
| `distributionBadge(skew)` | Returns styled HTML span: ▶▶/▶/●/◀/◀◀ based on skew value |
| `coverageBadge(cv)` | Returns "barely varies" badge span if cv < 0.01, else `''` |
| `loadDataExplorer()` | Fetches `/api/data_explorer`; populates F1/F2; bridges high_corr_pairs to `step1Flags`; shows prospective hint; calls `populateScatterDropdowns()` for F3. Duplicate-call-safe. |
| `populateScatterDropdowns()` | Fills F3 X/Y/colour-by dropdowns with all feature+target cols; defaults Y to second col; calls `updateScatterRangeHints()` |
| `updateScatterRangeHints()` | Updates F3 range input placeholders with data min/max from `appState.summary.stats` for the currently selected X/Y columns |
| `runUnusualDetector()` | Fetches `/api/unusual_runs?doe_type=…`; renders F4 lollipop, legend, caveat, top-runs table; stores `appState.unusualScores`; appends "Unusualness score" option to F3 colour-by dropdown |
| `runScatterPlot()` | Fetches `/api/data_scatter`; validates same-col/inverted-range; AbortController cancels in-flight; opacity-fade loading; renders F3 image + caption |
| `debouncedScatterPlot()` | 500ms debounce wrapper for `runScatterPlot()`; only fires if F3 image is already visible (range filter re-plot, not initial) |
| `renderRetrospectivePanel()` | Returns violet `.retro-panel` HTML from `appState.step1Flags` with inline "← Go back to Step 1" button. Inserted at top of `renderResults()`. |

### Sensitivity Plot Flow
1. User selects a feature from `<select class="sensitivity-feature-select" data-target="{target}">`
2. `loadSensitivity(selectEl)` fires on `change`
3. `buildRefParams()` collects any filled values from Step 4 `inp-{col}` inputs → appended as `&ref_{col}=value`
4. Fetches `/api/sensitivity?feature=X&target=Y[&ref_col=val...]`
5. Sets `img#sensitivity-plot-{target}.src` to the returned base64 PNG

### 2D Surface Plot Flow
1. User selects X-axis and Y-axis features from two `<select>` elements (`surface-x-select`, `surface-y-select`)
2. `loadSurface(target)` fires on `change` of either select
3. Fetches `/api/surface?feature_x=X&feature_y=Y&target=Z`
4. Shows "Computing…" spinner; sets `img#surface-plot-{target}.src` when done
5. Only rendered when `data.feature_cols.length >= 2`

### Model Comparison History Flow
1. After training, `/api/train` response includes `train_history` (last 3 runs)
2. `renderHistory(history, targetCols)` builds a table with columns: Run, Model, Kernel, Time, R² per target
3. Current run row has class `current-row` (blue highlight, bold, ★ label)
4. History panel hidden until first training run

### Learning Curve Flow
1. User clicks "Show Learning Curve" button per target in Step 3
2. `loadLearningCurve(target)` fetches `/api/learning_curve?target=Y`
3. Response includes `plot_b64`, `final_train_r2`, `final_val_r2`, `val_still_rising`
4. `learningCurveCaption(data)` generates a 1-line diagnostic; set into `#lc-caption-{target}`

### Prediction Explorer Flow (Step 4)
1. After `predictSingle()` succeeds, entry `{inputs, predictions}` is pushed to `appState.explorationHistory`
2. `renderExplorationPlots()` is called; renders one inline SVG per target
3. SVG shows: blue band (training min/max), dashed mean line, dots per prediction, ±1σ error bars
4. X-axis: exploration sequence by default; any feature column via `#exploration-x-feature` select
5. Max 20 entries — oldest dropped when full; `clearExplorationHistory()` resets
6. `downloadExplorationHistory()` builds CSV blob client-side (no server route)

### Data Explorer Flow (Step 1, after column confirm)
Framing line: "Four checks before you train: (1) redundant inputs? (2) non-linear? (3) investigate further? (4) suspicious runs?"

1. `confirmColumns()` calls `populateStep1Flags()`; shows `data-explorer-section`; resets F3 scatter panel and `appState.unusualScores = {}`
2. User expands `<details>` → `toggle` fires `loadDataExplorer()` (duplicate-safe: returns early if already loading)
3. `loadDataExplorer()` fetches `/api/data_explorer` — server cached on 2nd+ open; calls `populateScatterDropdowns()` to fill F3 dropdowns
4. F1: heatmap shown; if `high_corr_pairs` non-empty: warning with "← Uncheck a column" link; pairs pushed to `appState.step1Flags`
5. F2: scatter shown; caption "Showing N of M features. Linear fit checked for all M."; nonlinear features stored in `appState.nonlinearFeatures`
6. F3: X/Y/colour dropdowns populated (all feature+target cols); range inputs pre-filled with data min/max as placeholders; user clicks "Plot" for initial render; range input changes debounce-re-plot if image visible
7. F4: user selects DoE type then clicks "Run Detector" → `runUnusualDetector()` fetches `/api/unusual_runs`; renders lollipop + colour legend + caveat + flagged-row feature table; stores `appState.unusualScores`; adds "Unusualness score" option to F3 colour-by dropdown
8. After training (Step 3): `renderResults()` prepends `renderRetrospectivePanel()` — shows all `appState.step1Flags` with "← Go back to Step 1" button

### Plot Rendering Pattern (all plots)
```javascript
imgElement.src = 'data:image/png;base64,' + data.some_b64_field;
```

---

## Git History

```
(latest)  feat: Data Explorer Option B — custom scatter (F3 panel)
          (new "Investigate Any Two Variables" panel between F2 and F4;
           X/Y dropdowns, colour-by dropdown (None/col/unusualness), Plot button;
           range filters debounced 500ms, log X/Y checkboxes;
           filter-then-sample ordering; AbortController for race conditions;
           opacity-fade loading state; client-side validation;
           /api/data_scatter route; get_scatter_plot_b64() in data_utils;
           de_unusual_scores STATE key; all_scores added to /api/unusual_runs;
           framing line updated to 4 checks; F3 reset on column re-confirm + upload)
          feat: Data Explorer Option A — correctness + UX fixes (C1-C4, U1-U4)
          (C1: nonlinear scan covers unplotted features; C2: "Showing N of M" caption on F2;
           C3/C4: unusual run detector flagged-row feature table with IQR cell highlighting,
           updated action text, "Showing top N of M runs" count;
           U1: framing line "Three/Four checks before you train";
           U2: colour legend below lollipop chart; U4: F2 scatter shows up to 6 features)
          feat: Configure tab hyperparameter definitions — 13 expert-reviewed clarity fixes
          (D1: RF depth "high bias"/"low bias" → "may miss sharp gradients"/"may fail on unseen points";
           D2: Min Samples hint removes "leaf" jargon → "CFD runs in each prediction region";
           D3: Normalize label removes "StandardScaler" → "(recommended)";
           D4: Alpha hint rewritten: "keeps GPR math solver stable"; adds "values above 0.1 smooth predictions";
           D5: RF info box removes MDI parenthetical → "feature importance scores show which inputs most affect predictions";
           G1: Length scale hint adds "If accuracy poor, don't change this — check data/Data Explorer";
           G3: Normalize card adds "Leave checked — unchecking almost never beneficial";
           G4: k-fold card adds "k=3 for <20 runs; k=5 for 50+";
           G5: Panel gets "These settings control how the model learns..." descriptor;
           G6: Max depth hint rewritten to action voice;
           T1: Comparison table adds "Change from default when:" row;
           T2: Table footnote: "For smoothly varying aerodynamic data, linear models can perform as well as GPR")
          feat: Configure tab (Step 2) expert review — 18 fixes across bugs, UX, validation, and intelligence
          (B1: alpha step=any value=1e-6; B2: RF info box blue + "Will train 200 trees"; B3: Linear accuracy "High if data is linear";
           B4: CV card names R² metric; U1: "numerical regularisation" label + hint; U2: redundant normalize hint removed;
           U3: GPR/RF panel headers; U4: RF Unlimited label warns overfitting; U5: kernel card auto-optimize note;
           U6: "✓ (100 resamples)"; V1: test-set size warning; V2: cv-k > nRows/2 warning;
           V3: validateStep2() blocks Step 3 nav on invalid inputs; I1: recommendModel uses nonlinearFeatures;
           I2: GPR time estimate shown when >200 rows; I3: banner updates on manual model override;
           E4-C: dataset summary numbers bold+accent color; length-scale hint references std-deviation scale)
          fix: Data Explorer expert review — bug fixes, caching, UX improvements
          (B1: IsolationForest dropna not fillna, original indices preserved; B2: weak-linearity wording;
           B3: suptitle y=0.98; B4: adaptive contamination; P1: de_ cache in APP_STATE;
           U1: data-explorer-section moved outside column panel; U2: retro panel back-link;
           U3: high-corr warning with Uncheck button; port 5000→5001 doc fix)
775d918   docs: update CLAUDE.md for Round 2 UX changes
0e0cbc5   feat: Round 2 UX — outlier pairplot overlay, Step 2 help cards, sensitivity range display, grey exploration chart
          (pairplot orange ghost overlay; 5x ? cards in Step 2; sensitivity training-range lines+text;
           reference inputs open by default with min-max hints; exploration chart grey #e2e8f0)
          feat: aerospace engineer UX upgrade — self-teaching surrogate tool
b78f6f6   feat: aerospace engineer UX upgrade — self-teaching surrogate tool
          (onboarding panel, auto-recommendation, model health banner, plot captions,
           ? cards, confidence badges, prediction explorer, dynamic LC caption)
          (prior commits covered Tier 2+3: RF, GPR, learning curves, bootstrap uncertainty,
           outlier detection, 2D surface, extrapolation warnings, model history)
5f1fd38   feat: add four additional sample datasets with generator scripts
cd297ee   feat: add NACA 0012 sample dataset (96 rows) and generator script for tutorial use
c8050b1   docs: add CLAUDE.md with full codebase context for future development sessions
9c77b1b   docs: complete README with Anaconda launch instructions and usage guide
4c2061b   feat: 4-step wizard UI with training results, sensitivity, and prediction forms
```

---

## Known Issues and Gotchas

1. **Flask 1.x vs 2.x `send_file`**: Current code uses `attachment_filename=` (Flask 1.x).
   If Flask is upgraded, change all 4 `send_file` calls in `routes.py` to `download_name=`.

2. **Anaconda DLL issue**: Running `python` directly from Git Bash or PowerShell (not Anaconda Prompt)
   causes numpy DLL import failure. Always run from Anaconda Prompt or use `conda run -n base python ...`
   for testing.

3. **Pairplot warnings**: If a selected column has zero variance (all values identical), matplotlib
   emits "identical left == right" warnings. These are harmless — the plot still renders.

4. **GPR std bypass**: `sklearn.pipeline.Pipeline.predict()` does not forward `return_std=True`
   to the underlying GPR. Any code that needs GPR uncertainty must extract the estimator directly:
   `pipeline.named_steps['model']` and transform input separately via `pipeline.named_steps['scaler']`.
   This pattern is used in both `/api/predict` and `get_sensitivity_plot_b64()` and `get_surface_plot_b64()`.

5. **CV data for cross_val_score**: CV is run on `np.vstack([X_train, X_test])` with a fresh
   pipeline clone — NOT the already-fitted pipeline. This is intentional to avoid data leakage.

6. **Train/test split for multi-output**: `train_test_split(X, *y_dict.values(), ...)` splits
   all arrays with the same random indices. Results are unpacked as `y_splits[i*2]` for train
   and `y_splits[i*2+1]` for test per target. This is correct Python tuple unpacking.

7. **ARD length scale key**: After GPR fit, the optimized length scales are at:
   `gpr.kernel_.get_params()['k2__length_scale']`
   The `k2` prefix is because the kernel is `ConstantKernel * base_kernel` — k1 is the constant,
   k2 is the RBF/Matern. If you change the kernel composition this key will break.

8. **`_reset_downstream()` must be called on new upload**: Forgetting to call it means stale
   training results from a previous dataset will persist and be served incorrectly.
   `_reset_downstream('upload')` also clears `train_history`.

9. **RF + normalize**: StandardScaler is applied to RF input when `normalize=True`, but tree
   splits are scale-invariant so it has no effect on RF predictions. It's kept for pipeline
   uniformity (the same scaler object is reused in `get_surface_plot_b64` for consistency).

10. **2D surface `feature_x == feature_y`**: The `/api/surface` route returns a 400 error if
    both features are the same. The frontend `loadSurface()` guard (`featureX === featureY`)
    prevents the fetch, but the backend check is also there as a safety net.

---

## How to Test Changes

Run the smoke test pattern using conda:
```bash
# From project root in Anaconda Prompt:
conda activate base
python -c "
import matplotlib; matplotlib.use('Agg')
from app import create_app
app = create_app()
# ... test code here
"
```

Or write a test script and run:
```bash
conda run -n base python my_test.py
```

For a full end-to-end test, create a CSV with at least 10 rows, 2+ numeric feature columns,
and 1+ numeric target columns, then use the browser UI.

**Verification checklist (full UX upgrade including Round 2):**
1. Open tool — confirm improved subtitle visible; "What is a surrogate model?" panel expands/collapses
2. Upload NACA 0012 (96 rows) — confirm green "good dataset size" message appears immediately
3. Upload a tiny dataset (< 20 rows) — confirm red warning appears
4. Confirm columns — confirm Dataset Health card renders with ratio assessment; outlier `?` card works
5. If outliers flagged: confirm pairplot shows hollow orange circles on flagged rows
6. Toggle "Exclude flagged rows" ON → confirm pairplot regenerates with orange circles still shown on excluded rows
7. Navigate to Step 2 — confirm "Dataset:" numbers are bold blue; GPR pre-selected; recommendation banner shows correct reason
8. Switch model type manually (e.g. Linear) — confirm banner updates to show override message with original recommendation
9. Switch to GPR with 96 rows — confirm orange time-estimate banner appears ("~X seconds")
10. Click `?` on "Model Type" → confirm comparison table: Linear accuracy reads "High if data is linear"; bootstrap cell reads "✓ (100 resamples)"
11. Select GPR → confirm "GPR Hyperparameters" section header visible; alpha input shows "1e-6" not "0.000001"; alpha label reads "numerical regularisation"
12. Click `?` on "Kernel" → confirm new sentence about both kernels being auto-optimized
13. Confirm length scale hint references "standard deviation" scale after normalisation
14. Select RF → confirm "RF Hyperparameters" section header; info box is blue (not green); reads "Will train 200 trees"; Unlimited option warns "overfits unless data is large and clean"
15. Confirm no duplicate hint text below "Normalize inputs" checkbox (hint removed; ? card remains)
16. Set test split to 10% with 10-row dataset → confirm small-test-set warning appears
17. Enable k-fold, set k=8 with 10-row dataset → confirm k-too-large warning appears
18. Clear alpha field, click "Next: Train Model" → confirm validation error shown, navigation blocked
19. Click `?` on "Enable k-fold Cross-Validation" → confirm "CV score = average R²" is explicitly stated
20. Click `?` on "Train / Test Split" → confirm proof-test explanation with 96-run example
21. Click `?` on "Normalize inputs" → confirm scale explanation with AoA vs Mach example
22. Train GPR — confirm model health banner shows green "Good fit"; R² card is green; gap shown
23. Train Linear (expect poor fit) — confirm health banner amber/red with next steps listed
24. Click each plot's `?` button — confirm help card expands and collapses correctly
25. Click "Show Learning Curve" — confirm dynamic caption appears below plot label
26. Sensitivity: select a feature → confirm "Training range: X — Y" appears below dropdown
27. Confirm reference inputs `<details>` is open by default and each input shows min–max range hint
28. Confirm sensitivity plot has two grey dashed boundary lines and faint orange shading outside bounds
29. Go to Step 4 — enter prediction inside training range (GPR) — confirm confidence badge is green
30. Enter value outside training range — confirm low-confidence badge + extrapolation warning both appear
31. Enter 5 predictions — confirm exploration plot has grey (#e2e8f0) training band (not blue)
32. Confirm legend reads "Grey band = training data range"
33. Change X-axis dropdown to a feature — confirm plot re-renders in feature space
34. Click "⬇ Download History" — confirm CSV has correct columns and values
35. Click "Clear" — confirm plot resets and download button hides
36. Step 3: Train RF → verify RF info panel is blue and reads "Will train 200 trees"; 2D surface selects appear
37. After column confirm — confirm "Explore Data Relationships" appears as a separate panel BELOW the column panel (not inside it)
38. Expand Data Explorer → confirm F1 heatmap and F2 scatter load automatically; loading spinner disappears
39. Close and re-open Data Explorer → confirm no loading spinner (cached — instant)
40. Re-confirm columns → confirm Data Explorer collapses and re-opening triggers a fresh fetch
41. F2: if NACA 0012 shows non-linear features → confirm message reads "Linear fit is weak — may be non-linear or noisy"; recommendation banner in Step 2 references those features
42. F1: if high correlation detected → confirm warning has "← Uncheck a column" button
43. F4: select "Structured grid" + click "Run Detector" → confirm DoE caveat appears and lollipop chart shows
44. F4: confirm colour legend appears below lollipop (red/orange/blue dots); confirm "X run(s) above the review threshold" text; confirm feature table with IQR-highlighted cells if any rows >0.6; confirm "Showing top N of M runs" count
45. F2: confirm "Showing all N input features" or "Showing N of M input features. Linear fit checked for all M." caption below scatter
46. F3: expand Data Explorer → confirm "Investigate Any Two Variables" section appears between F2 and F4
47. F3: X/Y dropdowns populated with all columns; Y defaults to second column; range inputs show data min/max as placeholder
48. F3: select X=AoA, Y=AoA → click Plot → confirm "Select different columns" error appears
49. F3: select two different columns → click Plot → confirm scatter renders; caption shows row count
50. F3: set X max < X min → confirm "X min must be less than X max" error before fetch
51. F3: narrow range filter → confirm caption shows "Showing 50 of 200 rows in this range (random sample)" style text
52. F3: run Unusual Detector first → confirm "Unusualness score (from detector ↑)" appears in colour-by dropdown
53. F3: colour by unusualness → confirm plot has red/orange/blue points with legend; caption explains colour tiers
54. F3: colour by a numeric column → confirm viridis colorbar appears on plot
55. F3: re-confirm columns → confirm F3 image clears, dropdowns reset, unusualness option removed from colour-by
56. Train any model → confirm Step 3 retrospective panel shows Step 1 flags AND has "← Go back to Step 1" button

---

## Deferred Features (Next Iteration)

- **Flask 2.x upgrade**: Change all 4 `attachment_filename=` → `download_name=` in `routes.py`.

- **Batch prediction confidence**: `predictBatch()` currently shows row count only; could show per-row std columns in the downloaded CSV (already computed server-side, just not surfaced in batch response).

- **Step 3 sensitivity reference live update**: Reference inputs in the Step 3 sensitivity panel already work (`sens-ref-{target}-{col}` inputs). Could add a "Update all sensitivity plots" button to reload all visible plots at once after changing multiple reference values.

- **Multi-target exploration chart layout**: When 3+ targets are selected, exploration charts stack vertically. Could move to a 2-column grid for space efficiency.

- **Data Explorer: "Remove column" action**: High-correlation warning could uncheck the suggested column's checkbox directly. Requires two-way binding between the Data Explorer and the column-selection checkboxes — more involved refactor.

- **F3 scatter: "Download this view" button**: Add `format=csv` query param to `/api/data_scatter`; return filtered+sampled DataFrame as CSV. Reuses existing filter logic.

- **F3 scatter: "Investigate in scatter" shortcuts from F1/F2**: After F1/F2 load, add "Plot this pair →" buttons next to high-correlation pair warnings and weak-linearity messages that pre-populate F3 dropdowns and auto-render.

- **F3 scatter: Coverage gap detection**: After plotting, compute axis histogram; if any contiguous gap > 15% of axis range has zero data, add note: "Coverage gap: no data for [col] between [X] and [Y]."
