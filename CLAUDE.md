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
`stats` only populated for numeric columns: `{col: {min, max, mean}}`.

### `clean_data(df, feature_cols, target_cols) → (df_clean, n_dropped)`
Selects only the specified columns, drops rows with any NaN, resets index.

### `check_extrapolation(X_input, df_clean, feature_cols) → list[str]`
Checks each feature column in `X_input` (shape `[n_rows, n_features]`) against `[min, max]` of `df_clean`.
Returns one warning string per out-of-range feature. For batch input, counts total rows outside range.
Called in `/api/predict` before running model predictions. Never blocks — warnings only.

### `get_pairplot_b64(df, columns, max_cols=8) → str`
Caps at 8 columns. Uses `pandas.plotting.scatter_matrix` (no seaborn dependency).
The pairplot for a test dataset with constant-value columns will emit matplotlib warnings
about "identical left == right" — these are harmless.

### `get_outlier_flags(df, cols) → dict`
IQR-based outlier detection per column. Returns dict `{col: {row_indices, lo, hi, values}}`.
Only columns with at least one outlier are included. Skips non-numeric and zero-IQR (constant) columns.
Called in `/api/set_columns`. Frontend renders results in the outlier panel with `renderOutlierPanel()`.

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

### Key JS Functions Added in UX Upgrade
| Function | Purpose |
|---|---|
| `toggleHelp(id)` | Show/hide any `.help-card` div by ID |
| `getRowCountMessage(nRows)` | Returns `{type, text}` upload quality message |
| `renderDatasetHealthCard(nRows, nFeatures)` | Renders feature-to-run ratio card into `#dataset-health-card` |
| `recommendModel(nRows, nFeatures)` | Returns `{model, name, reason}` recommendation object |
| `applyModelRecommendation(nRows, nFeatures)` | Applies recommendation to banner + dropdown |
| `renderModelHealth(results, targetCols)` | Returns HTML string for colour-coded health banner |
| `confidenceBadge(value, std)` | Returns HTML badge: High/Moderate/Low confidence with ±σ |
| `learningCurveCaption(d)` | Returns 1-line diagnostic caption from `final_train_r2`, `final_val_r2`, `val_still_rising` |
| `renderExplorationPlots()` | Rebuilds all SVG exploration charts from `appState.explorationHistory` |
| `clearExplorationHistory()` | Empties `appState.explorationHistory` and re-renders |
| `downloadExplorationHistory()` | Client-side CSV blob download from `appState.explorationHistory` |

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

### Plot Rendering Pattern (all plots)
```javascript
imgElement.src = 'data:image/png;base64,' + data.some_b64_field;
```

---

## Git History

```
(latest)  feat: aerospace engineer UX upgrade — self-teaching surrogate tool
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

**Verification checklist (full UX upgrade):**
1. Open tool — confirm improved subtitle visible; "What is a surrogate model?" panel expands/collapses
2. Upload NACA 0012 (96 rows) — confirm green "good dataset size" message appears immediately
3. Upload a tiny dataset (< 20 rows) — confirm red warning appears
4. Confirm columns — confirm Dataset Health card renders with ratio assessment; outlier `?` card works
5. Navigate to Step 2 — confirm compact "Dataset: 96 rows · 4 input features" visible; GPR pre-selected; recommendation banner shows correct reason
6. Switch model type manually — confirm banner text doesn't change but dropdown follows
7. Train GPR — confirm model health banner shows green "Good fit"; R² card is green; gap shown
8. Train Linear (expect poor fit) — confirm health banner amber/red with next steps listed
9. Click each plot's `?` button — confirm help card expands and collapses correctly
10. Click "Show Learning Curve" — confirm dynamic caption appears below plot label
11. Go to Step 4 — enter prediction inside training range (GPR) — confirm confidence badge is green
12. Enter value outside training range — confirm low-confidence badge + extrapolation warning both appear
13. Enter 5 predictions — confirm exploration plot builds up with shaded training band visible
14. Change X-axis dropdown to a feature — confirm plot re-renders in feature space
15. Click "⬇ Download History" — confirm CSV has correct columns and values
16. Click "Clear" — confirm plot resets and download button hides
17. Step 3: Check outlier `?` card explains IQR, why it matters, when to exclude
18. Step 3: Train RF → verify "200 trees" info panel visible; 2D surface selects appear

---

## Deferred Features (Next Iteration)

- **Flask 2.x upgrade**: Change all 4 `attachment_filename=` → `download_name=` in `routes.py`.

- **Batch prediction confidence**: `predictBatch()` currently shows row count only; could show per-row std columns in the downloaded CSV (already computed server-side, just not surfaced in batch response).

- **Step 3 sensitivity reference live update**: Reference inputs in the Step 3 sensitivity panel already work (`sens-ref-{target}-{col}` inputs). Could add a "Update all sensitivity plots" button to reload all visible plots at once after changing multiple reference values.

- **Multi-target exploration chart layout**: When 3+ targets are selected, exploration charts stack vertically. Could move to a 2-column grid for space efficiency.
