import base64
import io
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ACCENT = '#2563EB'
N_BOOTSTRAP = 100
RF_N_ESTIMATORS = 200
GPR_N_RESTARTS = 5
SURFACE_N_GRID = 30
PLOT_LABEL_SIZE = 10
PLOT_TICK_SIZE = 9
PLOT_ANNOT_SIZE = 9
PLOT_TITLE_SIZE = 12
PLOT_TIGHT_PAD = 1.5


# ---------------------------------------------------------------------------
# Kernel and pipeline construction
# ---------------------------------------------------------------------------

def build_kernel(kernel_type, n_features, length_scale):
    ls_vec = np.ones(n_features) * float(length_scale)
    ls_bounds = (1e-3, 1e3)
    amplitude = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))

    if kernel_type == 'rbf':
        base = RBF(length_scale=ls_vec, length_scale_bounds=ls_bounds)
    elif kernel_type == 'matern':
        base = Matern(length_scale=ls_vec, length_scale_bounds=ls_bounds, nu=1.5)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    return amplitude * base


def build_pipeline(model_type, kernel, alpha, normalize, config=None):
    steps = []
    if normalize:
        steps.append(('scaler', StandardScaler()))

    if model_type == 'gpr':
        estimator = GaussianProcessRegressor(
            kernel=kernel,
            alpha=float(alpha),
            normalize_y=True,
            n_restarts_optimizer=GPR_N_RESTARTS,
            random_state=42,
        )
    elif model_type == 'rf':
        cfg = config or {}
        max_depth = cfg.get('max_depth', None)
        min_samples_leaf = int(cfg.get('min_samples_leaf', 1))
        estimator = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=max_depth if max_depth and int(max_depth) > 0 else None,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
    else:
        estimator = LinearRegression()

    steps.append(('model', estimator))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    return {
        'rmse': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        'r2': round(float(r2_score(y_true, y_pred)), 6),
        'mae': round(float(mean_absolute_error(y_true, y_pred)), 6),
    }


# ---------------------------------------------------------------------------
# Uncertainty helpers
# ---------------------------------------------------------------------------

def _gpr_std(pipeline, X):
    """Return GPR predictive std, bypassing Pipeline (which doesn't forward return_std)."""
    gpr = pipeline.named_steps['model']
    X_s = pipeline.named_steps['scaler'].transform(X) if 'scaler' in pipeline.named_steps else X
    _, std = gpr.predict(X_s, return_std=True)
    return std


def _rf_std(pipeline, X, floor=None):
    """Return std of individual tree predictions for RF.

    floor: minimum std value — prevents artificially zero uncertainty on extrapolation
    where all trees return the same leaf boundary.
    """
    rf = pipeline.named_steps['model']
    X_s = pipeline.named_steps['scaler'].transform(X) if 'scaler' in pipeline.named_steps else X
    tree_preds = np.array([t.predict(X_s) for t in rf.estimators_])
    std = tree_preds.std(axis=0)
    if floor is not None:
        std = np.maximum(std, floor)
    return std


def _bootstrap_std(bootstrap_models, X):
    """Return std of bootstrap model predictions for linear regression."""
    if not bootstrap_models:
        return None
    boot_preds = np.array([m.predict(X) for m in bootstrap_models])
    return boot_preds.std(axis=0)


# ---------------------------------------------------------------------------
# Plot helpers — all return base64-encoded PNG strings
# ---------------------------------------------------------------------------

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def get_parity_plot_b64(y_true, y_pred, target_name):
    """Returns (b64, caption_str)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')

    ax.scatter(y_true, y_pred, s=30, color=ACCENT, alpha=0.7,
               edgecolors='white', linewidths=0.5)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    lims = [lo - pad, hi + pad]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
    ax.text(lims[1] - (lims[1] - lims[0]) * 0.03, lims[1] - (lims[1] - lims[0]) * 0.03,
            'perfect fit', fontsize=8, color='#9ca3af', ha='right', va='top')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f} (test)',
            transform=ax.transAxes, va='top', fontsize=9, color='#374151')

    ax.set_xlabel(f'Actual {target_name}')
    ax.set_ylabel(f'Predicted {target_name}')
    ax.set_title(f'Parity Plot — {target_name}')
    ax.set_facecolor('white')
    plt.tight_layout()

    if r2 > 0.95:
        caption = 'Predictions closely match CFD results.'
    elif r2 >= 0.70:
        caption = 'Reasonable fit — some scatter present.'
    else:
        caption = 'Large scatter — model may not capture all response variation.'

    return _fig_to_b64(fig), caption


def get_residuals_plot_b64(y_true, y_pred, target_name):
    """Returns (b64, caption_str)."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('white')

    ax1.scatter(y_pred, residuals, s=30, color=ACCENT, alpha=0.7,
                edgecolors='white', linewidths=0.5)
    ax1.axhline(0, color='k', linestyle='--', lw=1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Residual')
    ax1.set_title('Residuals vs Predicted')
    ax1.set_facecolor('white')

    n_bins = max(5, min(20, len(residuals) // 2 + 1))
    ax2.hist(residuals, bins=n_bins, color=ACCENT, edgecolor='white')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Count')
    ax2.set_title('Residual Distribution')
    ax2.set_facecolor('white')

    plt.tight_layout()

    abs_res = np.abs(residuals)
    mean_abs = abs_res.mean()
    if mean_abs > 0 and abs_res.max() > 3 * mean_abs:
        caption = 'Outlier runs dominate the error — check the Unusual Runs detector.'
    else:
        caption = 'Errors spread evenly — no systematic bias detected.'

    return _fig_to_b64(fig), caption


def get_feature_importance_plot_b64(pipeline, feature_names, model_type, target_name):
    n = len(feature_names)
    fig_h = max(3, n * 0.5 + 1)
    fig, ax = plt.subplots(figsize=(6, fig_h))
    fig.patch.set_facecolor('white')

    if model_type == 'linear':
        coef = np.abs(pipeline.named_steps['model'].coef_)
        importance = coef / (coef.sum() + 1e-12)
        xlabel = 'Normalized |Coefficient|'
        title = f'Feature Importance — {target_name}'
    elif model_type == 'rf':
        importance = pipeline.named_steps['model'].feature_importances_
        xlabel = 'Mean Decrease in Impurity'
        title = f'Feature Importance (RF) — {target_name}'
    else:  # gpr
        gpr = pipeline.named_steps['model']
        ls = gpr.kernel_.get_params().get('k2__length_scale', None)
        if ls is None:
            ls = np.ones(n)
        ls = np.atleast_1d(ls)
        importance = 1.0 / (ls + 1e-12)
        importance = importance / (importance.sum() + 1e-12)
        xlabel = 'Relative Importance (1 / length_scale)'
        title = f'Feature Relevance (ARD) — {target_name}'

    sorted_idx = np.argsort(importance)
    sorted_vals = importance[sorted_idx]
    ax.barh(range(n), sorted_vals, color=ACCENT, edgecolor='white')
    ax.set_xlim(0, sorted_vals.max() * 1.25)
    for i, val in enumerate(sorted_vals):
        ax.text(val + sorted_vals.max() * 0.02, i, f'{val*100:.1f}%',
                va='center', fontsize=PLOT_ANNOT_SIZE, color='#374151')
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_facecolor('white')
    plt.tight_layout()
    return _fig_to_b64(fig)


def get_sensitivity_plot_b64(pipeline, X_ref, feature_names, feature_idx,
                              target_name, model_type, x_sweep,
                              bootstrap_models=None, train_lo=None, train_hi=None):
    """1D sensitivity plot with ±1σ uncertainty band for all model types.

    train_lo/train_hi: training data min/max for the swept feature — drawn as
    vertical boundary lines with faint orange extrapolation shading outside.
    """
    X_sweep = np.tile(X_ref, (len(x_sweep), 1))
    X_sweep[:, feature_idx] = x_sweep

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('white')

    y_pred = pipeline.predict(X_sweep)

    if model_type == 'gpr':
        y_std = _gpr_std(pipeline, X_sweep)
        label_std = '±1σ (GPR)'
    elif model_type == 'rf':
        y_std = _rf_std(pipeline, X_sweep)
        label_std = '±1σ (tree variance)'
    else:  # linear
        y_std = _bootstrap_std(bootstrap_models, X_sweep)
        label_std = '±1σ (bootstrap)'

    ax.plot(x_sweep, y_pred, color=ACCENT, lw=2, label='Mean')
    if y_std is not None:
        ax.fill_between(x_sweep, y_pred - y_std, y_pred + y_std,
                        alpha=0.25, color=ACCENT, label=label_std)
        ax.legend(fontsize=9)

    ref_val = float(X_ref[0, feature_idx])
    ax.axvline(ref_val, color='grey', linestyle='--', lw=1, alpha=0.7, label='reference')
    ax.text(ref_val, 1.0, f'ref={ref_val:.3g}',
            transform=ax.get_xaxis_transform(), ha='center', va='bottom',
            fontsize=PLOT_ANNOT_SIZE, color='#64748b')

    # Training data boundary lines + faint extrapolation shading
    if train_lo is not None:
        ax.axvline(train_lo, color='#94a3b8', linestyle='--', lw=1, alpha=0.7,
                   label='training min')
    if train_hi is not None:
        ax.axvline(train_hi, color='#94a3b8', linestyle='--', lw=1, alpha=0.7,
                   label='training max')
    if train_lo is not None and train_hi is not None:
        xmin, xmax = ax.get_xlim()
        ax.axvspan(xmin, train_lo, alpha=0.07, color='#f97316', zorder=0)
        ax.axvspan(train_hi, xmax, alpha=0.07, color='#f97316', zorder=0)
        ax.set_xlim(xmin, xmax)  # restore after axvspan may expand limits

    ax.set_xlabel(feature_names[feature_idx])
    ax.set_ylabel(target_name)
    ax.set_title(f'1D Sensitivity: {target_name} vs {feature_names[feature_idx]}')
    ax.set_facecolor('white')
    plt.tight_layout()
    return _fig_to_b64(fig)


def get_surface_plot_b64(pipeline, X_ref, feature_names, idx_x, idx_y,
                          target_name, model_type, x_range, y_range,
                          n_grid=SURFACE_N_GRID, X_train=None):
    """2D response surface. For GPR: side-by-side mean + σ. Others: mean only.

    X_train: optional training data array; when provided, actual data points are
    overlaid as white dots so engineers can see where real CFD data exists.
    """
    x_vals = np.linspace(x_range[0], x_range[1], n_grid)
    y_vals = np.linspace(y_range[0], y_range[1], n_grid)
    xx, yy = np.meshgrid(x_vals, y_vals)

    X_grid = np.tile(X_ref, (n_grid * n_grid, 1))
    X_grid[:, idx_x] = xx.ravel()
    X_grid[:, idx_y] = yy.ravel()

    def _overlay_training(ax):
        if X_train is not None:
            ax.scatter(X_train[:, idx_x], X_train[:, idx_y],
                       c='white', edgecolors='black', s=14, zorder=5, linewidths=0.5)

    if model_type == 'gpr':
        gpr = pipeline.named_steps['model']
        X_s = pipeline.named_steps['scaler'].transform(X_grid) if 'scaler' in pipeline.named_steps else X_grid
        z_mean, z_std = gpr.predict(X_s, return_std=True)
        z_mean = z_mean.reshape(n_grid, n_grid)
        z_std = z_std.reshape(n_grid, n_grid)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('white')

        cf1 = ax1.contourf(xx, yy, z_mean, levels=20, cmap='Blues')
        ax1.contour(xx, yy, z_mean, levels=10, colors='white', linewidths=0.4, alpha=0.5)
        _overlay_training(ax1)
        plt.colorbar(cf1, ax=ax1, label=f'Predicted {target_name}')
        ax1.set_xlabel(feature_names[idx_x])
        ax1.set_ylabel(feature_names[idx_y])
        ax1.set_title(f'Mean Prediction — {target_name}')
        ax1.set_facecolor('white')

        cf2 = ax2.contourf(xx, yy, z_std, levels=20, cmap='Oranges')
        _overlay_training(ax2)
        plt.colorbar(cf2, ax=ax2, label='σ')
        ax2.set_xlabel(feature_names[idx_x])
        ax2.set_ylabel(feature_names[idx_y])
        ax2.set_title(f'Uncertainty (σ) — {target_name}')
        ax2.set_facecolor('white')
    else:
        z_mean = pipeline.predict(X_grid).reshape(n_grid, n_grid)

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('white')

        cf = ax.contourf(xx, yy, z_mean, levels=20, cmap='Blues')
        ax.contour(xx, yy, z_mean, levels=10, colors='white', linewidths=0.4, alpha=0.5)
        _overlay_training(ax)
        plt.colorbar(cf, ax=ax, label=f'Predicted {target_name}')
        ax.set_xlabel(feature_names[idx_x])
        ax.set_ylabel(feature_names[idx_y])
        ax.set_title(f'Response Surface — {target_name}')
        ax.set_facecolor('white')

    plt.tight_layout()
    return _fig_to_b64(fig)


def get_learning_curve_plot_b64(train_sizes, train_scores, val_scores, target_name):
    """Plot train vs validation R² across training set sizes."""
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('white')

    ax.plot(train_sizes, train_mean, 'o-', color=ACCENT, lw=2, label='Train R²')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=ACCENT)

    ax.plot(train_sizes, val_mean, 'o-', color='#dc2626', lw=2, label='Validation R²')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='#dc2626')

    ax.axhline(1.0, color='grey', lw=0.8, linestyle=':', alpha=0.5)
    ax.set_xlabel('Training set size')
    ax.set_ylabel('R²')
    ax.set_title(f'Learning Curve — {target_name}')
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=min(0, val_mean.min() - 0.05))
    ax.set_facecolor('white')
    plt.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(pipeline, target_name, models_dir):
    safe_name = target_name.replace(' ', '_').replace('/', '-').replace('\\', '-')
    filepath = os.path.join(models_dir, f'model_{safe_name}.joblib')
    joblib.dump(pipeline, filepath)
    return filepath


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_all(X_train, X_test, y_dict_train, y_dict_test, config, models_dir,
             per_target_config=None):
    """Train one pipeline per target column.

    If per_target_config is provided ({target: {model_type}}), each target uses its
    own model type while sharing GPR/RF hyperparameters from the global config.

    Returns a dict keyed by target name with metrics, plots, model path,
    and (for linear) bootstrap models for uncertainty estimation.
    """
    results = {}
    global_model_type = config['model_type']
    n_features = X_train.shape[1]

    X_full = np.vstack([X_train, X_test])

    for target_idx, target_name in enumerate(y_dict_train):
        y_train = y_dict_train[target_name]
        y_test = y_dict_test[target_name]

        # Per-target model type overrides global; hyperparams always from global config
        if per_target_config and target_name in per_target_config:
            model_type = per_target_config[target_name].get('model_type', global_model_type)
        else:
            model_type = global_model_type

        kernel = build_kernel(config['kernel_type'], n_features, config['length_scale']) if model_type == 'gpr' else None

        pipeline = build_pipeline(model_type, kernel, config['alpha'], config['normalize'], config)
        pipeline.fit(X_train, y_train)

        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        metrics_train = compute_metrics(y_train, y_pred_train)
        metrics_test = compute_metrics(y_test, y_pred_test)

        cv_score = None
        if config.get('use_cv'):
            y_full = np.concatenate([y_train, y_test])
            fresh_kernel = (
                build_kernel(config['kernel_type'], n_features, config['length_scale'])
                if model_type == 'gpr' else None
            )
            fresh_pipeline = build_pipeline(model_type, fresh_kernel,
                                            config['alpha'], config['normalize'], config)
            kf = KFold(n_splits=int(config['cv_k']), shuffle=True, random_state=42)
            scores = cross_val_score(fresh_pipeline, X_full, y_full, cv=kf, scoring='r2')
            cv_score = round(float(scores.mean()), 6)

        # Bootstrap models for linear regression uncertainty — different seed per target
        bootstrap_models = []
        if model_type == 'linear':
            rng = np.random.default_rng(42 + target_idx)
            for _ in range(N_BOOTSTRAP):
                idx = rng.integers(0, len(X_train), size=len(X_train))
                bp = build_pipeline('linear', None, config['alpha'], config['normalize'])
                bp.fit(X_train[idx], y_train[idx])
                bootstrap_models.append(bp)

        # RF std floor: 10% of median in-sample std — prevents false zero uncertainty on extrapolation
        rf_std_floor = None
        if model_type == 'rf':
            in_sample_std = _rf_std(pipeline, X_train)
            median_std = float(np.median(in_sample_std))
            rf_std_floor = 0.1 * median_std if median_std > 0 else None

        feature_names = config.get('feature_cols', [])

        parity_b64, parity_caption = get_parity_plot_b64(y_test, y_pred_test, target_name)
        residuals_b64, residuals_caption = get_residuals_plot_b64(y_test, y_pred_test, target_name)
        feat_importance_b64 = get_feature_importance_plot_b64(
            pipeline, feature_names, model_type, target_name
        )

        optimized_kernel_str = None
        optimized_length_scales = None
        irrelevant_feature_warnings = []

        if model_type == 'gpr':
            gpr = pipeline.named_steps['model']
            optimized_kernel_str = str(gpr.kernel_)
            ls = gpr.kernel_.get_params().get('k2__length_scale', None)
            if ls is not None:
                optimized_length_scales = [round(float(v), 6) for v in np.atleast_1d(ls)]
                for feat, ls_val in zip(feature_names, optimized_length_scales):
                    if ls_val >= 500:
                        irrelevant_feature_warnings.append(
                            f"'{feat}' has length scale λ={ls_val:.0f} (near upper bound 1000). "
                            f"It may not contribute meaningfully to predicting '{target_name}'. "
                            f"Consider removing it for a cleaner model."
                        )

        model_path = save_model(pipeline, target_name, models_dir)

        results[target_name] = {
            'pipeline': pipeline,
            'model_type': model_type,
            'bootstrap_models': bootstrap_models,
            'model_path': model_path,
            'metrics_train': metrics_train,
            'metrics_test': metrics_test,
            'cv_score': cv_score,
            'parity_b64': parity_b64,
            'parity_caption': parity_caption,
            'residuals_b64': residuals_b64,
            'residuals_caption': residuals_caption,
            'feat_importance_b64': feat_importance_b64,
            'optimized_kernel_str': optimized_kernel_str,
            'optimized_length_scales': optimized_length_scales,
            'irrelevant_feature_warnings': irrelevant_feature_warnings,
            'rf_std_floor': rf_std_floor,
        }

    return results
