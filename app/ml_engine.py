import base64
import io
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ACCENT = '#2563EB'


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


def build_pipeline(model_type, kernel, alpha, normalize):
    steps = []
    if normalize:
        steps.append(('scaler', StandardScaler()))

    if model_type == 'gpr':
        estimator = GaussianProcessRegressor(
            kernel=kernel,
            alpha=float(alpha),
            normalize_y=True,
            n_restarts_optimizer=5,
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
# Plot helpers — all return base64-encoded PNG strings
# ---------------------------------------------------------------------------

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def get_parity_plot_b64(y_true, y_pred, target_name):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')

    ax.scatter(y_true, y_pred, s=30, color=ACCENT, alpha=0.7,
               edgecolors='white', linewidths=0.5)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    lims = [lo - pad, hi + pad]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(f'Actual {target_name}')
    ax.set_ylabel(f'Predicted {target_name}')
    ax.set_title(f'Parity Plot — {target_name}')
    ax.set_facecolor('white')
    plt.tight_layout()
    return _fig_to_b64(fig)


def get_residuals_plot_b64(y_true, y_pred, target_name):
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
    return _fig_to_b64(fig)


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
    else:
        gpr = pipeline.named_steps['model']
        ls = gpr.kernel_.get_params().get('k2__length_scale', None)
        if ls is None:
            # fallback: single length scale kernel
            ls = np.ones(n)
        ls = np.atleast_1d(ls)
        importance = 1.0 / (ls + 1e-12)
        importance = importance / (importance.sum() + 1e-12)
        xlabel = 'Relative Importance (1 / length_scale)'
        title = f'Feature Relevance (ARD) — {target_name}'

    sorted_idx = np.argsort(importance)
    ax.barh(range(n), importance[sorted_idx], color=ACCENT, edgecolor='white')
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_facecolor('white')
    plt.tight_layout()
    return _fig_to_b64(fig)


def get_sensitivity_plot_b64(pipeline, X_ref, feature_names, feature_idx,
                              target_name, model_type, x_sweep):
    X_sweep = np.tile(X_ref, (len(x_sweep), 1))
    X_sweep[:, feature_idx] = x_sweep

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('white')

    if model_type == 'gpr':
        gpr = pipeline.named_steps['model']
        if 'scaler' in pipeline.named_steps:
            X_scaled = pipeline.named_steps['scaler'].transform(X_sweep)
        else:
            X_scaled = X_sweep
        y_pred, y_std = gpr.predict(X_scaled, return_std=True)
        ax.plot(x_sweep, y_pred, color=ACCENT, lw=2, label='Mean')
        ax.fill_between(x_sweep, y_pred - y_std, y_pred + y_std,
                        alpha=0.25, color=ACCENT, label='±1σ')
        ax.legend(fontsize=9)
    else:
        y_pred = pipeline.predict(X_sweep)
        ax.plot(x_sweep, y_pred, color=ACCENT, lw=2)

    ref_val = float(X_ref[0, feature_idx])
    ax.axvline(ref_val, color='grey', linestyle='--', lw=1, alpha=0.7,
               label='reference')

    ax.set_xlabel(feature_names[feature_idx])
    ax.set_ylabel(target_name)
    ax.set_title(f'1D Sensitivity: {target_name} vs {feature_names[feature_idx]}')
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

def train_all(X_train, X_test, y_dict_train, y_dict_test, config, models_dir):
    """Train one pipeline per target column.

    Returns a dict keyed by target name with metrics, plots, and model path.
    """
    results = {}
    model_type = config['model_type']
    n_features = X_train.shape[1]

    X_full = np.vstack([X_train, X_test])

    for target_name in y_dict_train:
        y_train = y_dict_train[target_name]
        y_test = y_dict_test[target_name]

        if model_type == 'gpr':
            kernel = build_kernel(config['kernel_type'], n_features, config['length_scale'])
        else:
            kernel = None

        pipeline = build_pipeline(model_type, kernel, config['alpha'], config['normalize'])
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
                                            config['alpha'], config['normalize'])
            scores = cross_val_score(fresh_pipeline, X_full, y_full,
                                     cv=int(config['cv_k']), scoring='r2')
            cv_score = round(float(scores.mean()), 6)

        feature_names = config.get('feature_cols', [])

        parity_b64 = get_parity_plot_b64(y_test, y_pred_test, target_name)
        residuals_b64 = get_residuals_plot_b64(y_test, y_pred_test, target_name)
        feat_importance_b64 = get_feature_importance_plot_b64(
            pipeline, feature_names, model_type, target_name
        )

        optimized_kernel_str = None
        optimized_length_scales = None
        if model_type == 'gpr':
            gpr = pipeline.named_steps['model']
            optimized_kernel_str = str(gpr.kernel_)
            ls = gpr.kernel_.get_params().get('k2__length_scale', None)
            if ls is not None:
                optimized_length_scales = [round(float(v), 6)
                                           for v in np.atleast_1d(ls)]

        model_path = save_model(pipeline, target_name, models_dir)

        results[target_name] = {
            'pipeline': pipeline,
            'model_path': model_path,
            'metrics_train': metrics_train,
            'metrics_test': metrics_test,
            'cv_score': cv_score,
            'parity_b64': parity_b64,
            'residuals_b64': residuals_b64,
            'feat_importance_b64': feat_importance_b64,
            'optimized_kernel_str': optimized_kernel_str,
            'optimized_length_scales': optimized_length_scales,
        }

    return results
