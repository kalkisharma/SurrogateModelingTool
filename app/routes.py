import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (Blueprint, current_app, jsonify, render_template,
                   request, send_file)
from numpy.linalg import LinAlgError
from werkzeug.utils import secure_filename

from app import data_utils, ml_engine
from sklearn.base import clone as sklearn_clone
from sklearn.model_selection import learning_curve as sklearn_learning_curve, train_test_split

logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)


def _state():
    return current_app.config['STATE']


def _reset_downstream(state, level='upload'):
    """Clear state from the given level downward."""
    if level == 'upload':
        state.update({
            'df_raw': None, 'df_clean': None, 'upload_filename': '',
            'summary': None, 'pairplot_b64': None,
            'feature_cols': [], 'target_cols': [], 'n_dropped': 0,
            'train_history': [],
        })
    if level in ('upload', 'columns'):
        state.update({
            'trained': False, 'gpr_warning': None,
            'results': {}, 'last_predictions': None,
            'de_corr_b64': None, 'de_corr_pairs': None,
            'de_ft_b64': None, 'de_nonlinear_cols': None,
            'de_unusual_scores': None,
        })


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@main.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@main.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    f = request.files['file']
    if not f.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files are accepted.'}), 400

    f.seek(0, 2)
    size = f.tell()
    f.seek(0)
    if size > 10 * 1024 * 1024:
        return jsonify({'error': f'File is too large ({size // (1024*1024)} MB). Maximum allowed size is 10 MB.'}), 400

    filename = f'{int(time.time())}_{secure_filename(f.filename)}'
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)

    df, error = data_utils.validate_and_load_csv(filepath)
    if error:
        os.remove(filepath)
        return jsonify({'error': error}), 400

    state = _state()
    _reset_downstream(state, level='upload')
    state['df_raw'] = df
    state['upload_filename'] = f.filename
    state['summary'] = data_utils.get_summary(df)

    row_count_warning = None
    if len(df) > 5000:
        row_count_warning = (
            f'Large dataset ({len(df):,} rows). GPR training will be very slow — '
            f'Random Forest or Linear Regression recommended.'
        )

    logger.info('Uploaded %s (%d rows)', f.filename, len(df))
    return jsonify({
        'summary': state['summary'],
        'filename': f.filename,
        'row_count_warning': row_count_warning,
    })


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

@main.route('/api/set_columns', methods=['POST'])
def set_columns():
    body = request.get_json(silent=True) or {}
    feature_cols = body.get('feature_cols', [])
    target_cols = body.get('target_cols', [])

    state = _state()
    if state['df_raw'] is None:
        return jsonify({'error': 'No dataset uploaded.'}), 400

    df = state['df_raw']
    all_cols = set(df.columns.tolist())

    missing = [c for c in feature_cols + target_cols if c not in all_cols]
    if missing:
        return jsonify({'error': f'Unknown columns: {missing}'}), 400

    if not feature_cols:
        return jsonify({'error': 'Select at least one feature column.'}), 400

    if not target_cols:
        return jsonify({'error': 'Select at least one target column.'}), 400

    overlap = set(feature_cols) & set(target_cols)
    if overlap:
        return jsonify({'error': f'Columns cannot be both feature and target: {list(overlap)}'}), 400

    non_numeric = [c for c in target_cols
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        return jsonify({'error': f'Target columns must be numeric: {non_numeric}'}), 400

    df_clean, n_dropped = data_utils.clean_data(df, feature_cols, target_cols)

    if len(df_clean) < 5:
        return jsonify({'error': f'Too few rows after removing NaNs ({len(df_clean)}). Need at least 5.'}), 400

    # Detect IQR outliers before optional exclusion
    iqr_multiplier = float(body.get('iqr_multiplier', 1.5))
    outlier_info = data_utils.get_outlier_flags(df_clean, feature_cols + target_cols, multiplier=iqr_multiplier)

    # Collect outlier rows for pairplot ghost overlay (before any exclusion)
    all_outlier_idx = set()
    for col_flags in outlier_info.values():
        all_outlier_idx.update(col_flags['row_indices'])
    outlier_df = df_clean.loc[sorted(all_outlier_idx)] if all_outlier_idx else None

    # Optionally exclude detected outlier rows
    n_outliers_excluded = 0
    if body.get('exclude_outliers', False) and outlier_info:
        if all_outlier_idx:
            df_clean = df_clean.drop(index=sorted(all_outlier_idx)).reset_index(drop=True)
            n_outliers_excluded = len(all_outlier_idx)
            n_dropped += n_outliers_excluded

    pairplot_b64, pairplot_n_shown, pairplot_n_total = data_utils.get_pairplot_b64(
        df_clean, feature_cols + target_cols, outlier_df=outlier_df
    )

    _reset_downstream(state, level='columns')
    state['feature_cols'] = feature_cols
    state['target_cols'] = target_cols
    state['df_clean'] = df_clean
    state['n_dropped'] = n_dropped
    state['pairplot_b64'] = pairplot_b64

    return jsonify({
        'n_rows': len(df_clean),
        'n_dropped': n_dropped,
        'pairplot_b64': pairplot_b64,
        'pairplot_n_shown': pairplot_n_shown,
        'pairplot_n_total': pairplot_n_total,
        'outlier_info': outlier_info,
        'n_outliers_excluded': n_outliers_excluded,
    })


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@main.route('/api/train', methods=['POST'])
def train():
    body = request.get_json(silent=True) or {}
    state = _state()

    if state['df_clean'] is None or not state['feature_cols']:
        return jsonify({'error': 'No data or column selection found. Complete Step 1 first.'}), 400

    if state.get('training_in_progress'):
        return jsonify({'error': 'Training is already in progress. Please wait.'}), 409

    df_clean = state['df_clean']
    feature_cols = state['feature_cols']
    target_cols = state['target_cols']

    model_type = body.get('model_type', 'linear')

    # Parse max_depth: 0 or missing → None (unlimited)
    raw_max_depth = body.get('max_depth', 0)
    try:
        max_depth = int(raw_max_depth) or None
    except (ValueError, TypeError):
        max_depth = None

    config = {
        'model_type': model_type,
        'kernel_type': body.get('kernel_type', 'rbf'),
        'length_scale': float(body.get('length_scale', 1.0)),
        'alpha': float(body.get('alpha', 1e-6)),
        'test_size': float(body.get('test_size', 0.2)),
        'use_cv': bool(body.get('use_cv', False)),
        'cv_k': int(body.get('cv_k', 5)),
        'normalize': bool(body.get('normalize', True)),
        'feature_cols': feature_cols,
        'max_depth': max_depth,
        'min_samples_leaf': int(body.get('min_samples_leaf', 1)),
    }
    state['train_config'] = config

    # Optional per-target model type overrides (advanced mode)
    per_target_config = body.get('per_target_config') or None

    n_rows = len(df_clean)
    gpr_warning = None
    if config['model_type'] == 'gpr' and n_rows > 2000:
        gpr_warning = (
            f'Warning: GPR training on {n_rows} rows may be very slow (O(n³) complexity). '
            f'Consider Linear Regression or Random Forest for datasets larger than 2,000 rows.'
        )
    state['gpr_warning'] = gpr_warning

    X = df_clean[feature_cols].values
    y_dict = {col: df_clean[col].values for col in target_cols}

    test_size = max(0.1, min(0.4, config['test_size']))
    X_train, X_test, *y_splits = train_test_split(
        X, *y_dict.values(), test_size=test_size, random_state=42
    )

    y_dict_train = {col: y_splits[i * 2] for i, col in enumerate(target_cols)}
    y_dict_test = {col: y_splits[i * 2 + 1] for i, col in enumerate(target_cols)}

    # Snapshot mutable STATE keys so a mid-run exception leaves STATE clean
    _state_backup = {
        'results': state.get('results', {}),
        'trained': state.get('trained', False),
        'last_predictions': state.get('last_predictions'),
        'train_history': list(state.get('train_history', [])),
    }

    state['training_in_progress'] = True
    try:
        results = ml_engine.train_all(
            X_train, X_test, y_dict_train, y_dict_test,
            config, current_app.config['MODELS_FOLDER'],
            per_target_config=per_target_config,
        )
    except (LinAlgError, ValueError) as exc:
        state.update(_state_backup)
        msg = str(exc).lower()
        if 'svd' in msg or 'singular' in msg or 'linalg' in msg:
            user_msg = ('GPR training failed: the data may be too correlated or identical rows exist. '
                        'Try increasing alpha or switching to Random Forest.')
        elif 'nan' in msg or 'inf' in msg:
            user_msg = 'Training failed: the dataset contains invalid values (NaN or Inf). Check your CSV.'
        else:
            user_msg = f'Training failed: {exc}'
        logger.error('Training error', exc_info=True)
        return jsonify({'error': user_msg}), 500
    except MemoryError:
        state.update(_state_backup)
        logger.error('MemoryError during training', exc_info=True)
        return jsonify({'error': 'Not enough memory to train. Try Random Forest or reduce dataset size.'}), 500
    except Exception as exc:
        state.update(_state_backup)
        logger.error('Training error', exc_info=True)
        return jsonify({'error': f'Training failed: {exc}'}), 500
    finally:
        state['training_in_progress'] = False

    state['results'] = results
    state['trained'] = True
    state['last_predictions'] = None

    feature_medians = {col: float(df_clean[col].median()) for col in feature_cols}

    # Pre-compute output ranges (used in both history and results_json)
    target_ranges = {}
    for t in target_cols:
        y_all = np.concatenate([y_dict_train[t], y_dict_test[t]])
        target_ranges[t] = float(y_all.max() - y_all.min()) if len(y_all) > 1 else 1.0

    # Update lightweight training history (keep last 10)
    history_metrics = {}
    for t in target_cols:
        rmse = results[t]['metrics_test']['rmse']
        output_range = target_ranges[t]
        history_metrics[t] = {
            'r2_test': results[t]['metrics_test']['r2'],
            'rmse_pct': round(100 * rmse / output_range, 1) if output_range > 0 else None,
        }
    history_entry = {
        'model_type': config['model_type'],
        'kernel_type': config['kernel_type'] if config['model_type'] == 'gpr' else '—',
        'timestamp': time.strftime('%H:%M:%S'),
        'metrics': history_metrics,
    }
    history = state.get('train_history', [])
    history.append(history_entry)
    state['train_history'] = history[-10:]

    # Build JSON-serializable response (exclude the pipeline objects)
    results_json = {}
    for target, res in results.items():
        output_range = target_ranges[target]
        results_json[target] = {
            'model_type': res.get('model_type', config['model_type']),
            'output_range': output_range,
            'metrics_train': res['metrics_train'],
            'metrics_test': res['metrics_test'],
            'cv_score': res['cv_score'],
            'parity_b64': res['parity_b64'],
            'parity_caption': res.get('parity_caption', ''),
            'residuals_b64': res['residuals_b64'],
            'residuals_caption': res.get('residuals_caption', ''),
            'feat_importance_b64': res['feat_importance_b64'],
            'optimized_kernel_str': res['optimized_kernel_str'],
            'optimized_length_scales': res['optimized_length_scales'],
            'irrelevant_feature_warnings': res['irrelevant_feature_warnings'],
        }

    per_target_config_used = {t: {'model_type': res['model_type']} for t, res in results.items()}

    logger.info('Training complete: %s targets, model_type=%s', len(target_cols), config['model_type'])
    return jsonify({
        'trained': True,
        'gpr_warning': gpr_warning,
        'model_type': config['model_type'],
        'per_target_config_used': per_target_config_used,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'results': results_json,
        'train_history': state['train_history'],
        'feature_medians': feature_medians,
    })


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

@main.route('/api/download/model/<target>')
def download_model(target):
    state = _state()
    if not state['trained'] or target not in state['results']:
        return jsonify({'error': 'Model not found.'}), 404

    filepath = state['results'][target]['model_path']
    if not os.path.exists(filepath):
        return jsonify({'error': 'Model file missing on disk.'}), 404

    safe_name = target.replace(' ', '_').replace('/', '-')
    base = Path(state['upload_filename']).stem
    return send_file(
        filepath,
        as_attachment=True,
        attachment_filename=f'{base}_model_{safe_name}.joblib',
        mimetype='application/octet-stream',
    )


@main.route('/api/download/predictions')
def download_predictions():
    state = _state()
    if state['last_predictions'] is None:
        return jsonify({'error': 'No predictions available. Run a prediction first.'}), 404

    base = Path(state['upload_filename']).stem
    buf = io.BytesIO()
    state['last_predictions'].to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        attachment_filename=f'{base}_predictions.csv',
        mimetype='text/csv',
    )


@main.route('/api/download/config')
def download_config():
    state = _state()
    if not state['trained']:
        return jsonify({'error': 'No trained model. Complete Step 3 first.'}), 400

    cfg = {k: v for k, v in state['train_config'].items() if k != 'feature_cols'}
    cfg['feature_cols'] = state['feature_cols']
    cfg['target_cols'] = state['target_cols']
    cfg['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    cfg['metrics'] = {
        t: {
            'train': res['metrics_train'],
            'test': res['metrics_test'],
        }
        for t, res in state['results'].items()
    }
    cfg['optimized_kernels'] = {
        t: {
            'kernel_str': res.get('optimized_kernel_str'),
            'length_scales': res.get('optimized_length_scales'),
        }
        for t, res in state['results'].items()
    }

    base = Path(state['upload_filename']).stem
    buf = io.BytesIO(json.dumps(cfg, indent=2).encode('utf-8'))
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        attachment_filename=f'{base}_surrogate_config.json',
        mimetype='application/json',
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@main.route('/api/predict', methods=['POST'])
def predict():
    try:
        return _predict_impl()
    except Exception as exc:
        logger.error('Predict error', exc_info=True)
        return jsonify({'error': f'Prediction failed: {exc}'}), 500


def _predict_impl():
    state = _state()
    if not state['trained']:
        return jsonify({'error': 'No trained model. Complete Step 3 first.'}), 400

    feature_cols = state['feature_cols']
    target_cols = state['target_cols']
    global_model_type = state['train_config']['model_type']

    # Detect single-point vs batch
    n_skipped = 0
    input_rows = None
    content_type = request.content_type or ''
    if content_type.startswith('application/json'):
        body = request.get_json(silent=True) or {}
        inputs = body.get('inputs', {})
        missing = [c for c in feature_cols if c not in inputs]
        if missing:
            return jsonify({'error': f'Missing feature values: {missing}'}), 400
        try:
            row = {c: float(inputs[c]) for c in feature_cols}
        except (ValueError, TypeError) as exc:
            return jsonify({'error': f'Invalid input value: {exc}'}), 400
        X_input = pd.DataFrame([row])[feature_cols].values
    else:
        if 'file' not in request.files:
            return jsonify({'error': 'Provide a JSON body for single-point or a file for batch prediction.'}), 400
        f = request.files['file']
        try:
            df_new = pd.read_csv(f)
            df_new.columns = df_new.columns.str.strip()
        except Exception as exc:
            return jsonify({'error': f'Could not read CSV: {exc}'}), 400

        missing = [c for c in feature_cols if c not in df_new.columns]
        if missing:
            return jsonify({'error': f'Missing columns in uploaded CSV: {missing}'}), 400

        n_before = len(df_new)
        df_new = df_new[feature_cols].copy()
        df_new['_input_row'] = range(1, n_before + 1)
        df_new = df_new.dropna(subset=feature_cols).reset_index(drop=True)
        n_skipped = n_before - len(df_new)
        if len(df_new) == 0:
            return jsonify({'error': 'No valid rows in uploaded CSV after removing NaNs.'}), 400
        input_rows = df_new['_input_row'].tolist()
        X_input = df_new[feature_cols].values

    # Extrapolation check
    extrap_warnings = data_utils.check_extrapolation(X_input, state['df_clean'], feature_cols)

    # Run predictions
    pred_rows = [{} for _ in range(len(X_input))]
    for col in feature_cols:
        col_idx = feature_cols.index(col)
        for i, row in enumerate(pred_rows):
            row[col] = float(X_input[i, col_idx])

    for target in target_cols:
        res_entry = state['results'][target]
        pipeline = res_entry['pipeline']
        target_model_type = res_entry.get('model_type', global_model_type)
        preds = pipeline.predict(X_input)

        if target_model_type == 'gpr':
            gpr = pipeline.named_steps['model']
            if 'scaler' in pipeline.named_steps:
                X_scaled = pipeline.named_steps['scaler'].transform(X_input)
            else:
                X_scaled = X_input
            _, stds = gpr.predict(X_scaled, return_std=True)
            for i, row in enumerate(pred_rows):
                row[target] = round(float(preds[i]), 8)
                row[f'{target}_std'] = round(float(stds[i]), 8)
        elif target_model_type == 'rf':
            stds = ml_engine._rf_std(pipeline, X_input)
            for i, row in enumerate(pred_rows):
                row[target] = round(float(preds[i]), 8)
                row[f'{target}_std'] = round(float(stds[i]), 8)
        else:  # linear
            bootstrap_models = res_entry.get('bootstrap_models', [])
            stds = ml_engine._bootstrap_std(bootstrap_models, X_input)
            for i, row in enumerate(pred_rows):
                row[target] = round(float(preds[i]), 8)
                if stds is not None:
                    row[f'{target}_std'] = round(float(stds[i]), 8)

    if input_rows is not None:
        for i, row in enumerate(pred_rows):
            row['input_row'] = input_rows[i]

    state['last_predictions'] = pd.DataFrame(pred_rows)

    return jsonify({
        'predictions': pred_rows,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'model_types': {t: state['results'][t].get('model_type', global_model_type) for t in target_cols},
        'extrapolation_warnings': extrap_warnings,
        'n_skipped': n_skipped,
    })


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

@main.route('/api/sensitivity')
def sensitivity():
    state = _state()
    if not state['trained']:
        return jsonify({'error': 'No trained model.'}), 400

    feature = request.args.get('feature', '')
    target = request.args.get('target', '')
    feature_cols = state['feature_cols']
    target_cols = state['target_cols']

    if feature not in feature_cols:
        return jsonify({'error': f'Unknown feature: {feature}'}), 400
    if target not in target_cols:
        return jsonify({'error': f'Unknown target: {target}'}), 400

    df_clean = state['df_clean']
    feature_idx = feature_cols.index(feature)

    # Build reference point — use query params if provided, else median
    X_ref = df_clean[feature_cols].median().values.reshape(1, -1)
    for i, col in enumerate(feature_cols):
        param_val = request.args.get(f'ref_{col}')
        if param_val is not None:
            try:
                X_ref[0, i] = float(param_val)
            except ValueError:
                pass

    col_vals = df_clean[feature].values
    lo = col_vals.min()
    hi = col_vals.max()
    rng = hi - lo if hi > lo else 1.0
    x_sweep = np.linspace(lo - 0.1 * rng, hi + 0.1 * rng, 100)

    extrap_warnings = data_utils.check_extrapolation(X_ref, df_clean, feature_cols)

    pipeline = state['results'][target]['pipeline']
    model_type = state['results'][target].get('model_type', state['train_config']['model_type'])

    bootstrap_models = state['results'][target].get('bootstrap_models', [])
    try:
        plot_b64 = ml_engine.get_sensitivity_plot_b64(
            pipeline, X_ref, feature_cols, feature_idx, target, model_type, x_sweep,
            bootstrap_models=bootstrap_models,
            train_lo=float(lo), train_hi=float(hi),
        )
    except Exception as exc:
        logger.error('Sensitivity plot failed', exc_info=True)
        return jsonify({'error': str(exc)}), 500

    return jsonify({
        'plot_b64': plot_b64,
        'feature': feature,
        'target': target,
        'train_lo': float(lo),
        'train_hi': float(hi),
        'extrapolation_warnings': extrap_warnings,
    })


# ---------------------------------------------------------------------------
# 2D Response Surface
# ---------------------------------------------------------------------------

@main.route('/api/surface')
def surface():
    state = _state()
    if not state['trained']:
        return jsonify({'error': 'No trained model.'}), 400

    feature_x = request.args.get('feature_x', '')
    feature_y = request.args.get('feature_y', '')
    target = request.args.get('target', '')
    feature_cols = state['feature_cols']
    target_cols = state['target_cols']

    if feature_x not in feature_cols:
        return jsonify({'error': f'Unknown feature_x: {feature_x}'}), 400
    if feature_y not in feature_cols:
        return jsonify({'error': f'Unknown feature_y: {feature_y}'}), 400
    if feature_x == feature_y:
        return jsonify({'error': 'feature_x and feature_y must be different.'}), 400
    if target not in target_cols:
        return jsonify({'error': f'Unknown target: {target}'}), 400

    df_clean = state['df_clean']
    idx_x = feature_cols.index(feature_x)
    idx_y = feature_cols.index(feature_y)

    X_ref = df_clean[feature_cols].median().values.reshape(1, -1)

    def _range(col):
        vals = df_clean[col].values
        lo, hi = vals.min(), vals.max()
        rng = hi - lo if hi > lo else 1.0
        return (lo - 0.05 * rng, hi + 0.05 * rng)

    x_range = _range(feature_x)
    y_range = _range(feature_y)

    pipeline = state['results'][target]['pipeline']
    model_type = state['results'][target].get('model_type', state['train_config']['model_type'])

    X_train_arr = df_clean[feature_cols].values
    try:
        plot_b64 = ml_engine.get_surface_plot_b64(
            pipeline, X_ref, feature_cols, idx_x, idx_y,
            target, model_type, x_range, y_range, X_train=X_train_arr
        )
    except Exception as exc:
        logger.error('Surface plot failed', exc_info=True)
        return jsonify({'error': str(exc)}), 500

    return jsonify({
        'plot_b64': plot_b64,
        'feature_x': feature_x,
        'feature_y': feature_y,
        'target': target,
    })


# ---------------------------------------------------------------------------
# Learning Curve (on-demand)
# ---------------------------------------------------------------------------

@main.route('/api/learning_curve')
def learning_curve_route():
    state = _state()
    if not state['trained']:
        return jsonify({'error': 'No trained model.'}), 400

    target = request.args.get('target', '')
    if target not in state['target_cols']:
        return jsonify({'error': f'Unknown target: {target}'}), 400

    df_clean = state['df_clean']
    feature_cols = state['feature_cols']
    config = state['train_config']
    model_type = state['results'][target].get('model_type', config['model_type'])
    n_features = len(feature_cols)

    X = df_clean[feature_cols].values
    y = df_clean[target].values
    n = len(X)

    if model_type == 'gpr':
        fitted_gpr = state['results'][target]['pipeline'].named_steps['model']
        kernel = sklearn_clone(fitted_gpr.kernel_)
    else:
        kernel = None
    fresh_pipeline = ml_engine.build_pipeline(
        model_type, kernel, config['alpha'], config['normalize'], config
    )

    cv = min(5, max(2, n // 5))
    try:
        train_sizes, train_scores, val_scores = sklearn_learning_curve(
            fresh_pipeline, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv,
            scoring='r2',
            n_jobs=1,
        )
        plot_b64 = ml_engine.get_learning_curve_plot_b64(train_sizes, train_scores, val_scores, target)
    except Exception as exc:
        logger.error('Learning curve failed', exc_info=True)
        return jsonify({'error': str(exc)}), 500
    return jsonify({
        'plot_b64': plot_b64,
        'target': target,
        'final_train_r2': float(train_scores[-1].mean()),
        'final_val_r2': float(val_scores[-1].mean()),
        'val_still_rising': bool(val_scores[-1].mean() > val_scores[-2].mean()),
    })


# ---------------------------------------------------------------------------
# Data Explorer (on-demand, Step 1)
# ---------------------------------------------------------------------------

@main.route('/api/data_explorer')
def data_explorer():
    state = _state()
    if state['df_clean'] is None or not state['feature_cols']:
        return jsonify({'error': 'No data. Complete Step 1 first.'}), 400

    feature_cols = state['feature_cols']
    n_total_features = len(feature_cols)
    n_plotted = min(n_total_features, 6)

    # Read cache into locals once to avoid race condition between check and use
    cached_corr = state.get('de_corr_b64')
    cached_ft = state.get('de_ft_b64')
    cached_pairs = state.get('de_corr_pairs')
    cached_nonlinear = state.get('de_nonlinear_cols')

    df_clean = state['df_clean']
    target_cols = state['target_cols']
    include_targets = request.args.get('include_targets', 'false').lower() == 'true'

    if include_targets:
        heatmap_b64, high_corr_pairs = data_utils.get_correlation_heatmap_b64(
            df_clean, feature_cols, extra_cols=target_cols
        )
        feat_target_b64 = cached_ft
        nonlinear_hint_cols = cached_nonlinear
        if feat_target_b64 is None:
            feat_target_b64, nonlinear_hint_cols = data_utils.get_feat_target_grid_b64(
                df_clean, feature_cols, target_cols, max_feat_cols=n_plotted
            )
            state['de_ft_b64'] = feat_target_b64
            state['de_nonlinear_cols'] = nonlinear_hint_cols
    elif cached_corr is not None:
        return jsonify({
            'corr_heatmap_b64': cached_corr,
            'feat_target_grid_b64': cached_ft,
            'high_corr_pairs': cached_pairs,
            'nonlinear_hint_cols': cached_nonlinear,
            'n_plotted': n_plotted,
            'n_total_features': n_total_features,
        })
    else:
        heatmap_b64, high_corr_pairs = data_utils.get_correlation_heatmap_b64(
            df_clean, feature_cols
        )
        feat_target_b64, nonlinear_hint_cols = data_utils.get_feat_target_grid_b64(
            df_clean, feature_cols, target_cols, max_feat_cols=n_plotted
        )
        state['de_corr_b64'] = heatmap_b64
        state['de_corr_pairs'] = high_corr_pairs
        state['de_ft_b64'] = feat_target_b64
        state['de_nonlinear_cols'] = nonlinear_hint_cols

    return jsonify({
        'corr_heatmap_b64': heatmap_b64,
        'feat_target_grid_b64': feat_target_b64,
        'high_corr_pairs': high_corr_pairs,
        'nonlinear_hint_cols': nonlinear_hint_cols,
        'n_plotted': n_plotted,
        'n_total_features': n_total_features,
    })


@main.route('/api/unusual_runs')
def unusual_runs():
    state = _state()
    if state['df_clean'] is None or not state['feature_cols']:
        return jsonify({'error': 'No data. Complete Step 1 first.'}), 400

    doe_type = request.args.get('doe_type', 'grid')
    df_clean = state['df_clean']
    feature_cols = state['feature_cols']

    try:
        plot_b64, top_runs, all_scores = data_utils.get_unusual_runs_b64(df_clean, feature_cols)
    except Exception as exc:
        logger.error('Unusual runs failed', exc_info=True)
        return jsonify({'error': str(exc)}), 500

    state['de_unusual_scores'] = {r['row_idx']: r['score'] for r in all_scores}

    n_total = len(df_clean)
    flagged_rows = [r for r in top_runs if r['score'] > 0.6]
    if flagged_rows:
        outlier_flags = data_utils.get_outlier_flags(df_clean, feature_cols)
        feature_bounds = {col: {'lo': info['lo'], 'hi': info['hi']}
                          for col, info in outlier_flags.items()}
        flagged_detail = []
        for r in flagged_rows:
            row_idx = r['row_idx']
            vals = {col: round(float(df_clean.loc[row_idx, col]), 4)
                    for col in feature_cols if row_idx in df_clean.index}
            flagged_detail.append({'row_idx': row_idx, 'score': r['score'], 'values': vals})
    else:
        feature_bounds = {}
        flagged_detail = []

    doe_caveat = None
    if doe_type == 'grid':
        doe_caveat = (
            'Structured grid: boundary and corner points often score higher than '
            'interior points — this is expected in a full-factorial or parametric sweep. '
            'Review flagged rows for setup errors before deciding to exclude.'
        )

    return jsonify({
        'plot_b64': plot_b64,
        'top_runs': top_runs,
        'doe_caveat': doe_caveat,
        'n_total': n_total,
        'flagged_detail': flagged_detail,
        'feature_bounds': feature_bounds,
        'all_scores': all_scores,
    })


# ---------------------------------------------------------------------------
# Custom scatter (Data Explorer F3)
# ---------------------------------------------------------------------------

@main.route('/api/data_scatter', methods=['POST'])
def data_scatter():
    state = _state()
    if state['df_clean'] is None or not state['feature_cols']:
        return jsonify({'error': 'No data. Complete Step 1 first.'}), 400

    body = request.get_json(force=True) or {}
    x_col = body.get('x_col', '').strip()
    y_col = body.get('y_col', '').strip()
    if not x_col or not y_col:
        return jsonify({'error': 'x_col and y_col are required.'}), 400

    all_cols = state['feature_cols'] + state['target_cols']
    if x_col not in all_cols or y_col not in all_cols:
        return jsonify({'error': 'Column not in selected feature or target list.'}), 400

    color_col = (body.get('color_col') or '').strip() or None
    x_log = bool(body.get('x_log', False))
    y_log = bool(body.get('y_log', False))
    filters = body.get('filters', [])  # [{col, min, max}, ...]

    unusual_scores = state.get('de_unusual_scores')

    try:
        plot_b64, n_filtered, n_plotted, n_color_missing, log_warning = data_utils.get_scatter_plot_b64(
            state['df_clean'], x_col, y_col,
            filters=filters,
            color_col=color_col, unusual_scores=unusual_scores,
            x_log=x_log, y_log=y_log,
        )
    except Exception as exc:
        logger.error('Data scatter failed', exc_info=True)
        return jsonify({'error': str(exc)}), 500

    return jsonify({
        'plot_b64': plot_b64,
        'n_filtered': n_filtered,
        'n_plotted': n_plotted,
        'n_color_missing': n_color_missing,
        'log_warning': log_warning,
    })
