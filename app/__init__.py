import logging
import os
from logging.handlers import RotatingFileHandler

import matplotlib
matplotlib.use('Agg')  # must be set before any other matplotlib import

from flask import Flask


def _initial_state():
    return {
        'df_raw': None,
        'df_clean': None,
        'upload_filename': '',
        'summary': None,
        'pairplot_b64': None,
        'feature_cols': [],
        'target_cols': [],
        'n_dropped': 0,
        'train_config': {
            'model_type': 'linear',
            'kernel_type': 'rbf',
            'length_scale': 1.0,
            'alpha': 1e-6,
            'test_size': 0.2,
            'use_cv': False,
            'cv_k': 5,
            'normalize': True,
        },
        'trained': False,
        'gpr_warning': None,
        'results': {},
        'last_predictions': None,
        'train_history': [],
        'training_in_progress': False,
        'de_corr_b64': None,
        'de_corr_pairs': None,
        'de_ft_b64': None,
        'de_nonlinear_cols': None,
        'de_unusual_scores': None,
    }


def _read_version(base_dir):
    version_path = os.path.join(base_dir, 'VERSION')
    try:
        with open(version_path) as f:
            return f.read().strip()
    except OSError:
        return 'unknown'


def create_app():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = Flask(__name__, template_folder='templates')
    app.secret_key = 'surrogate-tool-local-key'

    app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads')
    app.config['MODELS_FOLDER'] = os.path.join(base_dir, 'models')
    app.config['STATE'] = _initial_state()

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

    log_path = os.path.join(base_dir, 'surrogate_tool.log')
    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=2)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Surrogate Modeling Tool v%s', _read_version(base_dir))

    from app.routes import main
    app.register_blueprint(main)

    return app
