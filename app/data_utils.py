import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix


def validate_and_load_csv(filepath):
    """Load CSV and validate it has at least 2 numeric columns.

    Returns (df, None) on success or (None, error_string) on failure.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        return None, f"Could not read CSV: {exc}"

    if len(df) == 0:
        return None, "CSV file is empty."

    if len(df.columns) < 2:
        return None, "CSV must have at least 2 columns."

    df.columns = df.columns.str.strip()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None, "CSV must have at least 2 numeric columns."

    return df, None


def get_summary(df):
    """Return a JSON-serializable summary dict for a DataFrame."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
        }

    return {
        'shape': list(df.shape),
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'null_counts': {col: int(n) for col, n in df.isnull().sum().items()},
        'stats': stats,
    }


def clean_data(df, feature_cols, target_cols):
    """Drop rows with NaN in any selected column.

    Returns (df_clean, n_dropped).
    """
    all_cols = feature_cols + target_cols
    df_sel = df[all_cols]
    df_clean = df_sel.dropna().reset_index(drop=True)
    n_dropped = len(df) - len(df_clean)
    return df_clean, n_dropped


def get_pairplot_b64(df, columns, max_cols=8):
    """Render a scatter matrix for up to max_cols numeric columns.

    Returns a base64-encoded PNG string, or '' if fewer than 2 numeric cols.
    """
    cols_to_plot = columns[:max_cols]
    df_plot = df[cols_to_plot].select_dtypes(include=[np.number])

    if len(df_plot.columns) < 2:
        return ''

    fig, axes = plt.subplots(
        len(df_plot.columns), len(df_plot.columns),
        figsize=(10, 10)
    )
    scatter_matrix(
        df_plot,
        ax=axes,
        alpha=0.5,
        diagonal='hist',
        hist_kwds={'bins': 15, 'color': '#2563EB', 'edgecolor': 'white'},
        color='#2563EB',
    )
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
