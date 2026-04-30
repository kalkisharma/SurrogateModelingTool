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


def check_extrapolation(X_input, df_clean, feature_cols):
    """Return warning strings for any feature value outside the training [min, max] range.

    For batch inputs, checks all rows and returns one warning per out-of-range feature.
    """
    out_of_range = {}
    for i, col in enumerate(feature_cols):
        lo = float(df_clean[col].min())
        hi = float(df_clean[col].max())
        col_vals = X_input[:, i]
        n_below = int((col_vals < lo).sum())
        n_above = int((col_vals > hi).sum())
        if n_below > 0 or n_above > 0:
            val_str = f'{float(col_vals[0]):.4g}' if len(col_vals) == 1 else f'{n_below + n_above} row(s)'
            out_of_range[col] = f"'{col}': {val_str} outside training range [{lo:.4g}, {hi:.4g}]"
    return list(out_of_range.values())


def get_outlier_flags(df, cols):
    """IQR-based outlier detection per column.

    Returns dict keyed by column name with row_indices, lo/hi bounds, and values.
    Only columns that have at least one outlier are included. Skips non-numeric and
    zero-IQR (constant) columns.
    """
    flags = {}
    for col in cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        Q1 = float(df[col].quantile(0.25))
        Q3 = float(df[col].quantile(0.75))
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lo = Q1 - 1.5 * IQR
        hi = Q3 + 1.5 * IQR
        mask = (df[col] < lo) | (df[col] > hi)
        rows = df.index[mask].tolist()
        if rows:
            flags[col] = {
                'row_indices': rows,
                'lo': round(lo, 6),
                'hi': round(hi, 6),
                'values': [round(float(df.loc[i, col]), 6) for i in rows],
            }
    return flags


def get_pairplot_b64(df, columns, max_cols=8, outlier_df=None):
    """Render a scatter matrix for up to max_cols numeric columns.

    outlier_df: optional DataFrame of outlier rows to overlay as hollow orange
    circles on every off-diagonal subplot (shown regardless of exclusion state).

    Returns a base64-encoded PNG string, or '' if fewer than 2 numeric cols.
    """
    cols_to_plot = columns[:max_cols]
    df_plot = df[cols_to_plot].select_dtypes(include=[np.number])

    if len(df_plot.columns) < 2:
        return ''

    plot_cols = df_plot.columns.tolist()
    fig, axes = plt.subplots(
        len(plot_cols), len(plot_cols),
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

    # Overlay outlier rows as hollow orange circles on off-diagonal subplots
    if outlier_df is not None and len(outlier_df) > 0:
        outlier_numeric = outlier_df[[c for c in plot_cols if c in outlier_df.columns]]
        for i, row_col in enumerate(plot_cols):
            for j, col_col in enumerate(plot_cols):
                if i != j and row_col in outlier_numeric.columns and col_col in outlier_numeric.columns:
                    axes[i, j].scatter(
                        outlier_numeric[col_col], outlier_numeric[row_col],
                        facecolors='none', edgecolors='#f97316',
                        s=70, linewidths=1.8, zorder=5, alpha=0.9,
                    )

    fig.patch.set_facecolor('white')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
