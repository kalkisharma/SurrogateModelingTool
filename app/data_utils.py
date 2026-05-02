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
        series = df[col].dropna()
        mean = float(series.mean())
        std = float(series.std()) if len(series) > 1 else 0.0
        stats[col] = {
            'min': float(series.min()),
            'max': float(series.max()),
            'mean': mean,
            'skew': float(series.skew()) if len(series) > 2 else 0.0,
            'cv': float(std / abs(mean)) if abs(mean) > 1e-10 else float(std),
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


def get_outlier_flags(df, cols, multiplier=1.5):
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
        lo = Q1 - multiplier * IQR
        hi = Q3 + multiplier * IQR
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


def get_correlation_heatmap_b64(df, feature_cols, threshold=0.92):
    """Pearson correlation heatmap for feature columns.

    Returns (plot_b64, high_corr_pairs) where high_corr_pairs is a list of
    {col_a, col_b, r} dicts for pairs with |r| >= threshold.
    Returns ('', []) when fewer than 2 numeric feature columns are available.
    """
    cols = [c for c in feature_cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        return '', []

    corr = df[cols].corr()
    n = len(cols)
    fig_size = max(4.0, n * 1.1)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, shrink=0.75, label='Pearson r')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(cols, fontsize=10)
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            text_col = 'white' if abs(val) > 0.65 else '#374151'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=text_col)
    ax.set_title('Input Feature Correlations (Pearson r)', fontsize=11, pad=12)
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

    high_corr_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr.iloc[i, j])
            if abs(r) >= threshold:
                high_corr_pairs.append({'col_a': cols[i], 'col_b': cols[j], 'r': round(r, 3)})

    return plot_b64, high_corr_pairs


def get_feat_target_grid_b64(df, feature_cols, target_cols, max_feat_cols=3):
    """Feature vs target scatter plots with a linear trend line.

    Returns (plot_b64, nonlinear_hint_cols) where nonlinear_hint_cols lists features
    whose linear-fit R² < 0.7 against at least one target, suggesting non-linearity.
    """
    feat_cols = [c for c in feature_cols
                 if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    targ_cols = [c for c in target_cols
                 if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols or not targ_cols:
        return '', []

    ncols = min(len(feat_cols), max_feat_cols)
    nrows = len(targ_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8),
                             squeeze=False)

    nonlinear_hint_cols = []

    for j, feat in enumerate(feat_cols[:ncols]):
        for i, targ in enumerate(targ_cols):
            ax = axes[i, j]
            x = df[feat].values.astype(float)
            y = df[targ].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            xm, ym = x[mask], y[mask]

            ax.scatter(xm, ym, alpha=0.55, s=18, color='#2563EB',
                       edgecolors='white', linewidths=0.3, zorder=3)
            ax.set_facecolor('#fafafa')

            if len(xm) >= 3:
                try:
                    coeffs = np.polyfit(xm, ym, 1)
                    x_line = np.linspace(xm.min(), xm.max(), 60)
                    ax.plot(x_line, np.polyval(coeffs, x_line),
                            color='#dc2626', lw=1.5, alpha=0.85, zorder=4)
                    y_pred_lin = np.polyval(coeffs, xm)
                    ss_res = float(np.sum((ym - y_pred_lin) ** 2))
                    ss_tot = float(np.sum((ym - ym.mean()) ** 2))
                    r2_lin = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
                    if r2_lin < 0.7 and feat not in nonlinear_hint_cols:
                        nonlinear_hint_cols.append(feat)
                except Exception:
                    pass

            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel(targ, fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle('What Your Model Will Learn  (red line = linear trend)',
                 fontsize=10, y=1.01)
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

    return plot_b64, nonlinear_hint_cols


def get_unusual_runs_b64(df, feature_cols, top_n=10):
    """Isolation Forest multivariate anomaly detection, displayed as a lollipop chart.

    Returns (plot_b64, top_runs) where top_runs is a list of {row_idx, score} dicts
    (score 1 = most unusual, 0 = typical), sorted highest score first.
    Returns ('', []) when fewer than 2 numeric features or fewer than 8 rows.
    """
    from sklearn.ensemble import IsolationForest

    cols = [c for c in feature_cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2 or len(df) < 8:
        return '', []

    X = df[cols].fillna(df[cols].mean()).values

    clf = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
    clf.fit(X)
    raw_scores = clf.score_samples(X)
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max - s_min < 1e-10:
        normalized = np.zeros(len(raw_scores))
    else:
        normalized = 1.0 - (raw_scores - s_min) / (s_max - s_min)

    n_show = min(top_n, len(df))
    idx_sorted = np.argsort(normalized)[-n_show:]
    scores_sorted = normalized[idx_sorted]
    order = np.argsort(scores_sorted)
    show_idx = idx_sorted[order]
    show_scores = scores_sorted[order]

    fig_h = max(3.0, n_show * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(6, fig_h))
    y_pos = list(range(n_show))

    for y, s in zip(y_pos, show_scores):
        col = '#dc2626' if s > 0.6 else '#f97316' if s > 0.4 else '#2563EB'
        ax.hlines(y, 0, s, color='#e2e8f0', lw=2)
        ax.plot(s, y, 'o', color=col, markersize=7, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Row {int(show_idx[i])}' for i in range(n_show)], fontsize=9)
    ax.set_xlabel('Unusualness score  (0 = typical · 1 = most unusual)', fontsize=9)
    ax.set_title('Unusual Run Detector', fontsize=10, pad=8)
    ax.axvline(x=0.6, color='#dc2626', lw=1.2, linestyle='--', alpha=0.7)
    ax.text(0.62, 0.98, 'review threshold', color='#dc2626', fontsize=7, alpha=0.8,
            transform=ax.get_xaxis_transform(), va='top')
    ax.set_xlim(-0.02, 1.08)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

    top_runs = [
        {'row_idx': int(show_idx[i]), 'score': round(float(show_scores[i]), 3)}
        for i in range(n_show - 1, -1, -1)
    ]

    return plot_b64, top_runs
