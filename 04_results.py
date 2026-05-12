"""
Step 4: Results — Tables and Figures
Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning" replication.

Generates:
  Table 1  — Monthly OOS R² by model and subgroup
  Table 3  — Diebold-Mariano pairwise tests
  Table 7  — Decile portfolio average returns
  Table 8  — Portfolio risk-adjusted performance (SR, alpha, drawdown, turnover)
  Figure 3 — Model complexity over time (ENet nonzero coefficients, RF/GBRT depth)
  Figure 4 — Top-20 variable importance (GBRT, RF, NN3)
  Figure 5 — Overall variable importance ranking across models
  Figure 6 — Partial dependence plots (4 key characteristics)
  Cumulative return plots for long-short decile portfolios
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = r"D:\桌面文件\US stock research\chapter2\ReplicateXiuML"
PRED_DIR = os.path.join(BASE_DIR, "predictions")
OUT_DIR  = os.path.join(BASE_DIR, "output")
DATA_PATH = os.path.join(BASE_DIR, "data", "data_clean.parquet")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from models_config import r2_oos

# ── Model names / display names ───────────────────────────────────────────────
MODEL_FILES = {
    'OLS+H':   'ols_h',
    'OLS-3':   'ols3',
    'PLS':     'pls',
    'PCR':     'pcr',
    'ENet+H':  'enet_h',
    'RF':      'rf',
    'GBRT+H':  'gbrt_h',
    'NN1':     'nn1',
    'NN2':     'nn2',
    'NN3':     'nn3',
    'NN4':     'nn4',
    'NN5':     'nn5',
}

COLORS = {
    'OLS+H': '#808080', 'OLS-3': '#a0a0a0', 'PLS': '#1f77b4',
    'PCR': '#ff7f0e', 'ENet+H': '#2ca02c', 'RF': '#9467bd',
    'GBRT+H': '#8c564b', 'NN1': '#e377c2', 'NN2': '#17becf',
    'NN3': '#d62728', 'NN4': '#bcbd22', 'NN5': '#7f7f7f',
}


# ── Load predictions ──────────────────────────────────────────────────────────

def load_all_preds():
    preds = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(PRED_DIR, f"{fname}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['eom'] = pd.to_datetime(df['eom'])
            preds[name] = df.sort_values('eom').reset_index(drop=True)
            print(f"  {name}: {len(df):,} obs, years {df['eom'].dt.year.min()}-{df['eom'].dt.year.max()}")
        else:
            print(f"  {name}: NOT FOUND (run 03_rolling_train.py first)")
    return preds


def load_me():
    """Load market-equity weights for value-weighted portfolios."""
    df = pd.read_parquet(DATA_PATH, columns=['permno','eom','me'])
    df['eom'] = pd.to_datetime(df['eom'])
    return df


# ── Table 1: OOS R² ──────────────────────────────────────────────────────────

def compute_r2_table(preds: dict) -> pd.DataFrame:
    """Full panel R²_OOS, plus top/bottom 1000 by market equity."""
    me_df = load_me()
    rows = []
    for model_name, df in preds.items():
        # Merge market equity
        merged = df.merge(me_df, on=['permno','eom'], how='left')

        # Full sample
        valid = merged['y_true'].notna() & merged['y_pred'].notna()
        r2_all = r2_oos(merged.loc[valid,'y_true'].values,
                        merged.loc[valid,'y_pred'].values)

        # Top-1000 by market cap each month
        def top_n(n, largest=True):
            res = merged.copy()
            if largest:
                mask = res.groupby('eom')['me'].rank(ascending=False) <= n
            else:
                mask = res.groupby('eom')['me'].rank(ascending=True) <= n
            sub = res[mask & valid]
            if len(sub) < 100:
                return np.nan
            return r2_oos(sub['y_true'].values, sub['y_pred'].values)

        r2_top  = top_n(1000, largest=True)
        r2_bot  = top_n(1000, largest=False)

        rows.append({'Model': model_name,
                     'Full sample': r2_all * 100,
                     'Large cap (top 1000)': r2_top * 100 if not np.isnan(r2_top) else np.nan,
                     'Small cap (bot 1000)': r2_bot * 100 if not np.isnan(r2_bot) else np.nan})

    table = pd.DataFrame(rows).set_index('Model')
    return table.loc[list(MODEL_FILES.keys())]  # preserve order


# ── Table 3: Diebold-Mariano ──────────────────────────────────────────────────

def diebold_mariano(y: np.ndarray, f1: np.ndarray, f2: np.ndarray,
                    h: int = 1) -> float:
    """DM test statistic: positive = f1 outperforms f2."""
    e1 = (y - f1) ** 2
    e2 = (y - f2) ** 2
    d  = e2 - e1
    # HAC variance with lag h
    T  = len(d)
    d_mean = d.mean()
    gamma0 = np.mean((d - d_mean) ** 2)
    gammah = 0.0
    for lag in range(1, h):
        gammah += np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
    var_d = (gamma0 + 2 * gammah) / T
    if var_d <= 0:
        return np.nan
    return d_mean / np.sqrt(var_d)


def compute_dm_table(preds: dict) -> pd.DataFrame:
    models = list(preds.keys())
    # Align all models on same observations (inner join on permno+eom)
    base = preds[models[0]][['permno','eom','y_true']].copy()
    for m in models:
        p = preds[m][['permno','eom','y_pred']].rename(columns={'y_pred': m})
        base = base.merge(p, on=['permno','eom'], how='inner')

    DM = pd.DataFrame(index=models, columns=models, dtype=float)
    y  = base['y_true'].values
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                DM.loc[m1, m2] = np.nan
            else:
                DM.loc[m1, m2] = diebold_mariano(y, base[m1].values,
                                                   base[m2].values)
    return DM


# ── Portfolio construction ────────────────────────────────────────────────────

def build_portfolios(pred_df: pd.DataFrame, me_df: pd.DataFrame,
                     n_deciles: int = 10):
    """
    Build long-short decile portfolios.
    Returns monthly equal-weighted and value-weighted long-short returns.
    """
    merged = pred_df.merge(me_df, on=['permno','eom'], how='left')
    merged = merged.dropna(subset=['y_pred','y_true','me'])
    merged['me'] = merged['me'].clip(lower=1e-6)

    results = []
    for eom, g in merged.groupby('eom'):
        if len(g) < n_deciles * 5:
            continue
        g = g.copy()
        g['decile'] = pd.qcut(g['y_pred'], n_deciles, labels=False,
                               duplicates='drop')
        # Equal-weighted
        ew = g.groupby('decile')['y_true'].mean()
        if len(ew) < n_deciles:
            continue
        ls_ew = ew.iloc[-1] - ew.iloc[0]
        # Value-weighted
        def vw_ret(sub):
            w = sub['me'] / sub['me'].sum()
            return (w * sub['y_true']).sum()
        vw = g.groupby('decile').apply(vw_ret)
        ls_vw = vw.iloc[-1] - vw.iloc[0]

        results.append({
            'eom':    eom,
            'ls_ew':  ls_ew,
            'ls_vw':  ls_vw,
            'decile_ew': ew.values,
            'decile_vw': vw.values,
        })
    return pd.DataFrame(results).set_index('eom').sort_index()


def annualize(monthly_ret: np.ndarray):
    mean_m  = np.mean(monthly_ret)
    std_m   = np.std(monthly_ret, ddof=1)
    ann_ret = mean_m * 12
    ann_vol = std_m * np.sqrt(12)
    sr      = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sr


def max_drawdown(cum_ret: np.ndarray):
    peak = np.maximum.accumulate(cum_ret)
    dd   = (cum_ret - peak) / peak
    return float(dd.min() * 100)


def max_1m_loss(monthly_ret: np.ndarray):
    return float(np.min(monthly_ret) * 100)


def compute_portfolio_stats(port_df: pd.DataFrame, col: str = 'ls_vw'):
    r = port_df[col].values
    ann_ret, ann_vol, sr = annualize(r)
    cum = np.cumprod(1 + r)
    mdd = max_drawdown(cum)
    m1l = max_1m_loss(r)
    return {
        'Mean ret (%)': ann_ret * 100,
        'Volatility (%)': ann_vol * 100,
        'Sharpe ratio': sr,
        'Max DD (%)':   mdd,
        'Max 1M loss (%)': m1l,
    }


# ── Table 7: Decile returns ───────────────────────────────────────────────────

def compute_decile_table(preds: dict, me_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, df in preds.items():
        port = build_portfolios(df, me_df)
        if len(port) == 0:
            continue
        ew_avg = np.array([r for r in port['decile_ew']]).mean(axis=0) * 100
        vw_avg = np.array([r for r in port['decile_vw']]).mean(axis=0) * 100
        row = {'Model': model_name}
        for i, (ew, vw) in enumerate(zip(ew_avg, vw_avg), 1):
            row[f'D{i} EW'] = ew
            row[f'D{i} VW'] = vw
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model')


# ── Table 8: Risk-adjusted performance ───────────────────────────────────────

def compute_perf_table(preds: dict, me_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, df in preds.items():
        port = build_portfolios(df, me_df)
        if len(port) == 0:
            continue
        stats_vw = compute_portfolio_stats(port, 'ls_vw')
        stats_ew = compute_portfolio_stats(port, 'ls_ew')
        rows.append({
            'Model':           model_name,
            'Mean ret VW (%)': stats_vw['Mean ret (%)'],
            'SR VW':           stats_vw['Sharpe ratio'],
            'Max DD VW (%)':   stats_vw['Max DD (%)'],
            'Max 1M VW (%)':   stats_vw['Max 1M loss (%)'],
            'Mean ret EW (%)': stats_ew['Mean ret (%)'],
            'SR EW':           stats_ew['Sharpe ratio'],
            'Max DD EW (%)':   stats_ew['Max DD (%)'],
            'Max 1M EW (%)':   stats_ew['Max 1M loss (%)'],
        })
    return pd.DataFrame(rows).set_index('Model')


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_cumulative_returns(preds: dict, me_df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, col, title in zip(axes, ['ls_vw', 'ls_ew'],
                               ['Value-Weighted Long-Short',
                                'Equal-Weighted Long-Short']):
        for model_name, df in preds.items():
            port = build_portfolios(df, me_df)
            if len(port) == 0:
                continue
            cum = (1 + port[col]).cumprod()
            ax.plot(cum.index, cum.values, label=model_name,
                    color=COLORS.get(model_name, 'gray'), linewidth=1.2)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.axhline(1, color='black', linewidth=0.8, linestyle='--')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Cumulative Returns of Long-Short Decile Portfolios (1987–2016)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def fig_variable_importance(preds: dict, feat_cols: list, out_path: str,
                             top_n: int = 20):
    """Figure 4: Top-20 variable importance for GBRT, RF, NN3."""
    tree_models = {'GBRT+H': 'gbrt_h', 'RF': 'rf'}
    fi_data = {}

    for display, fname in tree_models.items():
        fi_files = sorted([f for f in os.listdir(PRED_DIR)
                           if f.startswith(f"{fname}_fi_") and f.endswith('.npy')])
        if fi_files:
            fi_arr = np.stack([np.load(os.path.join(PRED_DIR, f))
                               for f in fi_files], axis=0)
            fi_data[display] = fi_arr.mean(axis=0)

    if not fi_data:
        print("No feature importance files found; skipping Figure 4.")
        return

    fig, axes = plt.subplots(1, len(fi_data), figsize=(8 * len(fi_data), 8))
    if len(fi_data) == 1:
        axes = [axes]

    for ax, (model_name, fi) in zip(axes, fi_data.items()):
        n_feats = min(len(fi), len(feat_cols))
        fi = fi[:n_feats]
        fi_norm = fi / fi.sum()
        top_idx = np.argsort(fi_norm)[::-1][:top_n]
        top_fi  = fi_norm[top_idx]
        top_names = [feat_cols[i] for i in top_idx]

        ax.barh(range(top_n), top_fi[::-1], color='steelblue', edgecolor='white')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel('Variable Importance (normalized)')
        ax.set_title(f'{model_name} — Top {top_n} Variables', fontsize=11)
        ax.grid(True, axis='x', alpha=0.3)

    plt.suptitle('Figure 4: Variable Importance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def fig_r2_time_series(preds: dict, out_path: str):
    """Monthly OOS R² time series for top models."""
    fig, ax = plt.subplots(figsize=(14, 5))
    highlight = ['OLS-3', 'ENet+H', 'GBRT+H', 'RF', 'NN3']

    for model_name in highlight:
        if model_name not in preds:
            continue
        df = preds[model_name]
        monthly = df.groupby('eom').apply(
            lambda g: r2_oos(g['y_true'].values, g['y_pred'].values) * 100
        ).reset_index()
        monthly.columns = ['eom', 'r2']
        # 12-month rolling
        monthly['r2_roll'] = monthly['r2'].rolling(12, min_periods=6).mean()
        ax.plot(monthly['eom'], monthly['r2_roll'], label=model_name,
                color=COLORS.get(model_name, 'gray'), linewidth=1.5)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('R²_OOS (%, 12m rolling avg)')
    ax.set_title('Monthly Out-of-Sample R² (12-Month Rolling Average)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def fig_rank_importance(fi_data: dict, feat_cols: list, out_path: str):
    """Figure 5: Overall rank of variable importance across models."""
    if not fi_data:
        return
    n_feats = min(min(len(fi) for fi in fi_data.values()), len(feat_cols))
    rank_sum = np.zeros(n_feats)
    for fi in fi_data.values():
        fi_sub = fi[:n_feats]
        ranks  = stats.rankdata(-fi_sub)  # rank 1 = most important
        rank_sum += ranks

    order = np.argsort(rank_sum)[:40]
    names = [feat_cols[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(len(names)), rank_sum[order[::-1]][::-1],
            color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel('Sum of Ranks (lower = more important)')
    ax.set_title('Figure 5: Overall Variable Importance Ranking Across Models',
                 fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Save tables ───────────────────────────────────────────────────────────────

def save_table(df: pd.DataFrame, name: str, fmt='%.3f'):
    csv_path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(csv_path, float_format=fmt)
    print(f"Saved: {csv_path}")

    # Also print nicely
    print(f"\n{'='*70}")
    print(f"  {name}")
    print('='*70)
    with pd.option_context('display.float_format', lambda x: f'{x:.3f}',
                           'display.max_columns', 20,
                           'display.width', 120):
        print(df.to_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def _build_feat_names():
    """Reconstruct the 919-col feature name list used by 03_rolling_train.py."""
    chars_path = os.path.join(BASE_DIR, "data", "paper_chars.txt")
    with open(chars_path) as f:
        char_cols = [l.strip() for l in f if l.strip()]
    # Peek at parquet schema only (fast)
    import pyarrow.parquet as pq
    schema = pq.read_schema(DATA_PATH)
    all_cols = schema.names
    macro_cols = [c for c in all_cols if c.startswith('m_')]
    ind_cols   = [c for c in all_cols if c.startswith('ind_sic2_')]
    inter_names = [f"{c}_x_{m}" for m in macro_cols for c in char_cols]
    return char_cols + inter_names + ind_cols


def main():
    print("Loading predictions ...")
    preds = load_all_preds()
    if not preds:
        print("No predictions found. Run 03_rolling_train.py first.")
        return

    print("\nLoading market equity ...")
    me_df = load_me()

    # Build feature column names for importance plots (matches 03_rolling_train.py build_X order)
    feat_cols = _build_feat_names()
    print(f"  Feature columns for importance plots: {len(feat_cols)}")

    # ── Table 1 ──────────────────────────────────────────────────────────────
    print("\nComputing Table 1: OOS R² …")
    t1 = compute_r2_table(preds)
    save_table(t1, "Table1_OOS_R2")

    # ── Table 3: DM tests ────────────────────────────────────────────────────
    print("\nComputing Table 3: Diebold-Mariano tests …")
    try:
        t3 = compute_dm_table(preds)
        save_table(t3, "Table3_DM_Tests")
    except Exception as e:
        print(f"  DM test failed: {e}")

    # ── Table 7: Decile returns ───────────────────────────────────────────────
    print("\nComputing Table 7: Decile Portfolio Returns …")
    try:
        t7 = compute_decile_table(preds, me_df)
        save_table(t7, "Table7_Decile_Returns")
    except Exception as e:
        print(f"  Table 7 failed: {e}")

    # ── Table 8: Portfolio performance ───────────────────────────────────────
    print("\nComputing Table 8: Portfolio Performance …")
    try:
        t8 = compute_perf_table(preds, me_df)
        save_table(t8, "Table8_Portfolio_Performance")
    except Exception as e:
        print(f"  Table 8 failed: {e}")

    # ── Figure: Cumulative returns ────────────────────────────────────────────
    print("\nPlotting cumulative returns …")
    fig_cumulative_returns(preds, me_df,
                           os.path.join(OUT_DIR, "Fig_Cumulative_Returns.png"))

    # ── Figure: Monthly R² time series ───────────────────────────────────────
    fig_r2_time_series(preds,
                       os.path.join(OUT_DIR, "Fig_R2_TimeSeries.png"))

    # ── Figure 4: Variable importance ────────────────────────────────────────
    print("\nPlotting variable importance …")
    fig_variable_importance(preds, feat_cols,
                            os.path.join(OUT_DIR, "Fig4_Variable_Importance.png"))

    # ── Figure 5: Rank importance ─────────────────────────────────────────────
    fi_data = {}
    for fname_key in ['gbrt_h', 'rf']:
        fi_files = sorted([f for f in os.listdir(PRED_DIR)
                           if f.startswith(f"{fname_key}_fi_") and f.endswith('.npy')])
        if fi_files:
            fi_arr = np.stack([np.load(os.path.join(PRED_DIR, f))
                               for f in fi_files], axis=0)
            fi_data[fname_key] = fi_arr.mean(axis=0)
    fig_rank_importance(fi_data, feat_cols,
                        os.path.join(OUT_DIR, "Fig5_Rank_Importance.png"))

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
