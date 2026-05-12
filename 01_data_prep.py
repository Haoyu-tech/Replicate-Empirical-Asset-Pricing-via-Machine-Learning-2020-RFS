"""
Step 1: Data Preparation
Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning" replication.

Uses official Xiu datashare.csv (94 exact characteristics) merged with
WRDS Kelly_factor_153 (returns, ME, universe filters).

Output data_clean.parquet stores:
  94 stock characteristics (rank-normalised to [-1,1])
  8 Welch-Goyal macro variables  (m_dp, m_ep, m_bm, …)
  ~74 SIC-2 industry dummies     (ind_sic2_*)

Interaction terms (94×8 = 752) are built on-the-fly during training to
avoid the ~10 GB memory spike of storing them in the cleaned file.
"""

import zipfile, io, os, gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASHARE_PATH = r"D:\桌面文件\US stock research\chapter2\ReplicateXiuML\data\datashare.csv"
ZIP_PATH       = r"D:\桌面文件\US stock research\wrds\Kelly_factor_153.parquet.zip"
MACRO_PATH     = r"D:\桌面文件\US stock research\chapter2\ReplicateXiuML\data\macro_clean.csv"
OUT_DIR        = r"D:\桌面文件\US stock research\chapter2\ReplicateXiuML\data"
CLEAN_PATH     = os.path.join(OUT_DIR, "data_clean.parquet")

TRAIN_START = "1957-01-01"
TEST_END    = "2016-12-31"

# 8 Welch-Goyal macro variable names (m_ prefix avoids clash with chars ep/bm)
MACRO_VARS     = ['m_dp', 'm_ep', 'm_bm', 'm_ntis', 'm_tbl', 'm_tms', 'm_dfy', 'm_svar']
_MACRO_CSV_COLS = ['dp',   'ep',   'bm',   'ntis',   'tbl',   'tms',   'dfy',   'svar']


# ── Load official datashare.csv ───────────────────────────────────────────────

def load_datashare() -> tuple[pd.DataFrame, list]:
    """
    Load Xiu datashare.csv.
    Returns (df, paper_chars) where paper_chars is the exact list of 94 names.
    """
    print("Reading datashare.csv …")
    ds = pd.read_csv(DATASHARE_PATH, low_memory=False)
    print(f"  Raw shape: {ds.shape}")

    # DATE is YYYYMMDD integer → parse to month-end datetime
    ds['eom'] = pd.to_datetime(ds['DATE'].astype(str), format='%Y%m%d')
    ds['eom'] = ds['eom'] + pd.offsets.MonthEnd(0)   # snap to month-end
    ds = ds.drop(columns=['DATE'])

    # date filter
    ds = ds[(ds['eom'] >= TRAIN_START) & (ds['eom'] <= TEST_END)].copy()
    print(f"  After date filter: {len(ds):,} rows")

    # identify the 94 characteristic columns
    non_chars = {'permno', 'eom', 'sic2'}
    paper_chars = [c for c in ds.columns if c not in non_chars]
    print(f"  Characteristics found: {len(paper_chars)}")
    return ds, paper_chars


# ── Load WRDS returns & universe flags ───────────────────────────────────────

_WRDS_COLS = ['permno', 'eom', 'excntry', 'obs_main', 'common',
              'primary_sec', 'ret_exc_lead1m', 'ret_exc', 'me']

def load_wrds_returns() -> pd.DataFrame:
    """Load minimal columns from WRDS parquet for returns and filters."""
    print("Reading WRDS parquet in chunks …")
    chunks = []
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        raw = z.read(z.namelist()[0])
    buf = io.BytesIO(raw)
    pf  = pq.ParquetFile(buf)

    # Load only what we need
    available = pf.schema_arrow.names
    cols = [c for c in _WRDS_COLS if c in available]

    for batch in tqdm(pf.iter_batches(batch_size=200_000, columns=cols),
                      desc="WRDS chunks"):
        df = batch.to_pandas()
        mask = (
            (df['excntry']     == 'USA') &
            (df['obs_main']    == 1) &
            (df['common']      == 1) &
            (df['primary_sec'] == 1)
        )
        df = df[mask].copy()
        if len(df) == 0:
            continue
        df['eom'] = pd.to_datetime(df['eom'])
        df = df[(df['eom'] >= TRAIN_START) & (df['eom'] <= TEST_END)]
        if len(df) > 0:
            chunks.append(df[['permno', 'eom', 'ret_exc_lead1m', 'ret_exc', 'me']])

    wrds = pd.concat(chunks, ignore_index=True)
    wrds = wrds.sort_values(['eom', 'permno']).reset_index(drop=True)
    print(f"  WRDS US common stocks: {len(wrds):,} rows")
    return wrds


# ── Rank normalisation to [-1, 1] ────────────────────────────────────────────

def rank_normalize_chars(df: pd.DataFrame, char_cols: list) -> pd.DataFrame:
    """
    Cross-sectional rank each characteristic each month → [-1, 1].
    Missing values are first filled with cross-sectional median (→ 0 after rank).
    Uses numpy loop for memory efficiency.
    """
    eom_arr = df['eom'].values
    X = df[char_cols].values.astype('float32')

    for j in tqdm(range(len(char_cols)), desc="Rank-normalizing"):
        col_vals = X[:, j]
        for eom_val in np.unique(eom_arr):
            mask = eom_arr == eom_val
            v = col_vals[mask]
            if np.any(np.isnan(v)):
                med = np.nanmedian(v)
                v = np.where(np.isnan(v), 0.0 if np.isnan(med) else med, v)
            n = len(v)
            if n > 1:
                order = np.argsort(v)
                ranks = np.empty(n, dtype='float32')
                ranks[order] = np.arange(1, n + 1, dtype='float32')
                v = (ranks / n) * 2.0 - 1.0
            col_vals[mask] = v
        X[:, j] = col_vals

    df = df.copy()
    df[char_cols] = X
    return df


# ── Macro variables ───────────────────────────────────────────────────────────

def load_macro() -> pd.DataFrame:
    macro = pd.read_csv(MACRO_PATH)
    macro['date'] = pd.to_datetime(macro['date'])
    macro = macro.set_index('date')[_MACRO_CSV_COLS]
    macro.index = macro.index + pd.offsets.MonthEnd(0)
    macro.columns = MACRO_VARS      # rename → m_dp, m_ep, …
    return macro


def attach_macro(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    eom_idx = df['eom'].values
    for mv in MACRO_VARS:
        df[mv] = macro.reindex(eom_idx)[mv].values
    for mv in MACRO_VARS:
        df[mv] = df[mv].fillna(0.0).astype('float32')
    return df


# ── SIC-2 industry dummies ────────────────────────────────────────────────────

def create_industry_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Create SIC-2 dummies; drop the most frequent as reference category."""
    # datashare sic2 is already SIC-2 (two-digit)
    df['sic2'] = df['sic2'].fillna(0).astype(int)
    top = df['sic2'].value_counts().index.tolist()
    drop_cat = top[0]
    for s in top[1:]:
        df[f'ind_sic2_{s}'] = (df['sic2'] == s).astype('float32')
    print(f"  Industry dummies: {len(top)-1}  (ref SIC2={drop_cat})")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load official characteristics
    ds, paper_chars = load_datashare()

    # 2. Load WRDS returns + universe flags
    wrds = load_wrds_returns()

    # 3. Merge: inner join keeps only stocks in WRDS filtered universe
    print("Merging datashare + WRDS ...")
    df = pd.merge(wrds, ds, on=['permno', 'eom'], how='inner')
    df = df.sort_values(['eom', 'permno']).reset_index(drop=True)
    print(f"  Merged: {len(df):,} rows × {df.shape[1]} cols")
    del ds, wrds; gc.collect()

    # 4. Rank-normalize 94 characteristics to [-1, 1]
    print("Rank-normalizing characteristics …")
    df = rank_normalize_chars(df, paper_chars)

    # 5. Attach 8 macro variables
    print("Attaching macro variables …")
    macro = load_macro()
    df = attach_macro(df, macro)

    # 6. SIC-2 industry dummies
    print("Creating industry dummies …")
    df = create_industry_dummies(df)

    # 7. Build keep list
    ind_cols  = [c for c in df.columns if c.startswith('ind_sic2_')]
    feat_cols = paper_chars + MACRO_VARS + ind_cols
    total_feats = len(paper_chars) + len(paper_chars) * len(MACRO_VARS) + len(ind_cols)
    print(f"\n  Chars:        {len(paper_chars)}")
    print(f"  Interactions: {len(paper_chars)*len(MACRO_VARS)}  (computed at train time)")
    print(f"  Ind dummies:  {len(ind_cols)}")
    print(f"  TOTAL:        {total_feats}  (paper: 920)")

    keep = ['permno', 'eom', 'ret_exc_lead1m', 'ret_exc', 'me', 'sic2'] + feat_cols
    keep = [c for c in keep if c in df.columns]
    df   = df[keep].copy()

    # 8. Save
    print(f"\nSaving to {CLEAN_PATH} …")
    df.to_parquet(CLEAN_PATH, index=False, compression='snappy')
    print(f"Saved.  Shape: {df.shape}")
    print(f"Date range: {df['eom'].min().date()} → {df['eom'].max().date()}")
    print(f"Observations: {len(df):,}")
    print(f"Unique stocks: {df['permno'].nunique():,}")

    # Write PAPER_CHARS list to a helper file so other scripts can import it
    chars_path = os.path.join(OUT_DIR, "paper_chars.txt")
    with open(chars_path, 'w') as f:
        f.write('\n'.join(paper_chars))
    print(f"Characteristic names saved to {chars_path}")


if __name__ == "__main__":
    main()
