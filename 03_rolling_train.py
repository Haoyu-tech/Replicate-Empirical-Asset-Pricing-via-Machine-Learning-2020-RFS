"""
Step 3: Rolling Window Training
Gu, Kelly, Xiu (2020) replication.

Rolling window scheme per paper Section 2.3:
  Train:      1957 ... (test_year - 13)   [expanding]
  Validation: (test_year - 12) ... (test_year - 1)   [12-year rolling]
  Test:       test_year   [1987 ... 2016]
  Refit:      annually

Feature matrix built on-the-fly each year:
  94 chars  +  94×8 interactions  +  ~74 industry dummies  =  ~920 total
"""

import os, gc, time
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = r"D:\桌面文件\US stock research\chapter2\ReplicateXiuML"
DATA_PATH = os.path.join(BASE_DIR, "data", "data_clean.parquet")
PRED_DIR  = os.path.join(BASE_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

import sys
sys.path.insert(0, BASE_DIR)
from models_config import (
    OLS3Model, OLSHuberModel, ENetHuberModel, PCRModel, PLSModel,
    GBRTHuberModel, RFModel, NNModel,
    ENET_GRID, PCR_GRID, PLS_GRID, GBRT_GRID, RF_GRID, NN_L1_GRID,
    r2_oos, huber_loss_val,
)

# ── Configuration ─────────────────────────────────────────────────────────────
TRAIN_START  = "1957-01-01"
VAL_YEARS    = 12
TEST_START_Y = 1987
TEST_END_Y   = 2016

RUN_OLS3   = True
RUN_OLS_H  = True
RUN_ENET_H = True
RUN_PCR    = True
RUN_PLS    = True
RUN_GBRT_H = True
RUN_RF     = True
RUN_NN     = True


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    print("Loading cleaned data ...")
    df = pd.read_parquet(DATA_PATH)
    df['eom'] = pd.to_datetime(df['eom'])
    df = df.sort_values(['eom', 'permno']).reset_index(drop=True)

    # Read characteristic names saved by 01_data_prep.py
    chars_path = os.path.join(BASE_DIR, "data", "paper_chars.txt")
    with open(chars_path) as f:
        char_cols = [l.strip() for l in f if l.strip()]

    # Macro columns stored with m_ prefix
    macro_cols = [c for c in df.columns if c.startswith('m_')]
    # Industry dummy columns
    ind_cols   = [c for c in df.columns if c.startswith('ind_sic2_')]

    print(f"  Chars: {len(char_cols)}, Macro: {len(macro_cols)}, "
          f"Ind dummies: {len(ind_cols)}, Obs: {len(df):,}")
    return df, char_cols, macro_cols, ind_cols


# ── Feature matrix ────────────────────────────────────────────────────────────

def build_X(sub: pd.DataFrame, char_cols, macro_cols, ind_cols) -> np.ndarray:
    """
    Assemble 920-feature matrix from stored columns.
    94 chars  +  94×8 interaction terms  +  ~74 industry dummies
    """
    char_arr  = sub[char_cols].values.astype('float32')   # (N, 94)
    macro_arr = sub[macro_cols].values.astype('float32')  # (N, 8)
    ind_arr   = sub[ind_cols].values.astype('float32')    # (N, ~74)

    # char_i × macro_j for all i,j  →  (N, 752)
    inter = np.hstack([char_arr * macro_arr[:, j:j+1]
                       for j in range(macro_arr.shape[1])])

    return np.hstack([char_arr, inter, ind_arr])          # (N, ~920)


def build_feat_names(char_cols, macro_cols, ind_cols):
    """Column names for the 920-feature matrix."""
    inter_names = [f"{c}_x_{m}" for m in macro_cols for c in char_cols]
    return char_cols + inter_names + ind_cols


# ── Rolling splits ────────────────────────────────────────────────────────────

def get_splits(df, test_year):
    eom = df['eom']
    train_end  = pd.Timestamp(f"{test_year - VAL_YEARS - 1}-12-31")
    val_start  = pd.Timestamp(f"{test_year - VAL_YEARS}-01-01")
    val_end    = pd.Timestamp(f"{test_year - 1}-12-31")
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end   = pd.Timestamp(f"{test_year}-12-31")
    return (
        (eom >= TRAIN_START) & (eom <= train_end),
        (eom >= val_start)   & (eom <= val_end),
        (eom >= test_start)  & (eom <= test_end),
    )


def get_arrays(df, mask, char_cols, macro_cols, ind_cols):
    sub   = df[mask & df['ret_exc_lead1m'].notna()].copy()
    X     = build_X(sub, char_cols, macro_cols, ind_cols)
    y     = sub['ret_exc_lead1m'].values.astype('float32')
    idx   = sub.index.values
    perms = sub['permno'].values
    eoms  = sub['eom'].values
    return X, y, idx, perms, eoms


def fill_na_median(X_tr, X_va, X_te):
    medians = np.nanmedian(X_tr, axis=0)
    for X in [X_tr, X_va, X_te]:
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    return X_tr, X_va, X_te


# ── Prediction I/O ────────────────────────────────────────────────────────────

def load_preds(name):
    path = os.path.join(PRED_DIR, f"{name}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame(columns=['permno', 'eom', 'y_true', 'y_pred'])


def save_preds(name, df):
    df.to_parquet(os.path.join(PRED_DIR, f"{name}.parquet"), index=False)


def append_preds(name, perms, eom_vals, y_true, y_pred):
    existing = load_preds(name)
    new = pd.DataFrame({
        'permno': perms,
        'eom':    pd.to_datetime(eom_vals),
        'y_true': y_true.astype('float32'),
        'y_pred': y_pred.astype('float32'),
    })
    save_preds(name, pd.concat([existing, new], ignore_index=True))


def done_years(name):
    ex = load_preds(name)
    if len(ex) == 0:
        return set()
    return set(pd.to_datetime(ex['eom']).dt.year.unique())


# ── OLS-3 (size, value, momentum) ────────────────────────────────────────────

def run_ols3(df, char_cols, macro_cols, ind_cols):
    name = 'ols3'
    print(f"\n{'='*60}\nModel: OLS-3")
    dy = done_years(name)

    # OLS-3 only uses mvel1, bm, mom12m — build tiny 3-col X directly
    THREE = ['mvel1', 'bm', 'mom12m']
    three_idx = [char_cols.index(c) for c in THREE if c in char_cols]

    for yr in range(TEST_START_Y, TEST_END_Y + 1):
        if yr in dy:
            print(f"  {yr}: skip"); continue
        t0 = time.time()
        tr, va, te = get_splits(df, yr)

        def get3(mask):
            sub = df[mask & df['ret_exc_lead1m'].notna()]
            X3 = sub[THREE].values.astype('float32')
            y  = sub['ret_exc_lead1m'].values.astype('float32')
            return X3, y, sub['permno'].values, sub['eom'].values

        X_tr, y_tr, _, _      = get3(tr)
        X_va, y_va, _, _      = get3(va)
        X_te, y_te, p_te, e_te = get3(te)

        # fill NaN with train median
        meds = np.nanmedian(X_tr, axis=0)
        for X in [X_tr, X_va, X_te]:
            nm = np.isnan(X)
            if nm.any(): X[nm] = np.take(meds, np.where(nm)[1])

        X_tv = np.vstack([X_tr, X_va]); y_tv = np.concatenate([y_tr, y_va])
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_tv, y_tv)
        y_pred = model.predict(X_te)
        append_preds(name, p_te, e_te, y_te, y_pred)
        print(f"  {yr}: R2={r2_oos(y_te,y_pred)*100:.3f}%  [{time.time()-t0:.0f}s]")
        del X_tr, X_va, X_te, X_tv; gc.collect()


# ── OLS + Huber ───────────────────────────────────────────────────────────────

def run_ols_h(df, char_cols, macro_cols, ind_cols):
    name = 'ols_h'
    print(f"\n{'='*60}\nModel: OLS+H")
    dy = done_years(name)

    for yr in range(TEST_START_Y, TEST_END_Y + 1):
        if yr in dy:
            print(f"  {yr}: skip"); continue
        t0 = time.time()
        tr, va, te = get_splits(df, yr)
        X_tr, y_tr, _, _, _       = get_arrays(df, tr, char_cols, macro_cols, ind_cols)
        X_va, y_va, _, _, _       = get_arrays(df, va, char_cols, macro_cols, ind_cols)
        X_te, y_te, _, p_te, e_te = get_arrays(df, te, char_cols, macro_cols, ind_cols)
        X_tr, X_va, X_te = fill_na_median(X_tr, X_va, X_te)

        X_tv = np.vstack([X_tr, X_va]); y_tv = np.concatenate([y_tr, y_va])
        model = OLSHuberModel()
        model.fit(X_tv, y_tv)
        y_pred = model.predict(X_te)
        append_preds(name, p_te, e_te, y_te, y_pred)
        print(f"  {yr}: R2={r2_oos(y_te,y_pred)*100:.3f}%  [{time.time()-t0:.0f}s]")
        del X_tr, X_va, X_te, X_tv; gc.collect()


# ── Generic tuned model (ENet, PCR, PLS, GBRT, RF) ───────────────────────────

def run_tuned(name, ModelClass, grid, df, char_cols, macro_cols, ind_cols):
    print(f"\n{'='*60}\nModel: {name}")
    dy = done_years(name)

    for yr in range(TEST_START_Y, TEST_END_Y + 1):
        if yr in dy:
            print(f"  {yr}: skip"); continue
        t0 = time.time()
        tr, va, te = get_splits(df, yr)
        X_tr, y_tr, _, _, _       = get_arrays(df, tr, char_cols, macro_cols, ind_cols)
        X_va, y_va, _, _, _       = get_arrays(df, va, char_cols, macro_cols, ind_cols)
        X_te, y_te, _, p_te, e_te = get_arrays(df, te, char_cols, macro_cols, ind_cols)
        X_tr, X_va, X_te = fill_na_median(X_tr, X_va, X_te)

        # Grid search on validation set (Huber loss)
        best_loss, best_params = np.inf, grid[0]
        for params in grid:
            try:
                m = ModelClass(**params)
                m.fit(X_tr, y_tr)
                loss = huber_loss_val(y_va, m.predict(X_va))
                if loss < best_loss:
                    best_loss, best_params = loss, params
            except Exception:
                continue

        # Refit on train + val
        X_tv = np.vstack([X_tr, X_va]); y_tv = np.concatenate([y_tr, y_va])
        model = ModelClass(**best_params)
        model.fit(X_tv, y_tv)
        y_pred = model.predict(X_te)

        if hasattr(model, 'feature_importances') and model.feature_importances() is not None:
            np.save(os.path.join(PRED_DIR, f"{name}_fi_{yr}.npy"),
                    model.feature_importances())

        append_preds(name, p_te, e_te, y_te, y_pred)
        print(f"  {yr}: R2={r2_oos(y_te,y_pred)*100:.3f}%  params={best_params}  [{time.time()-t0:.0f}s]")
        del X_tr, X_va, X_te, X_tv; gc.collect()


# ── Neural networks ───────────────────────────────────────────────────────────

def run_nn(n_layers, df, char_cols, macro_cols, ind_cols):
    name = f"nn{n_layers}"
    print(f"\n{'='*60}\nModel: NN{n_layers}")
    dy = done_years(name)

    for yr in range(TEST_START_Y, TEST_END_Y + 1):
        if yr in dy:
            print(f"  {yr}: skip"); continue
        t0 = time.time()
        tr, va, te = get_splits(df, yr)
        X_tr, y_tr, _, _, _       = get_arrays(df, tr, char_cols, macro_cols, ind_cols)
        X_va, y_va, _, _, _       = get_arrays(df, va, char_cols, macro_cols, ind_cols)
        X_te, y_te, _, p_te, e_te = get_arrays(df, te, char_cols, macro_cols, ind_cols)
        X_tr, X_va, X_te = fill_na_median(X_tr, X_va, X_te)

        # Tune L1 penalty on val set; for each lambda train 10-net ensemble
        best_val_loss, best_l1, best_epochs = np.inf, NN_L1_GRID[0], 100
        for l1 in NN_L1_GRID:
            try:
                m = NNModel(n_layers=n_layers, lr=1e-3, l1_lambda=l1,
                            max_epochs=100, batch_size=10000, patience=5)
                m.fit(X_tr, y_tr, Xva=X_va, yva=y_va)
                vp = m.predict(X_va)
                vloss = float(np.mean((y_va - vp)**2))
                if vloss < best_val_loss:
                    best_val_loss = vloss
                    best_l1 = l1
                    best_epochs = getattr(m, '_best_epoch', 100)
            except Exception as e:
                print(f"    NN{n_layers} l1={l1} error: {e}")

        # Refit on train+val with best hyperparams
        X_tv = np.vstack([X_tr, X_va]); y_tv = np.concatenate([y_tr, y_va])
        model = NNModel(n_layers=n_layers, lr=1e-3, l1_lambda=best_l1,
                        max_epochs=best_epochs, batch_size=10000, patience=999)
        model.fit(X_tv, y_tv)
        y_pred = model.predict(X_te)

        append_preds(name, p_te, e_te, y_te, y_pred)
        print(f"  {yr}: R2={r2_oos(y_te,y_pred)*100:.3f}%  l1={best_l1}  [{time.time()-t0:.0f}s]")
        del X_tr, X_va, X_te, X_tv; gc.collect()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    df, char_cols, macro_cols, ind_cols = load_data()

    if RUN_OLS3:
        run_ols3(df, char_cols, macro_cols, ind_cols)

    if RUN_OLS_H:
        run_ols_h(df, char_cols, macro_cols, ind_cols)

    if RUN_ENET_H:
        run_tuned('enet_h', ENetHuberModel, ENET_GRID, df, char_cols, macro_cols, ind_cols)

    if RUN_PCR:
        run_tuned('pcr', PCRModel, PCR_GRID, df, char_cols, macro_cols, ind_cols)

    if RUN_PLS:
        run_tuned('pls', PLSModel, PLS_GRID, df, char_cols, macro_cols, ind_cols)

    if RUN_GBRT_H:
        run_tuned('gbrt_h', GBRTHuberModel, GBRT_GRID, df, char_cols, macro_cols, ind_cols)

    if RUN_RF:
        run_tuned('rf', RFModel, RF_GRID, df, char_cols, macro_cols, ind_cols)

    if RUN_NN:
        for n in range(1, 6):
            run_nn(n, df, char_cols, macro_cols, ind_cols)

    print("\nAll models done.")


if __name__ == "__main__":
    main()
