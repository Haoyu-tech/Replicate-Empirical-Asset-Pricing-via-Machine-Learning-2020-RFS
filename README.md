# Replication: Gu, Kelly, Xiu (2020)
## "Empirical Asset Pricing via Machine Learning"
*Journal of Financial Studies*, 33(5), 2020

---

## Overview

This project replicates the main tables and figures of Gu, Kelly & Xiu (2020), which benchmarks 13 machine learning models for predicting US stock returns out-of-sample.

**Sample:** NYSE / AMEX / NASDAQ common stocks, March 1957 – December 2016  
**Target variable:** Monthly excess return at t+1  
**Feature set:** 920 predictors = 94 stock characteristics × (1 + 8 macro interactions) + 74 SIC-2 industry dummies  

---

## Data Sources

| File | Source | Description |
|------|--------|-------------|
| `data/datashare.csv` | [Dacheng Xiu's website](https://dachxiu.chicagobooth.edu/) | Official 94 stock characteristics, 1957–2016 |
| `wrds/Kelly_factor_153.parquet.zip` | WRDS | Returns, market equity, universe filters |
| `data/macro_clean.csv` | Welch & Goyal (2008) via Tidy Finance | 8 macroeconomic predictors |

### 94 Stock Characteristics
Exactly as in Green, Hand & Zhang (2017). See `data/paper_chars.txt` for the full list.

### 8 Macro Variables (Welch-Goyal)
`dp`, `ep`, `bm`, `ntis`, `tbl`, `tms`, `dfy`, `svar`  
Stored with `m_` prefix (`m_dp`, `m_ep`, …) to avoid name collision with stock characteristics.

---

## Models (13 total)

| Model | Type | Key hyperparameters |
|-------|------|---------------------|
| OLS-3 | Linear | 3 predictors: size, value, momentum |
| OLS+H | Linear | Huber loss (ε=1.35), no penalty |
| ENet+H | Linear | Elastic net + Huber loss; α ∈ {1e-4…1}, L1 ratio ∈ {0.1, 0.5, 0.9} |
| PLS | Dim. reduction | n_components ∈ {1, 2, 3, 5} |
| PCR | Dim. reduction | n_components ∈ {3, 5, 10, 20, 50} |
| GBRT+H | Tree | Depth ∈ {1–6}, lr ∈ {0.01, 0.1}, leaf ∈ {500, 1000} |
| RF | Tree | min_samples_leaf ∈ {500, 1000, 2000} |
| NN1–NN5 | Neural net | Depth 1–5; widths 32→16→8→4→2; L1 penalty, 10-ensemble |

---

## Rolling Window Scheme

```
|<──── Train (expanding) ────>|<── Val (12 yr) ──>|< Test >|
 1957                       t-13              t-1     t
```

- **Train:** all data from 1957 to (test_year − 13)
- **Validation:** 12-year rolling window preceding test year
- **Test:** one calendar year at a time, 1987–2016 (30 years)
- Hyperparameters selected by minimising Huber loss on validation set
- Model refitted on train + val before test prediction

---

## File Structure

```
ReplicateXiuML/
├── 01_data_prep.py        # Data cleaning & feature construction → data_clean.parquet
├── models_config.py       # All 13 model classes + hyperparameter grids
├── 03_rolling_train.py    # Rolling-window training for all models → predictions/
├── 04_results.py          # Tables 1, 3, 7, 8 + all figures → output/
├── run_all.py             # Runs steps 1, 3, 4 in sequence
│
├── data/
│   ├── datashare.csv      # Official Xiu characteristics (1.5 GB)
│   ├── macro_clean.csv    # 8 Welch-Goyal macro variables (1957–2016)
│   ├── data_clean.parquet # Cleaned panel: 3.27M obs × 181 cols (snappy)
│   └── paper_chars.txt    # 94 characteristic names (one per line)
│
├── predictions/           # Per-model parquet files (cols: permno, eom, y_true, y_pred)
│   ├── ols3.parquet
│   ├── ols_h.parquet
│   └── …
│
└── output/                # Tables (CSV) and figures (PNG)
    ├── Table1_OOS_R2.csv
    ├── Table3_DM_Tests.csv
    ├── Table7_Decile_Returns.csv
    ├── Table8_Portfolio_Performance.csv
    └── Fig_*.png
```

---

## How to Run

### Step 0 — Install dependencies
```
pip install pandas numpy pyarrow scikit-learn torch tqdm matplotlib scipy
```

### Step 1 — Data preparation (~5 min)
```
python 01_data_prep.py
```
Outputs `data/data_clean.parquet` (181 columns, 3.27M rows).

### Step 2 — Rolling-window training (hours to days depending on hardware)
```
python 03_rolling_train.py
```
Saves one parquet file per model to `predictions/`.  
Models with `RUN_*` flags at the top can be toggled on/off.

> **Memory note:** The 920-feature matrix is built on-the-fly each year to avoid storing the 94×8=752 interaction columns (would require ~10 GB). Peak RAM usage is ~6 GB for the largest training window (year 2016).

### Step 3 — Results & figures
```
python 04_results.py
```
Reads `predictions/*.parquet`, writes tables and figures to `output/`.

### Run all steps
```
python run_all.py
```

---

## Key Implementation Details

### Feature construction (per paper Section 2.1)
1. **Cross-sectional rank normalisation** of 94 characteristics each month → [-1, 1]
2. **Missing values** → filled with cross-sectional median before ranking (= 0 after)
3. **Interaction terms** `char_i × macro_j` (94 × 8 = 752 columns) computed at training time to save disk space
4. **Industry dummies:** SIC-2 level, most frequent category dropped as reference

### Neural network architecture (Section 3.5)
- NN1: P→32→1 | NN2: P→32→16→1 | … | NN5: P→32→16→8→4→2→1
- BatchNorm + ReLU at each hidden layer
- L1 regularisation on all weight matrices (λ ∈ {1e-5, 1e-4, 1e-3})
- 10-net ensemble; early stopping with patience = 5 on validation MSE

### Out-of-sample R²
```
R²_OOS = 1 - SS_res / SS_tot
```
where SS_tot = Σ(y²) (benchmark: zero return, *not* the historical mean).

---

## Reference

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning.  
*The Review of Financial Studies*, 33(5), 2223–2273.  
https://doi.org/10.1093/rfs/hhaa009
