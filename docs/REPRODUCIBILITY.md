# Reproduction guide

## Environment and inputs

Install [requirements.txt](../requirements.txt). The list follows the committed source/README and is unpinned; it is not a lock from a validated experiment.

Default input paths are anchored to the repository root:

| Input | Default |
|---|---|
| Characteristics | `data/datashare.csv` |
| Macro predictors | `data/macro_clean.csv` |
| WRDS archive | `wrds/Kelly_factor_153.parquet.zip` |

Override inputs with `GKX_DATASHARE_PATH`, `GKX_MACRO_PATH`, and `GKX_WRDS_ZIP_PATH`. Generated outputs remain under this checkout to keep the preparation, training and reporting stages consistent.

## Execution order

```powershell
python -m pip install -r requirements.txt
python 01_data_prep.py
python 03_rolling_train.py
python 04_results.py
```

Alternatively `python run_all.py` runs the same three stages. Inspect model flags in `03_rolling_train.py` before a long run.

## Outputs and validation

Prepared data is written under `data/`, predictions under `predictions/`, and tables/figures under `output/`. The archive does not include all raw inputs.

Record raw-data versions, preprocessing choices, training windows, seeds and model flags. README sample sizes, timing and results are existing claims, not newly verified outputs. This cleanup checks source syntax and path configuration; it does not rerun decades of rolling training.

## Run record

For every result you intend to compare or share, retain the repository commit, exact command/configuration, input-data identity, environment versions and output location. Save a dependency snapshot after a successful reference run; newly added requirements lists are intentionally unpinned until that validation exists.

## Validation status

Cleanup review: 14 October 2026. File paths and documented entry points were checked against the checkout. Full empirical results were not regenerated. Project-specific external prerequisites and known blockers are listed above.
