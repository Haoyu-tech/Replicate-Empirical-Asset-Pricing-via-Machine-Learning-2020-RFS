# Reproduction guide

## Environment and external inputs

Install [requirements.txt](../requirements.txt). The dependency list is unpinned pending a validated reference environment.

The study requires the stock-characteristic CSV, macro-predictor CSV and WRDS archive described in the [README](../README.md). They are external inputs.

## Current main-branch path setup

The original scripts retain machine-specific paths. Before execution:

1. Set `DATASHARE_PATH`, `ZIP_PATH`, `MACRO_PATH` and `OUT_DIR` in `01_data_prep.py`.
2. Set `BASE_DIR` consistently in `03_rolling_train.py` and `04_results.py`.
3. Set `BASE` in `run_all.py` to the same local project folder.

Repository-relative paths and environment-variable overrides are proposed in the separate cleanup PR; they are not part of this documentation-only main update.

## Execution order

```powershell
python -m pip install -r requirements.txt
python 01_data_prep.py
python 03_rolling_train.py
python 04_results.py
```

Alternatively use `python run_all.py` after configuring its base path. Inspect model flags before launching a long run.

## Outputs and validation

Prepared data, predictions and tables/figures are written under the configured local base. Match data versions, preprocessing, model flags, random seeds and window definitions when comparing results. No full training run was performed during documentation editing.
