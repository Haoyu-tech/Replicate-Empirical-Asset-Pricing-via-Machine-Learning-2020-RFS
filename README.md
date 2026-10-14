<div align="center">

# Empirical Asset Pricing via Machine Learning

Research replication of Gu, Kelly and Xiu (2020).

![Models: Linear + Trees + Neural nets](https://img.shields.io/badge/Models-Linear%20%2B%20Trees%20%2B%20Neural%20nets-7C3AED?style=flat-square) ![Data: Monthly US equities](https://img.shields.io/badge/Data-Monthly%20US%20equities-0F766E?style=flat-square) ![Validation: Rolling windows](https://img.shields.io/badge/Validation-Rolling%20windows-3776AB?style=flat-square)

[Quick start](#quick-start) · [Project map](#project-map) · [Reproduction guide](docs/REPRODUCIBILITY.md)

</div>

---

## Overview

A modular replication workflow for monthly return prediction with stock characteristics, macroeconomic interactions and industry indicators. Model families include linear estimators, dimension reduction, trees and neural-network ensembles.

## Experiment design

| Component | Original configuration described in the project |
|---|---|
| Equity sample | US common stocks, March 1957–December 2016 |
| Target | Next-month excess return |
| Features | 94 characteristics, 8 macro interaction variables and industry dummies |
| Training | Expanding window |
| Validation | 12-year rolling window before the test year |
| Testing | One calendar year at a time, 1987–2016 |

## Quick start

Prepare the external characteristics, macro-predictor CSV and WRDS archive, then configure the original scripts' paths:

| Script | Paths to configure |
|---|---|
| `01_data_prep.py` | `DATASHARE_PATH`, `ZIP_PATH`, `MACRO_PATH` and `OUT_DIR` |
| `03_rolling_train.py` / `04_results.py` | `BASE_DIR` |
| `run_all.py` | `BASE` |

The current main-branch scripts contain machine-specific paths. Point every stage at the same local project and prepared panel before running.

```powershell
python -m pip install -r requirements.txt
python run_all.py
```

The raw inputs are external. Training can be expensive; inspect the model flags before a full run.

## Project map

| Script | Output |
|---|---|
| [01_data_prep.py](01_data_prep.py) | Prepared panel under `data/` |
| [models_config.py](models_config.py) | Model implementations and hyperparameter grids |
| [03_rolling_train.py](03_rolling_train.py) | Per-model predictions under `predictions/` |
| [04_results.py](04_results.py) | Tables and figures under `output/` |
| [run_all.py](run_all.py) | Preparation → training → reporting |

## Reproduction notes

Keep preparation, training and reporting pointed at the same local artifact locations. The dependency list is unpinned pending a validated reference run.

Match preprocessing, target alignment, model flags and evaluation definitions before comparing with the paper. Detailed model settings and the original methodology notes are preserved in [the project reference](docs/PROJECT_REFERENCE.md).

## Reference

Gu, S., Kelly, B. and Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. The Review of Financial Studies, 33(5), 2223–2273. DOI: 10.1093/rfs/hhaa009.

## Documentation

- [Reproduction guide](docs/REPRODUCIBILITY.md): inputs, execution order and validation limits.
- [Code and artifact map](docs/REPOSITORY_MAP.md): source and configuration paths.

<sub>Research artifacts reflect the recorded experiment settings. Full results were not regenerated during documentation cleanup.</sub>
