# Changelog

All notable changes by version.

## [v1.0.0] — 2023‑05‑01
### Baseline
- First release of basic data processing scripts for the dissertation:
- DOI: 10.1016/j.imu.2023.101428  
  - Code: https://github.com/AikoIs/Data.git  
  - MIMIC‑III (Labevents): https://mit-lcp.github.io/mimic-schema-spy/tables/labevents.html

## [v1.1.0] — 2024‑07‑15
### Switch to MIMIC‑IV
- Updated data source: https://physionet.org/content/mimic4wdb/0.1.0/
- In the modules `eda_improved.py`, `preprocessing_improved.py`, `feature_engineering_improved.py`, `model_training_improved.py`, adaptation to the new table structure.

## [v1.2.0] — 2024‑08‑01
### Preprocessing improvements
- Filtering outliers by IQR
- Log transformations of highly skewed features
- Centralized scaling (Min–Max, RobustScaler)

## [v1.3.0] — 2024‑09‑01
### New EDA modules
- `eda_improved.py`:
  - visualization of time series by key labels
  - reports on missing values and correlations

## [v1.4.0] — 2024‑10‑01
### Model training
- Added `model_training_improved.py`:
  - support for CatBoost, XGBoost
  - unified interface for comparing algorithms