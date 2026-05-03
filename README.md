# BanglaYield-CP

BanglaYield-CP is a crop-yield prediction research repository built on the **SPAS-Dataset-BD** dataset. The project investigates how far classical machine learning and gradient-boosting regressors can predict **AP Ratio** as a proxy target for crop productivity across Bangladeshi districts, seasons, and crop types, with an emphasis on uncertainty-aware prediction under both i.i.d. and out-of-distribution settings.

The repository is structured as an end-to-end experimental workflow:

1. raw data inspection and cleanup,
2. preprocessing and normalization of agronomic calendar fields,
3. feature engineering for weather, seasonality, and crop-timing variables,
4. supervised regression under both standard and out-of-distribution evaluation settings,
5. visual analysis of model comparison, residual behavior, and generalization gaps.

## Project Scope

The central goal of this repository is to build a reproducible modeling pipeline for crop-yield estimation using structured tabular inputs derived from agricultural and weather-related descriptors. The project is not only concerned with in-distribution test performance; it also evaluates robustness under:

- **random i.i.d. splits**,
- **spatial out-of-distribution splits** across unseen districts,
- **season-shift evaluation** across Bangladesh's agronomic seasons.

This makes the repository more research-oriented than a standard leaderboard notebook, because it explicitly studies generalization under realistic deployment shifts.

## Repository Contents

```text
BanglaYield-CP/
├── data_handling.ipynb
├── dataset/
│   ├── SPAS-Dataset-BD.csv
│   ├── preprocessed_data.csv
│   └── crop_engineered.csv
├── models/
│   ├── catboost_apply_init.ipynb
│   ├── crop_ap_ratio_prediction.ipynb
│   └── regression_crop_pred.ipynb
├── plots/
│   ├── feature_importance.png
│   ├── gen_gap.png
│   ├── heatmaps.png
│   ├── model_comparison.png
│   ├── pred_vs_actual.png
│   └── residuals.png
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset Overview

### Primary Data Source

The raw dataset used in this repository is:

- [dataset/SPAS-Dataset-BD.csv](/Users/rishitkar/Desktop/BanglaYield-CP/dataset/SPAS-Dataset-BD.csv)
- SPAS-Dataset-BD (Mendeley Data): [https://data.mendeley.com/datasets/cphdw4z5kw/2](https://data.mendeley.com/datasets/cphdw4z5kw/2)

For the subset used in this repository, the available columns are:

- `Area`
- `AP Ratio`
- `District`
- `Season`
- `Avg Temp`
- `Avg Humidity`
- `Crop Name`
- `Transplant`
- `Growth`
- `Harvest`
- `Production`
- `Max Temp`
- `Min Temp`
- `Max Relative Humidity`
- `Min Relative Humidity`

### Observed Dataset Characteristics

Based on the files currently committed in the repository:

- Raw dataset size: **4,608 rows x 15 columns**
- Cleaned dataset size: **4,190 rows x 15 columns**
- Engineered dataset size: **4,190 rows x 24 columns**
- Districts in cleaned data: **64**
- Crop categories in cleaned data: **72**
- Agronomic seasons: **3** (`Rabi`, `Kharif 1`, `Kharif 2`)

### Cleaning Outcome

The preprocessing notebooks indicate that the raw dataset contains malformed target values in `AP Ratio`, including spreadsheet-style invalid entries such as `#DIV/0!`. These are coerced to numeric, converted to missing values, and removed during preprocessing.

Repository-observed cleaning summary:

- Invalid `AP Ratio` entries coerced to `NaN`: **416**
- Total rows removed from raw to cleaned dataset: **418**
- Missing values in final cleaned dataset: **0**

## Data Lineage

The repository follows a clear three-stage data lineage:

### 1. Raw dataset

- [dataset/SPAS-Dataset-BD.csv](/Users/rishitkar/Desktop/BanglaYield-CP/dataset/SPAS-Dataset-BD.csv)

This is the original source table containing raw categorical month strings, production values, and the original humidity column naming.

### 2. Cleaned / preprocessed dataset

- [dataset/preprocessed_data.csv](/Users/rishitkar/Desktop/BanglaYield-CP/dataset/preprocessed_data.csv)

This file is the cleaned modeling base table. The preprocessing notebook removes the `Production` field to avoid direct leakage into the target pipeline and standardizes calendar strings such as transplant, growth, and harvest month expressions.

Key changes visible in this artifact:

- `Production` removed
- relative humidity columns renamed to `Humidity Min` and `Humidity Max`
- `Humidity Range` added
- month labels normalized to short forms such as `Jan`, `Feb`, `Jun`, `Oct`
- invalid `AP Ratio` values removed

### 3. Engineered feature dataset

- [dataset/crop_engineered.csv](/Users/rishitkar/Desktop/BanglaYield-CP/dataset/crop_engineered.csv)

This is the feature-ready matrix exported by the main modeling notebook and used for regression experiments. It contains encoded and derived variables designed for seasonality-aware machine learning.

## Proposed Architecture

### 3.1 Problem Formulation

We address crop-yield prediction with uncertainty quantification using SPAS-Dataset-BD, a precision agriculture dataset for Bangladesh. Each record corresponds to a specific district, crop, and season, together with associated agronomic and environmental descriptors.

Let `x in R^d` denote the input feature vector and `y in R` denote the target. The prediction target is defined as:

- `y = AP Ratio`

In this project, `AP Ratio` is treated as a yield-per-unit-area target, approximately corresponding to `Production / Area`. To prevent label leakage, `AP Ratio` itself is excluded from the input feature set.

The supervised learning dataset is therefore represented as `{(x_i, y_i)}^N_{i=1}`, where:

- numeric features include `Area`, `Avg Temp`, `Avg Humidity`, `Max Temp`, `Min Temp`, `Max Relative Humidity`, and `Min Relative Humidity`,
- categorical features include `District`, `Season`, `Crop Name`, `Transplant`, `Growth`, and `Harvest`.

The goal is to learn a regression function `f : X -> R` for point prediction together with a calibrated prediction interval `[l(x), u(x)]` for uncertainty quantification.

### 3.2 Data Preprocessing and Splits

The repository follows a train, validation, calibration, and testing perspective for predictive modeling and uncertainty estimation.

Preprocessing consists of the minimum steps necessary to make the SPAS subset usable for tabular regression:

- basic consistency checks on numeric columns,
- coercion and removal of malformed `AP Ratio` values,
- normalization of calendar strings for `Transplant`, `Growth`, and `Harvest`,
- removal of `Production` from direct predictive inputs in the cleaned modeling table,
- optional scaling for non-tree models,
- categorical handling through either native categorical support or encoded representations, depending on model family.

Categorical variables (`District`, `Season`, `Crop Name`, `Transplant`, `Growth`, `Harvest`) are handled as follows:

- passed as categorical features to CatBoost,
- encoded for comparison models such as linear models, random forests, XGBoost, and LightGBM.

The evaluation design uses three regimes:

1. i.i.d. split: a random 70/15/15 train/validation/test split for model comparison and tuning.
2. Spatial OOD split: district-disjoint train, validation, and test partitions to measure generalization to unseen districts.
3. Temporal or seasonal OOD split: when explicit year information is unavailable, the repository uses agronomic season order as a temporal-structure proxy.

### 3.3 Base Regression Models

The project uses a small model zoo of tabular regressors, with **CatBoost Regressor** as the primary model.

Primary model:

- **CatBoost Regressor** using mixed numeric and categorical inputs, optimized with RMSE, early stopping, and tuned tree depth, learning rate, number of iterations, and regularization.

Baseline models currently reflected in the repository notebooks:

- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

The architecture text also allows for a multilayer perceptron baseline as an optional extension, but this is not currently implemented in the committed notebooks.

For each model and split, the repository reports standard regression metrics such as RMSE, MAE, and `R^2`.

### 3.4 Uncertainty Quantification via Conformal Prediction

The intended uncertainty layer of the architecture is **split conformal prediction**, implemented with MAPIE around the best-performing base regressor, typically CatBoost.

The conformal workflow is:

1. split the available training domain into a proper training set and a calibration set,
2. fit the base regressor on the proper training set only,
3. compute absolute residuals on the calibration set,
4. estimate a residual quantile for a target confidence level such as 90% or 95%,
5. construct prediction intervals of the form `[y_hat - q_alpha, y_hat + q_alpha]`.

This provides distribution-free finite-sample prediction intervals under standard exchangeability assumptions. The architecture also considers a naive Gaussian baseline using a global residual standard deviation, and optionally group-conditional conformal prediction to study subgroup-level coverage behavior across crop types or district clusters.

At present, the committed repository mainly documents the point-prediction pipeline and model-comparison notebooks. The README therefore describes conformal prediction as the intended architecture component rather than claiming it is already fully implemented in the checked-in code.

### 3.5 Evaluation Metrics

The architecture evaluates both predictive accuracy and uncertainty quality.

Point-prediction metrics:

- RMSE
- MAE
- `R^2`

Uncertainty metrics for conformal or Gaussian intervals:

- empirical coverage,
- average interval width,
- subgroup-wise coverage by crop type or district.

## Methodology

### 1. Preprocessing

The preprocessing logic is primarily documented in [data_handling.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/data_handling.ipynb). The notebook performs data inspection and normalization steps including:

- conversion of `AP Ratio` to numeric with coercion of invalid values,
- removal of rows with resulting missing targets,
- abbreviation and normalization of transplant, growth, and harvest month strings,
- removal of the `Production` column from the modeling table,
- humidity column cleanup for downstream consistency.

### 2. Feature Engineering

The main feature-engineering workflow is implemented in:

- [models/crop_ap_ratio_prediction.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/models/crop_ap_ratio_prediction.ipynb)
- [models/regression_crop_pred.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/models/regression_crop_pred.ipynb)

The engineered features fall into six groups.

### Thermal features

- `temp_diurnal_range = Max Temp - Min Temp`
- `temp_mean = (Max Temp + Min Temp) / 2`
- `gdd_proxy` as a simple growing-degree style proxy with base 10 C
- `svp_proxy` using a Magnus-style saturation vapor pressure approximation
- `vpd_proxy` as a vapor pressure deficit proxy

### Humidity features

- `humidity_mid`
- `Humidity Range`

### Crop-calendar features

- `transplant_month`
- `growth_start_month`
- `harvest_start_month`

These are derived by extracting ordinal month positions from the textual crop calendar fields.

### Season-structure features

- ordinal season encoding through an agronomic order: `Rabi -> 0`, `Kharif 1 -> 1`, `Kharif 2 -> 2`
- cyclic season encodings: `season_sin`, `season_cos`

### Interaction and scaling-oriented features

- `area_log`
- `area_x_diurnal`
- `temp_x_humidity`

### Encoded categorical variables

- `Season_enc`
- `Crop Name_enc`
- `Transplant_enc`
- `Growth_enc`
- `Harvest_enc`
- `District_enc`

### 3. Prediction Target

The prediction target used throughout the repository is:

- `AP Ratio`

From the committed cleaned dataset:

- minimum: **0.0**
- median: **2.5231**
- mean: **4.8058**
- 75th percentile: **4.7173**
- maximum: **951.2167**

The very large upper tail indicates a highly skewed target distribution, which helps explain why robust error analysis and residual inspection are important in this project.

## Evaluation Design

The strongest aspect of this repository is that it does not rely on a single train/test split. The main modeling notebook evaluates under three different regimes that match the proposed architecture at the experimental level.

### 1. Random split

Standard 70/15/15 row-wise split:

- Train: **2,933**
- Validation: **628**
- Test: **629**

This setting measures conventional i.i.d. predictive performance.

### 2. Spatial out-of-distribution split

Districts are partitioned into mutually exclusive train, validation, and test groups:

- Train districts: **44**
- Validation districts: **9**
- Test districts: **11**

Resulting row counts:

- Train: **2,879**
- Validation: **587**
- Test: **724**

This setting is intended to measure geographic generalization to unseen districts.

### 3. Season-shift out-of-distribution split

The repository defines an agronomic season-order split:

- Train: `Rabi` -> **1,192** rows
- Validation: `Kharif 1` -> **1,420** rows
- Test: `Kharif 2` -> **1,578** rows

This is a meaningful temporal-structure proxy even though no explicit year column exists in the dataset, and it is the repository's practical substitute for a true temporal OOD split.

## Model Families Evaluated

The notebooks indicate experiments with the following regressors:

- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

CatBoost appears in two roles:

- as an initial standalone baseline in [models/catboost_apply_init.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/models/catboost_apply_init.ipynb),
- and as an integrated comparator inside the main model-comparison notebook.

## Notebooks

### [data_handling.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/data_handling.ipynb)

Purpose:

- initial exploratory inspection of the raw dataset,
- cleaning and normalization of month/calendar fields,
- target validation and row removal.

### [models/catboost_apply_init.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/models/catboost_apply_init.ipynb)

Purpose:

- first CatBoost baseline on the cleaned dataset,
- sensitivity check after removing non-positive target rows,
- direct categorical handling via CatBoost's native categorical feature support,
- preliminary feature-importance inspection,
- residual and outlier error inspection.

Saved notebook output in the committed file shows:

- a baseline run using all cleaned rows,
- a second run on `AP Ratio > 0`,
- stronger performance after excluding zero-yield target cases,
- top CatBoost drivers including `Area`, `Max Temp`, `Avg Temp`, and `District`.

### [models/regression_crop_pred.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/models/regression_crop_pred.ipynb)

Purpose:

- feature engineering,
- engineered dataset export,
- classical regression benchmarking,
- random and spatial OOD evaluation,
- residual analysis and generalization-gap analysis.

This notebook appears to be an earlier comparative modeling pipeline.

### [models/crop_ap_ratio_prediction.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/models/crop_ap_ratio_prediction.ipynb)

Purpose:

- primary model-comparison notebook,
- full engineered-feature pipeline,
- random, spatial OOD, and season OOD evaluation,
- integrated CatBoost benchmark,
- consolidated visual reporting.

This is the most complete notebook in the repository and should be treated as the principal experimental entry point.

## Generated Figures

The repository already includes exported visual artifacts in [plots](/Users/rishitkar/Desktop/BanglaYield-CP/plots):

- [plots/model_comparison.png](/Users/rishitkar/Desktop/BanglaYield-CP/plots/model_comparison.png)
- [plots/heatmaps.png](/Users/rishitkar/Desktop/BanglaYield-CP/plots/heatmaps.png)
- [plots/pred_vs_actual.png](/Users/rishitkar/Desktop/BanglaYield-CP/plots/pred_vs_actual.png)
- [plots/feature_importance.png](/Users/rishitkar/Desktop/BanglaYield-CP/plots/feature_importance.png)
- [plots/residuals.png](/Users/rishitkar/Desktop/BanglaYield-CP/plots/residuals.png)
- [plots/gen_gap.png](/Users/rishitkar/Desktop/BanglaYield-CP/plots/gen_gap.png)

Collectively, these plots document:

- comparative model performance across split strategies,
- error heatmaps,
- predicted-vs-actual behavior,
- feature-importance ranking for the best tree-based model,
- residual distributions,
- validation-to-test generalization gaps.

## Reproducibility

### Environment

The repository includes:

- [requirements.txt](/Users/rishitkar/Desktop/BanglaYield-CP/requirements.txt)

The listed stack includes:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `catboost`

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: the committed `requirements.txt` is encoded as UTF-16 LE. If your environment expects UTF-8 requirement files, convert it before installation.

## Suggested Execution Order

For a clean rerun of the workflow, use the notebooks in the following order:

1. [data_handling.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/data_handling.ipynb)
2. [models/catboost_apply_init.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/models/catboost_apply_init.ipynb)
3. [models/regression_crop_pred.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/models/regression_crop_pred.ipynb)
4. [models/crop_ap_ratio_prediction.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/models/crop_ap_ratio_prediction.ipynb)

If only one notebook is to be used for the main experiment, prefer:

- [models/crop_ap_ratio_prediction.ipynb](/Users/rishitkar/Desktop/BanglaYield-CP/models/crop_ap_ratio_prediction.ipynb)

## License

The repository includes a [LICENSE](/Users/rishitkar/Desktop/BanglaYield-CP/LICENSE) file. Review that file for the governing usage terms.

## Citation Guidance

If you use this repository in a report, thesis, or academic submission, cite:

- the repository itself,
- the underlying SPAS-Dataset-BD data source: [Mendeley Data link](https://data.mendeley.com/datasets/cphdw4z5kw/2),
- the individual model libraries used for experimentation, where appropriate.
