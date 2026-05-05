# AgroField

<p align="center">
  <a href="https://colab.research.google.com/github/Rklearns/BanglaYield-CP/blob/main/yieldML.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
  </a>
</p>

<p align="center">
  An end-to-end machine learning workflow for agricultural yield prediction using climate, pesticide, and crop production data.
</p>

## Overview

AgroField is a regression-based crop yield prediction project built around a unified country-year agricultural dataset. The notebook pipeline integrates rainfall, temperature, pesticides, and yield records, performs structured preprocessing, and benchmarks a set of classical and advanced regression models to predict agricultural yield.

The project is implemented in a single primary notebook:

- [yieldML.ipynb](yieldML.ipynb)

It covers the complete workflow from data loading to preprocessing, model comparison, and post-hoc interpretability.

## Dataset

The dataset used in this project was gathered from the resource folder of the following repository:

- [Drinkler/Yield-Prediction - `res` directory](https://github.com/Drinkler/Yield-Prediction/tree/main/res)

The integrated dataset combines the following sources:

- Rainfall: [World Bank Group | Climate Change Knowledge Portal](https://climateknowledgeportal.worldbank.org/)
- Temperature: [World Bank Group | Climate Change Knowledge Portal](https://climateknowledgeportal.worldbank.org/)
- Yield: [FAOSTAT](https://www.fao.org/faostat/en/#data)
- Pesticides: [FAOSTAT](https://www.fao.org/faostat/en/#data)

## Data Assembly

The notebook loads four separate datasets directly from the upstream resource folder:

- `rainfall.csv`
- `temp.csv`
- `yield.csv`
- `pesticides.csv`

These sources are consolidated into a single modeling table through the following steps:

1. rainfall records are grouped by `Year`, `Country`, and `ISO3`, then aggregated with summation;
2. temperature records are grouped by `Year`, `Country`, and `ISO3`, then aggregated with mean values;
3. yield data is cleaned by removing `Domain` and `Element`, then renaming the target column to `Yield (hg/ha)`;
4. pesticides data is cleaned by removing `Domain`, `Element`, `Unit`, and `Item`, then renaming the usage column to `Pesticides (tonnes)`;
5. all four sources are merged on shared country-year keys;
6. `ISO3` is removed after merging, and a final compact table is formed with:
   - `Year`
   - `Country`
   - `Item`
   - `Rainfall (mm)`
   - `Temperature (Celsius)`
   - `Pesticides (tonnes)`
   - `Yield (hg/ha)`

This gives the project a unified tabular dataset where each row represents a crop-country-year observation.

## Preprocessing Pipeline

The preprocessing in [yieldML.ipynb](yieldML.ipynb) is done in clearly separated stages.

### 1. One-Hot Encoding

The categorical variables `Country` and `Item` are expanded using one-hot encoding. After encoding, the target column `Yield (hg/ha)` is restored as the final column in the processed frame.

### 2. Numeric Coercion and Cleaning

All model features are explicitly converted to numeric format. The notebook then:

- replaces infinite values with `NaN`,
- drops rows containing invalid or missing values,
- ensures the final feature matrix is strictly floating-point.

This step is important because the feature-selection stage uses statistical modeling that is sensitive to non-numeric types.

### 3. Backward Elimination

The notebook applies backward elimination using ordinary least squares with `statsmodels`. Features with p-values above `0.05` are iteratively removed until only statistically retained predictors remain.

This reduces the feature set before model training and acts as a lightweight feature-selection step.

### 4. Outlier Removal

Outliers are filtered using z-scores computed over the feature matrix. Rows are retained only when all absolute z-scores are below `11`.

This is a relatively permissive threshold, intended to remove only extreme observations without aggressively shrinking the dataset.

### 5. Feature Scaling

The remaining features are scaled using `MinMaxScaler`, while the target `Yield (hg/ha)` is kept in its original scale.

### 6. Train-Test Split

The final dataset is divided into:

- `80%` training data
- `20%` test data

using `train_test_split(..., random_state=42)`.

## Models Compared

The notebook benchmarks nine regression models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- Hist Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

Each model is trained on the same processed feature space and evaluated on the held-out test set.

## Evaluation Metrics

The project compares models using:

- `R2`
- `MAE`
- `MSE`
- `RMSE`
- `Max Error`
- `MAPE (%)`
- training time in seconds

## Benchmark Results

The latest benchmark results, taken from the generated `model_results.csv`, are shown below.

| Model | R2 | MAE | RMSE | Max Error | MAPE (%) | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Extra Trees | 0.9490 | 11,658.13 | 23,960.58 | 237,854.50 | 18.26 | 17.36 |
| Random Forest | 0.9489 | 11,975.00 | 23,987.95 | 222,566.80 | 19.46 | 13.63 |
| Decision Tree | 0.9247 | 13,248.39 | 29,107.03 | 529,707.00 | 20.23 | 0.32 |
| LightGBM | 0.8704 | 24,287.90 | 38,191.37 | 380,485.65 | 55.79 | 0.32 |
| XGBoost | 0.8687 | 25,033.43 | 38,446.84 | 360,881.72 | 59.07 | 0.89 |
| Hist Gradient Boosting | 0.8344 | 27,719.61 | 43,174.00 | 391,118.29 | 64.75 | 2.16 |
| CatBoost | 0.7980 | 31,268.10 | 47,682.99 | 429,394.33 | 74.20 | 0.64 |
| Gradient Boosting | 0.7593 | 34,731.63 | 52,053.80 | 472,664.19 | 87.57 | 2.03 |
| Linear Regression | 0.7062 | 39,883.82 | 57,502.29 | 410,768.80 | 105.98 | 0.15 |

### Best Performing Model

The strongest model in the current benchmark is **Extra Trees**, with:

- `R2 = 0.9490`
- `MAE = 11,658.13`
- `RMSE = 23,960.58`
- `MAPE = 18.26%`

Random Forest performs almost identically and remains one of the strongest alternatives in the benchmark.

## Visual Outputs

The notebook produces a 3 x 3 model-comparison figure where each subplot shows predicted versus actual values, together with summary metrics and runtime for the corresponding regressor.

Additional explainability outputs are also generated in the later section of the notebook:

- top-15 feature importance plot,
- SHAP summary plot,
- SHAP bar importance plot,
- SHAP waterfall explanation for a single prediction.

## Repository Structure

```text
AgroField/
├── yieldML.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## Running the Project

### In Google Colab

Use the badge at the top of this page to open the notebook directly in Colab:

- [Open `yieldML.ipynb` in Colab](https://colab.research.google.com/github/Rklearns/BanglaYield-CP/blob/main/yieldML.ipynb)

### Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open and run:

```bash
jupyter notebook yieldML.ipynb
```

## Outputs Generated by the Notebook

When executed, the notebook saves:

- `model_comparison.png`
- `model_results.csv`
- `model_predictions.csv`
- `feature_importance_top15.png`
- `shap_summary_top15.png`
- `shap_bar_top15.png`
- `shap_waterfall.png`

## Notes

- The benchmark table in this README reflects the latest results from your generated `model_results.csv`.
- The interpretability section of the notebook trains a Random Forest model for SHAP-based analysis, even though Extra Trees is the top scorer in the benchmark table.

## License

This repository includes a [LICENSE](LICENSE) file. Refer to it for usage and distribution terms.
