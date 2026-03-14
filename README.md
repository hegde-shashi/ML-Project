# Bengaluru House Price Prediction

This project predicts Bengaluru house prices using a machine learning pipeline built in the notebook `House_price_Prediction.ipynb`. The workflow covers exploratory data analysis, data cleaning, feature engineering, linear-model experiments, and final submission file generation.

## Project Overview

The model is trained on housing listings with features such as:

- `area_type`
- `location`
- `size`
- `society`
- `total_sqft`
- `bath`
- `balcony`

The notebook also uses a helper dataset containing distance from the city centre to enrich location-based features. The final output is a submission CSV with predicted prices for the test set.

## Project Structure

```text
ML Project/
|-- House_price_Prediction.ipynb
|-- README.md
|-- Datasets/
|   |-- train_(2)_(1)_(1).csv
|   |-- test_(2)_(1)_(1).csv
|   |-- dist_from_city_centre_(1).csv
|   |-- avg_rent_(1)_(1)_(1).csv
|   `-- sample_submission_(3)_(1)_(1).csv
`-- Submisions/
    `-- submission_3.csv
```

## Dataset Summary

- Training data: `10,656` rows and `10` columns
- Test data: `2,664` rows and `9` columns
- Helper data:
  - `dist_from_city_centre_(1).csv` maps locations to `dist_from_city`
  - `avg_rent_(1)_(1)_(1).csv` is included in the project, but it is not used in the current final modeling pipeline

## Workflow in the Notebook

The notebook follows this flow:

1. Load the training, test, and helper datasets.
2. Perform EDA to inspect missing values, duplicates, distributions, and category frequencies.
3. Clean raw fields:
   - remove low-value columns such as `ID` and `availability`
   - standardize location names
   - convert `size` into numeric bedroom counts
   - convert `total_sqft` ranges into numeric averages
4. Engineer features:
   - add `dist_from_city`
   - create `area_price_tier`
   - group infrequent `location` and `society` values
5. Handle missing values and trim outliers.
6. Scale numeric features and one-hot encode categorical columns.
7. Train and compare:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Elastic Net
8. Use feature selection with RFE and polynomial features for `total_sqft`.
9. Generate final predictions and save them to `Submisions/submission.csv`.

## Modeling Notes

- The target variable `price` is log-transformed before training.
- `total_sqft` is also log-transformed to reduce skew.
- Model selection is done with `GridSearchCV` using 5-fold cross-validation.
- The notebook conclusion indicates that Ridge Regression with polynomial `total_sqft` features gave the most stable validation performance.

## Requirements

Install the Python packages used in the notebook:

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn jupyter
```

## How to Run

1. Open the project folder.
2. Start Jupyter:

```bash
jupyter notebook
```

3. Open `House_price_Prediction.ipynb`.
4. Run the notebook cells in order.
5. Find the generated prediction file in `Submisions/`.

## Output

The generated submission file contains:

- `ID`: record identifier from the test set
- `price`: predicted house price

Example:

```text
ID,price
0,66.186999
1,72.970997
2,88.212124
```

