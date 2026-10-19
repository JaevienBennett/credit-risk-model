# Credit Risk Model — Predicting Consumer Loan Default

An end-to-end credit risk analytics project: from raw borrower data to a validated
probability-of-default (PD) model and the business decisions that follow from it
(approve/decline thresholds, expected loss, and pricing).

**Author:** _your name_ · **Stack:** Python, pandas, scikit-learn, XGBoost, Jupyter

---

## Why this project

Lenders make money by approving borrowers who repay and lose money on those who default.
The core task of consumer credit risk is estimating **P(default)** accurately, then turning
that estimate into decisions. This project does both — modeling *and* the economics behind
the cutoff.

## Data

**Give Me Some Credit** (Kaggle) — 150,000 anonymized borrower records, 10 features,
target `SeriousDlqin2yrs` (financial distress within 2 years).

- Default rate: **6.7%** (imbalanced — handled explicitly)
- Missing: `MonthlyIncome` ~20%, `NumberOfDependents` ~2.6%

Raw data is **not** committed (see `.gitignore`). Download instructions in `data/README.md`.

## Project structure

```
credit-risk-model/
├── data/
│   ├── raw/            # source CSV (gitignored)
│   └── processed/      # cleaned/feature sets (gitignored)
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation_business_impact.ipynb
├── src/                # reusable functions imported by notebooks
├── reports/figures/    # exported charts
├── requirements.txt
└── README.md
```

## Approach (high level)

1. **Data understanding & cleaning** — data types, missingness patterns, outliers, sentinel values.
2. **EDA** — distribution of default by feature; the variables that actually carry signal.
3. **Feature engineering** — ratios, delinquency aggregations, missing-value indicators.
4. **Modeling** — logistic regression (interpretable baseline) vs. gradient boosting (XGBoost);
   class imbalance handled with class weights; proper train/validation split.
5. **Evaluation** — AUC/ROC, KS statistic, precision-recall (rare event), calibration.
6. **Business impact** — cost-sensitive threshold selection, expected loss, and a simple
   approve/decline policy with dollar framing.

## Reproduce

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate credit-risk
jupyter lab
```

## Results

_TBD — filled in as notebooks are completed._

## Status

🚧 In progress.
