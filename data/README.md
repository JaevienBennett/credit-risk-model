# Data

## Source
**Give Me Some Credit** (Kaggle competition, 2011).

- 150,000 rows, 10 predictive features + target
- Target: `SeriousDlqin2yrs` = 1 if the borrower experienced 90+ days past due within 2 years

## Download

The raw file is not committed. To get it:

**Option A — direct (no login):**
```bash
curl -sL -o data/raw/cs-training.csv \
  "https://raw.githubusercontent.com/DrIanGregory/Kaggle-GiveMeSomeCredit/refs/heads/master/data/GiveMeSomeCredit-training.csv"
```

**Option B — Kaggle:** download `cs-training.csv` from the competition page and place it in `data/raw/`.

## Column dictionary

| Column | Meaning |
|---|---|
| `SeriousDlqin2yrs` | Target: default/delinquency in next 2 years (1/0) |
| `RevolvingUtilizationOfUnsecuredLines` | Total balance / credit limit on unsecured lines |
| `age` | Borrower age |
| `NumberOfTime30-59DaysPastDueNotWorse` | # times 30–59 days late in last 2 years |
| `DebtRatio` | Monthly debt payments / monthly income |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | # open loans and credit lines |
| `NumberOfTimes90DaysLate` | # times 90+ days late |
| `NumberRealEstateLoansOrLines` | # mortgage and real-estate lines |
| `NumberOfTime60-89DaysPastDueNotWorse` | # times 60–89 days late |
| `NumberOfDependents` | # dependents in household |
