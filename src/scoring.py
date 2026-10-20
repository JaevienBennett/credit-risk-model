"""Reusable scoring logic for the credit risk model.

Kept in sync with notebook 03 (feature engineering). Importing this in the app
guarantees the same transformations used at training time are applied at serve time.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

DELINQ_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# Illustrative business assumptions (match notebook 05)
LOAN_AMOUNT = 10_000
LGD = 0.60
LOSS_ON_DEFAULT = LOAN_AMOUNT * LGD
PROFIT_ON_GOOD = 1_500
TRAIN_DEFAULT_MEDIAN_INCOME = 5400.0  # fallback median used in nb03


def build_features(raw: dict) -> pd.DataFrame:
    """Turn a raw borrower dict into the model's 17-feature frame (one row).

    Kept in sync with notebook 03: sentinel handling, imputation, and the
    winsorize bounds applied at training time.
    """
    df = pd.DataFrame([raw])

    # Winsorize at the training p1/p99 bounds (matches notebook 03). Without
    # this, an out-of-range input would score a value the model never saw.
    bounds = json.loads((MODELS_DIR / "metadata.json").read_text()).get(
        "winsorize_bounds", {}
    )
    for col, b in bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(b["lower"], b["upper"])

    # Sentinel codes -> missing + flag, then impute 0
    df["had_sentinel_delinq"] = (df[DELINQ_COLS] >= 96).any(axis=1).astype(int)
    for c in DELINQ_COLS:
        df[c] = df[c].where(df[c] < 96)
    df[DELINQ_COLS] = df[DELINQ_COLS].fillna(0)

    # Income missingness flag + median imputation.
    # Cast to numeric first: a None value makes the column object-dtype, which
    # fillna leaves as object and XGBoost refuses to score (this crashed the
    # "Income unknown" path in the app). The flag is derived from the numeric
    # column so non-numeric input is treated as missing too.
    income_numeric = pd.to_numeric(df["MonthlyIncome"], errors="coerce")
    df["income_missing"] = income_numeric.isnull().astype(int)
    df["MonthlyIncome"] = income_numeric.fillna(TRAIN_DEFAULT_MEDIAN_INCOME)
    df["NumberOfDependents"] = pd.to_numeric(
        df["NumberOfDependents"], errors="coerce"
    ).fillna(0)

    # Domain features
    df["total_past_due"] = df[DELINQ_COLS].sum(axis=1)
    df["has_any_delinq"] = (df["total_past_due"] > 0).astype(int)
    df["income_per_dependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["credit_lines_per_age"] = (
        df["NumberOfOpenCreditLinesAndLoans"] / df["age"].replace(0, np.nan)
    ).fillna(0)
    df["is_high_util"] = (df["RevolvingUtilizationOfUnsecuredLines"] >= 0.9).astype(int)

    return df


@lru_cache(maxsize=1)
def load_artifacts():
    """Load models + feature list from metadata (cached in-process).

    Uses the *calibrated* XGBoost for probabilities (raw scale_pos_weight PDs are
    inflated ~5x and unusable for decisions).
    """
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    xgb = joblib.load(MODELS_DIR / "xgboost_calibrated.joblib")
    logit = joblib.load(MODELS_DIR / "logistic.joblib")
    return xgb, logit, meta


def score_borrower(raw: dict, model="xgboost"):
    """Return PD, decision, feature contributions, and expected value.

    Returns a dict so the app stays presentation-only.
    """
    xgb, logit, meta = load_artifacts()
    features = meta["features"]
    X = build_features(raw)[features]

    if model == "xgboost":
        pd_score = float(xgb.predict_proba(X)[:, 1][0])
    else:
        pd_score = float(logit.predict_proba(X)[:, 1][0])

    # Feature contributions always come from the interpretable logistic scorecard,
    # regardless of which model produced the headline PD.
    clf = logit.named_steps["clf"]
    scaler = logit.named_steps["scaler"]
    z = scaler.transform(X)[0]
    contributions = pd.Series(clf.coef_[0] * z, index=features).sort_values(ascending=False)

    # Single source of truth: the served threshold lives in metadata.json
    # (empirical profit optimum from notebook 05), not recomputed here.
    threshold = meta.get("decision_threshold", {}).get(
        "served", PROFIT_ON_GOOD / (PROFIT_ON_GOOD + LOSS_ON_DEFAULT)
    )
    decision = "APPROVE" if pd_score < threshold else "DECLINE"

    # Expected value of the decision on this single applicant
    ev_approve = (1 - pd_score) * PROFIT_ON_GOOD - pd_score * LOSS_ON_DEFAULT
    ev_decline = 0.0

    return {
        "pd": pd_score,
        "decision": decision,
        "threshold": threshold,
        "contributions": contributions,
        "expected_value_if_approved": ev_approve,
        "expected_value_if_declined": ev_decline,
        "features": features,
        "X": X,
    }
