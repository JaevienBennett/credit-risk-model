"""Tests for src/scoring.py — feature building and end-to-end borrower scoring."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scoring import (  # noqa: E402
    TRAIN_DEFAULT_MEDIAN_INCOME,
    build_features,
    score_borrower,
)

MODEL_FEATURES = json.loads((ROOT / "models" / "metadata.json").read_text())["features"]

BASE_BORROWER = {
    "RevolvingUtilizationOfUnsecuredLines": 0.30,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.30,
    "MonthlyIncome": 6000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2,
}

RESULT_KEYS = {
    "pd",
    "decision",
    "threshold",
    "contributions",
    "expected_value_if_approved",
    "expected_value_if_declined",
}


def borrower(overrides=None):
    raw = dict(BASE_BORROWER)
    raw.update(overrides or {})
    return raw


class TestBuildFeatures:
    def test_contains_all_model_features(self):
        out = build_features(borrower())
        missing = [f for f in MODEL_FEATURES if f not in out.columns]
        assert missing == [], f"missing features: {missing}"
        assert len(MODEL_FEATURES) == 17

    @pytest.mark.parametrize(
        "col,value",
        [
            ("NumberOfTime30-59DaysPastDueNotWorse", 96),
            ("NumberOfTime60-89DaysPastDueNotWorse", 98),
            ("NumberOfTimes90DaysLate", 99),
        ],
    )
    def test_sentinel_codes_set_flag(self, col, value):
        out = build_features(borrower({col: value}))
        assert out["had_sentinel_delinq"].iloc[0] == 1
        # sentinel is treated as missing, then imputed to 0
        assert out[col].iloc[0] == 0

    def test_no_sentinel_when_all_normal(self):
        out = build_features(borrower())
        assert out["had_sentinel_delinq"].iloc[0] == 0

    def test_missing_income_sets_flag_and_fills_median(self):
        out = build_features(borrower({"MonthlyIncome": None}))
        row = out.iloc[0]
        assert row["income_missing"] == 1
        assert row["MonthlyIncome"] == TRAIN_DEFAULT_MEDIAN_INCOME == 5400.0

    def test_present_income_clears_flag(self):
        out = build_features(borrower())
        assert out["income_missing"].iloc[0] == 0
        assert out["MonthlyIncome"].iloc[0] == 6000


class TestScoreBorrower:
    def test_low_risk_profile_is_approved(self):
        result = score_borrower(
            borrower(
                {
                    "RevolvingUtilizationOfUnsecuredLines": 0.05,
                    "age": 45,
                    "MonthlyIncome": 8000,
                    "NumberOfTime30-59DaysPastDueNotWorse": 0,
                    "NumberOfTime60-89DaysPastDueNotWorse": 0,
                    "NumberOfTimes90DaysLate": 0,
                }
            )
        )
        assert result["decision"] == "APPROVE"
        assert 0.0 <= result["pd"] < 0.10

    def test_high_risk_profile_is_declined(self):
        result = score_borrower(
            borrower(
                {
                    "RevolvingUtilizationOfUnsecuredLines": 0.95,
                    "NumberOfTimes90DaysLate": 3,
                    "NumberOfTime30-59DaysPastDueNotWorse": 5,
                }
            )
        )
        assert result["decision"] == "DECLINE"
        assert result["pd"] > 0.30

    def test_missing_income_scores_without_error(self):
        result = score_borrower(borrower({"MonthlyIncome": None}))
        assert 0.0 <= result["pd"] <= 1.0
        assert result["decision"] in {"APPROVE", "DECLINE"}

    def test_result_keys(self):
        result = score_borrower(borrower())
        assert RESULT_KEYS.issubset(result.keys())
        assert isinstance(result["contributions"], pd.Series)
