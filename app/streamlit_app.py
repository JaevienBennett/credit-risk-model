"""Streamlit demo — Consumer Credit Risk Scorer.

Run:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scoring import (  # noqa: E402
    score_borrower,
    LOSS_ON_DEFAULT,
    PROFIT_ON_GOOD,
)

st.set_page_config(page_title="Credit Risk Scorer", page_icon="🏦", layout="wide")


@st.cache_resource
def _warm_cache():
    # Load models once on startup (first score_borrower call triggers it)
    score_borrower({
        "RevolvingUtilizationOfUnsecuredLines": 0.3, "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0, "DebtRatio": 0.3,
        "MonthlyIncome": 6000, "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfDependents": 2,
    })
    return True


_warm_cache()

st.title("🏦 Consumer Credit Risk Scorer")
st.caption(
    "Enter a borrower profile to estimate the probability of default and get an "
    "explainable approve/decline decision. Powered by a calibrated XGBoost model "
    "(test AUC ≈ 0.87)."
)

with st.sidebar:
    st.header("Borrower profile")
    st.caption("Adjust the inputs to see the score update live.")

    age = st.slider("Age", 21, 90, 45)
    monthly_income = st.number_input("Monthly income ($)", 0, 50000, 6000, step=100)
    income_unknown = st.checkbox("Income unknown (treat as missing)")
    dependents = st.slider("Number of dependents", 0, 10, 2)

    st.subheader("Credit behavior")
    st.caption("Input ranges are limited to what the model saw in training.")
    utilization = st.slider("Credit utilization", 0.0, 1.09, 0.30, step=0.01,
                            help="Balance ÷ credit limit. 1.0+ = maxed out / over limit.")
    debt_ratio = st.slider("Debt ratio", 0.0, 5.0, 0.30, step=0.01,
                           help="Monthly debt payments ÷ monthly income.")
    open_lines = st.slider("Open credit lines & loans", 0, 25, 8)
    real_estate = st.slider("Real-estate loans / lines", 0, 4, 1)

    st.subheader("Delinquency history")
    d30 = st.slider("Times 30–59 days late", 0, 13, 0)
    d60 = st.slider("Times 60–89 days late", 0, 11, 0)
    d90 = st.slider("Times 90+ days late", 0, 17, 0)

raw = {
    "RevolvingUtilizationOfUnsecuredLines": utilization,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": d30,
    "DebtRatio": debt_ratio,
    "MonthlyIncome": None if income_unknown else monthly_income,
    "NumberOfOpenCreditLinesAndLoans": open_lines,
    "NumberOfTimes90DaysLate": d90,
    "NumberRealEstateLoansOrLines": real_estate,
    "NumberOfTime60-89DaysPastDueNotWorse": d60,
    "NumberOfDependents": dependents,
}

result = score_borrower(raw)
pd_score = result["pd"]
decision = result["decision"]
threshold = result["threshold"]
ev = result["expected_value_if_approved"]

# Out-of-distribution check: warn when the profile resembles very few training borrowers
total_delinq = d30 + d60 + d90
ood_flags = []
if total_delinq >= 15:
    ood_flags.append("very high delinquency (≥15 events)")
if utilization >= 1.0 and d90 >= 5:
    ood_flags.append("maxed-out utilization combined with serious delinquency")
if age <= 21 and total_delinq >= 10:
    ood_flags.append("very young with extensive delinquency history")
if open_lines >= 20 and total_delinq >= 10:
    ood_flags.append("many credit lines combined with many delinquencies")

if ood_flags:
    st.warning(
        "⚠️ **Out-of-distribution input.** This combination — "
        + "; ".join(ood_flags)
        + " — is extremely rare in the training data (a handful of borrowers in "
        "150,000). The model has little experience with profiles like this, so treat "
        "the score below with caution. This is a known limitation of models on rare "
        "combinations, not a fixed rule."
    )

c1, c2, c3 = st.columns(3)
c1.metric("Probability of default", f"{pd_score:.1%}")
c2.metric("Decision", decision)
c3.metric("Expected value if approved", f"${ev:,.0f}")

if decision == "APPROVE":
    st.success(f"APPROVE — PD {pd_score:.1%} is below the {threshold:.0%} cutoff.")
else:
    st.error(f"DECLINE — PD {pd_score:.1%} meets/exceeds the {threshold:.0%} cutoff.")

st.progress(min(pd_score, 1.0), text=f"Risk score: {pd_score:.1%}")
st.caption(
    f"Cutoff = {threshold:.0%} (theoretical breakeven from "
    f"+${PROFIT_ON_GOOD:,} profit per good loan vs. −${LOSS_ON_DEFAULT:,} per default)."
)

st.divider()
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Why? feature contributions (logistic scorecard)")
    contributions = result["contributions"]
    st.bar_chart(contributions.sort_values(), height=380)
    st.caption(
        "Positive = pushes toward default; negative = pushes toward approval. "
        "From the interpretable logistic scorecard, matching the regulatory need "
        "for adverse-action reasons."
    )

with right:
    st.subheader("How this works")
    st.markdown(
        f"""
- **Model:** XGBoost, isotonic-calibrated so predicted PDs match observed default rates.
- **Ranking skill:** test AUC **0.87**, KS **0.58**.
- **Decision rule:** approve while expected profit is positive
  (cutoff ≈ **{threshold:.0%}**).
- **Top drivers:** prior delinquency, credit utilization, debt ratio.

*Dollar assumptions are illustrative. Swap in real pricing/risk appetite and the
same machinery sets a production cutoff.*
"""
    )

st.divider()
st.caption("Credit Risk Model — Give Me Some Credit dataset · built with pandas, scikit-learn, XGBoost")
