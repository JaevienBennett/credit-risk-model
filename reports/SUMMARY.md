# Credit Risk Model — Plain-English Summary

*A non-technical walkthrough of what this project does and why it matters.*

---

## The problem

A lender makes money when borrowers repay and loses money when they don't. **Credit risk**
is the discipline of predicting, *before lending*, who is likely to default — so the lender
can approve the good borrowers and decline the risky ones.

## What we did

Using real (anonymized) data on **150,000 past borrowers**, we built a system that:
1. cleans and understands the data,
2. finds what actually predicts default,
3. trains a model to score each borrower's risk, and
4. converts that score into a money-maximizing **approve/decline** policy.

## What we found

**Past behavior is the best predictor of future behavior.** A borrower who had ever been
90+ days late on a payment defaulted at a rate of **41%**, versus **4.6%** for those who
hadn't — roughly **9x** more likely. Credit utilization (how maxed-out someone is) came
second.

## The model

We trained two models and evaluated them on borrowers the models had never seen:

| Model | AUC (ranking skill) |
|---|---|
| Logistic Regression (simple, explainable) | 0.863 |
| XGBoost (powerful) | **0.870** |

AUC of 0.5 is a coin flip; 1.0 is perfect. **0.87 means the model separates risky
borrowers from safe ones reliably.** XGBoost won by a hair, but the simpler model stayed
close — an important point, because lenders often must *explain* every decision.

## The money

A probability isn't a decision — the dollars are. We assumed:
- Approving a good borrower earns **+$1,500**.
- Approving a defaulter loses **−$6,000** ($10k loan × 60% loss-given-default).

Applying the profit-maximizing cutoff to a 30,000-loan test portfolio:

| Policy | Expected profit |
|---|---|
| Approve everyone | $29.96M |
| **Model-based policy** | **$33.81M** |

➡️ **The model adds ~$3.9 million** in expected profit — by saying no to the borrowers
whose risk of default outweighs the profit they'd bring.

## Why it matters

This is the full analytics cycle a lender actually runs: raw data → insight → model →
**money decision**, with explanations for every decline. It's not just a "prediction" —
it's a policy that makes and saves real dollars.

## A note on assumptions

The dollar figures use **illustrative** costs. Swap in a lender's real pricing and risk
appetite, and the same machinery produces their real cutoff. The method is the point.
