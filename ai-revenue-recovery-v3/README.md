# RecoverAI

**AI-driven revenue recovery — predicting which failed payments can be recovered, and automatically deciding what to do about them.**

## The problem

Every business collecting online payments loses a chunk of revenue to failed transactions — expired cards, bank timeouts, insufficient balance, OTP failures. Industry-wide, 20–30% of payments fail on first attempt. Most of that money isn't gone; it's *recoverable* if you retry the right way at the right time. Most businesses just retry blindly, on a fixed schedule, regardless of *why* the payment failed — and recover only a fraction of what's actually recoverable.

This project builds an AI system that fixes that: it predicts recoverability, prioritizes effort, and recommends the right action per transaction — instead of chasing every failure the same way.

## Result

> **Targeting the top 40% of failed transactions by AI recovery score captures 53% of all recoverable revenue** — vs. a flat, linear return if transactions were chased randomly.

On the validation set:
- **Recovery rate among AI-recommended "chase" transactions:** 55%
- **Recovery rate among AI-recommended "suppress" transactions:** 12.8%
- The system correctly filters out low-probability cases, missing only ~0.5% of true recoverable value while saving effort on the rest.

<p align="center"><i>[screenshot of dashboard.html here]</i></p>

## How it works

The pipeline mirrors a real recovery-ops workflow:

```
Failed Transaction
      ↓
[1] Classify   → why did it fail? (insufficient funds, expired card, bank timeout...)
      ↓
[2] Score      → what's the probability this is recoverable?
      ↓
[3] Recommend  → what should happen next? (retry now / retry later / ask customer / suppress)
      ↓
[4] Validate   → check recommendation quality against actual outcomes
```

**1. Failure classification** — each failed transaction carries a root cause, derived from transaction metadata (amount, payment method, bank, timing, customer history). Different causes need completely different responses; retrying an expired card is pointless, but retrying a bank timeout in an hour often works.

**2. Recoverability scoring** — an XGBoost classifier trained on transaction features (failure reason, amount, day-of-month, customer failure history, tenure) predicts a probability of recovery. ROC-AUC: 0.75.

**3. Action recommendation** — a decision layer combines the recovery score with a simple expected-value calculation (`score × amount − cost of chasing`) to output one of five actions:
| Action | When |
|---|---|
| `RETRY_NOW` | Transient failure (network/bank timeout) |
| `RETRY_SCHEDULED` | Needs a delay (e.g. insufficient funds → retry near salary date) |
| `REQUEST_UPDATE` | Needs customer action (expired card, wrong CVV) |
| `NUDGE_CUSTOMER` | Send a reminder |
| `SUPPRESS` | Expected value too low to justify chasing |

**4. Validation** — recommendations are checked against held-out actual outcomes to confirm the strategy actually works, not just that the model has good accuracy in isolation.

## What drives recovery

Feature importance from the model confirms real-world intuition:
- `failure_reason` dominates — expired cards and fraud blocks are the strongest negative signals
- Timing matters — insufficient-funds failures recover far better in salary-credit windows (month start/end)
- Repeat failures and high transaction value both reduce recovery probability

## Tech stack

- **Data**: synthetic dataset (8,000 transactions) generated with realistic failure/recovery patterns, since real payment data isn't public
- **Modeling**: Python, pandas, XGBoost, scikit-learn
- **Decision layer**: rule-informed expected-value logic on top of model scores
- **Dashboard**: HTML/CSS/JS, hand-built SVG charts (no chart library dependency)

## Project structure

```
ai-revenue-recovery/
├── README.md
├── data/
│   └── generate_data.py         # synthetic failed-transaction dataset
├── models/
│   ├── train_recovery_model.py  # recoverability scoring model
│   └── action_recommender.py    # score → recommended action
└── dashboard/
    └── recovery_dashboard.html  # visual demo
```

## Running it

```bash
pip install pandas numpy scikit-learn xgboost matplotlib joblib

python data/generate_data.py            # generates failed_transactions.csv
python models/train_recovery_model.py   # trains model, prints metrics + business impact
python models/action_recommender.py     # generates action recommendations + validation
```

Then open `dashboard/recovery_dashboard.html` in a browser.

## Why this problem

This isn't a generic ML demo — it's built around a real, measurable business metric (revenue recovered, not just model accuracy) that maps directly to how a payments company evaluates its own product. The goal wasn't to build the most sophisticated model, but the most *useful* decision system: one that tells you not just "will this recover" but "what should we actually do about it, and is it worth doing."

## Limitations & next steps

- Trained on synthetic data — real transaction data would have messier, less separable patterns
- The expected-value cost constants (`COST_PER_RETRY`, `COST_PER_NUDGE`) are illustrative placeholders, not calibrated to real infra/messaging costs
- Next: personalized recovery messaging (LLM-generated nudges), and a live retry-outcome feedback loop to continuously improve the recoverability model
