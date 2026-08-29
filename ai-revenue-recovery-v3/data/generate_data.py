"""
Synthetic dataset generator for AI Revenue Recovery project.
Simulates failed payment transactions with realistic patterns,
so we can build a failure classifier + recovery predictor on top of it.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 8000  # number of failed transactions to simulate

payment_methods = ["Credit Card", "Debit Card", "UPI", "Netbanking", "Wallet"]
banks = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Yes Bank", "IDFC", "PNB"]
failure_reasons = [
    "insufficient_funds",
    "expired_card",
    "bank_timeout",
    "otp_failure",
    "fraud_block",
    "network_error",
    "invalid_cvv",
]

# Realistic base probabilities for failure reasons (not uniform - some are more common)
failure_reason_probs = [0.28, 0.12, 0.20, 0.15, 0.05, 0.12, 0.08]

# Recovery probability per failure reason (ground truth pattern we'll simulate from)
# This reflects real-world intuition: some failures are much easier to recover than others
base_recovery_prob = {
    "insufficient_funds": 0.55,   # recoverable if retried at the right time
    "expired_card": 0.10,         # needs customer action, rarely self-resolves
    "bank_timeout": 0.75,         # often just a transient issue
    "otp_failure": 0.60,          # customer usually retries successfully
    "fraud_block": 0.05,          # usually a hard block
    "network_error": 0.80,        # transient, high recovery
    "invalid_cvv": 0.35,          # needs correction, moderate recovery
}

# Best retry delay (in days) per failure reason - used to simulate "optimal timing" pattern
best_retry_delay = {
    "insufficient_funds": 4,   # near salary credit
    "expired_card": 999,       # essentially don't retry, needs new card
    "bank_timeout": 0.1,       # retry within hours
    "otp_failure": 0.05,       # retry almost immediately
    "fraud_block": 999,        # don't retry
    "network_error": 0.2,      # retry same day
    "invalid_cvv": 1,          # retry next day after customer corrects
}

def random_datetime(start, end):
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)

start_date = datetime(2025, 1, 1)
end_date = datetime(2026, 8, 1)

rows = []
for i in range(N):
    txn_time = random_datetime(start_date, end_date)
    method = random.choice(payment_methods)
    bank = random.choice(banks)
    amount = round(np.random.lognormal(mean=6.5, sigma=1.0), 2)  # skewed, mostly small-medium txns
    amount = min(amount, 100000)  # cap outliers

    reason = np.random.choice(failure_reasons, p=failure_reason_probs)

    # customer history: how many times has this customer failed before (loyalty/behavior signal)
    past_failures = np.random.poisson(1.2)
    customer_tenure_days = np.random.randint(1, 900)

    # day of month matters a lot for insufficient_funds recovery (salary cycles)
    day_of_month = txn_time.day

    # simulate recovery outcome using base prob + realistic adjustments
    prob = base_recovery_prob[reason]

    # adjust probability based on realistic factors
    if reason == "insufficient_funds":
        # higher recovery near month start (salary credited) or specific salary-like windows
        if day_of_month <= 5 or day_of_month >= 28:
            prob += 0.15
        else:
            prob -= 0.10
    if past_failures >= 3:
        prob -= 0.15  # repeat failers are harder to recover
    if amount > 20000:
        prob -= 0.10  # high value transactions slightly harder to recover
    if customer_tenure_days > 365:
        prob += 0.05  # loyal customers recover better (more likely to fix payment method)

    prob = np.clip(prob, 0.01, 0.95)

    recovered = np.random.rand() < prob
    # simulate actual retry delay used historically (not necessarily optimal - this is what merchants did)
    retry_delay_used = round(np.random.exponential(scale=2), 1)

    # optimal delay from our lookup, with a little noise
    optimal_delay = best_retry_delay[reason]
    if optimal_delay < 999:
        optimal_delay = max(0, optimal_delay + np.random.normal(0, 0.5))

    rows.append({
        "transaction_id": f"txn_{i:06d}",
        "timestamp": txn_time,
        "amount_inr": amount,
        "payment_method": method,
        "bank": bank,
        "failure_reason": reason,
        "day_of_month": day_of_month,
        "customer_past_failures": past_failures,
        "customer_tenure_days": customer_tenure_days,
        "retry_delay_days_used": retry_delay_used,
        "optimal_retry_delay_days": round(optimal_delay, 2) if optimal_delay < 999 else None,
        "recovered": recovered,
    })

df = pd.DataFrame(rows)
df.to_csv("failed_transactions.csv", index=False)

print(f"Generated {len(df)} rows")
print("\nOverall recovery rate: {:.1%}".format(df['recovered'].mean()))
print("\nRecovery rate by failure reason:")
print(df.groupby("failure_reason")["recovered"].agg(["mean", "count"]).sort_values("mean", ascending=False))
print("\nSample rows:")
print(df.head(5).to_string())
