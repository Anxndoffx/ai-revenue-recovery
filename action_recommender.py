"""
Step 3 of the pipeline: Action Recommendation.

Takes a scored, classified failed transaction and outputs a concrete
next-best-action, plus WHEN to take it and WHY (for explainability).

Actions:
  - RETRY_NOW          : transient failure, retry immediately
  - RETRY_SCHEDULED     : retry on a specific future date (e.g. salary window)
  - REQUEST_UPDATE      : ask customer to fix payment method (expired card / cvv)
  - NUDGE_CUSTOMER      : send a personalized reminder/message
  - SUPPRESS            : not worth chasing (low value + low probability, or fraud)

Decision logic combines:
  - recovery_score (from the ML model)
  - failure_reason (drives the TYPE of action)
  - amount (drives whether it's worth chasing + urgency)
  - cost-to-chase heuristic (a simple expected-value calculation)
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta

# ---- Config: cost assumptions for expected-value calc ----
COST_PER_RETRY = 2          # ₹ cost of a retry attempt (infra/gateway cost)
COST_PER_NUDGE = 5          # ₹ cost of sending SMS/WhatsApp/email nudge
MIN_EXPECTED_VALUE = 15     # below this, don't bother chasing at all

# Best retry delay per failure reason (days). None = don't blind-retry, needs customer action.
RETRY_DELAY_MAP = {
    "network_error": 0.1,
    "bank_timeout": 0.15,
    "otp_failure": 0.05,
    "insufficient_funds": 4,
    "invalid_cvv": None,
    "expired_card": None,
    "fraud_block": None,
}

REASON_ACTION_TYPE = {
    "network_error": "RETRY_NOW",
    "bank_timeout": "RETRY_NOW",
    "otp_failure": "RETRY_NOW",
    "insufficient_funds": "RETRY_SCHEDULED",
    "invalid_cvv": "REQUEST_UPDATE",
    "expired_card": "REQUEST_UPDATE",
    "fraud_block": "SUPPRESS",
}

MESSAGE_TEMPLATES = {
    "RETRY_NOW": "We'll automatically retry your payment shortly — no action needed.",
    "RETRY_SCHEDULED": "We'll retry your payment on {date}. You can also pay now via a different method.",
    "REQUEST_UPDATE": "Your payment didn't go through. Please update your card details to complete this payment.",
    "NUDGE_CUSTOMER": "Reminder: your payment of ₹{amount} is still pending. Complete it in one tap.",
    "SUPPRESS": None,
}


def recommend_action(row):
    reason = row["failure_reason"]
    score = row["recovery_score"]
    amount = row["amount_inr"]

    base_action = REASON_ACTION_TYPE.get(reason, "SUPPRESS")

    # Expected value of chasing this transaction
    if base_action in ("RETRY_NOW", "RETRY_SCHEDULED"):
        cost = COST_PER_RETRY
    elif base_action == "REQUEST_UPDATE":
        cost = COST_PER_NUDGE
    else:
        cost = 0

    expected_value = (score * amount) - cost

    # Override to SUPPRESS if expected value too low, regardless of reason
    if expected_value < MIN_EXPECTED_VALUE and base_action != "SUPPRESS":
        final_action = "SUPPRESS"
        reasoning = f"Expected recovery value (₹{expected_value:.0f}) too low to justify chasing."
    else:
        final_action = base_action
        if final_action == "RETRY_NOW":
            reasoning = f"{reason.replace('_',' ').title()} is usually transient — high chance of success on immediate retry."
        elif final_action == "RETRY_SCHEDULED":
            delay = RETRY_DELAY_MAP[reason]
            reasoning = f"{reason.replace('_',' ').title()} recovers best after a {delay:.0f}-day delay (e.g. next salary credit)."
        elif final_action == "REQUEST_UPDATE":
            reasoning = f"{reason.replace('_',' ').title()} needs the customer to fix their payment method — retrying won't help."
        else:
            reasoning = "Low recovery probability (fraud block or exhausted expected value) — not worth pursuing."

    # Scheduled retry date
    retry_date = None
    if final_action == "RETRY_SCHEDULED":
        delay_days = RETRY_DELAY_MAP.get(reason, 3)
        retry_date = (datetime.now() + timedelta(days=delay_days)).strftime("%b %d")

    # Build customer-facing message
    if final_action == "RETRY_SCHEDULED":
        message = MESSAGE_TEMPLATES[final_action].format(date=retry_date)
    elif final_action == "NUDGE_CUSTOMER":
        message = MESSAGE_TEMPLATES[final_action].format(amount=f"{amount:,.0f}")
    else:
        message = MESSAGE_TEMPLATES[final_action]

    return pd.Series({
        "recommended_action": final_action,
        "expected_value_inr": round(expected_value, 2),
        "reasoning": reasoning,
        "retry_date": retry_date,
        "customer_message": message,
    })


if __name__ == "__main__":
    df = pd.read_csv("test_predictions.csv")

    actions_df = df.apply(recommend_action, axis=1)
    result = pd.concat([df, actions_df], axis=1)

    print("=" * 60)
    print("ACTION DISTRIBUTION")
    print("=" * 60)
    print(result["recommended_action"].value_counts())

    print("\n" + "=" * 60)
    print("EXPECTED VALUE BY ACTION")
    print("=" * 60)
    print(result.groupby("recommended_action")["expected_value_inr"].agg(["sum", "mean", "count"]).round(1))

    # ---- Validate against ACTUAL outcomes: does chasing what we recommend pay off? ----
    chased = result[result["recommended_action"] != "SUPPRESS"]
    suppressed = result[result["recommended_action"] == "SUPPRESS"]

    chased_recovery_rate = chased["actual_recovered"].mean()
    suppressed_recovery_rate = suppressed["actual_recovered"].mean()

    print("\n" + "=" * 60)
    print("STRATEGY VALIDATION (using held-out actual outcomes)")
    print("=" * 60)
    print(f"Recovery rate among CHASED transactions:     {chased_recovery_rate:.1%}  (n={len(chased)})")
    print(f"Recovery rate among SUPPRESSED transactions:  {suppressed_recovery_rate:.1%}  (n={len(suppressed)})")
    print(f"-> Suppression correctly filters out low-probability cases: "
          f"{'YES' if suppressed_recovery_rate < chased_recovery_rate else 'NO'}")

    money_chased_value = chased["amount_inr"].sum()
    money_recovered_from_chased = chased.loc[chased["actual_recovered"] == 1, "amount_inr"].sum()
    money_left_on_table_suppressed = suppressed.loc[suppressed["actual_recovered"] == 1, "amount_inr"].sum()

    print(f"\nValue actually recovered from CHASED transactions: ₹{money_recovered_from_chased:,.0f}")
    print(f"Value that WOULD have recovered from SUPPRESSED (missed):  ₹{money_left_on_table_suppressed:,.0f}")
    print(f"  (this is the model's error cost - acceptable if small relative to cost saved)")

    # Sample output for the README / demo
    print("\n" + "=" * 60)
    print("SAMPLE RECOMMENDATIONS")
    print("=" * 60)
    sample = result.sample(5, random_state=1)[[
        "failure_reason", "amount_inr", "recovery_score",
        "recommended_action", "expected_value_inr", "reasoning"
    ]]
    print(sample.to_string(index=False))

    result.to_csv("transactions_with_actions.csv", index=False)
    print("\nSaved transactions_with_actions.csv")
