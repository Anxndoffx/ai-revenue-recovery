"""
Step 2 of the pipeline: Recoverability Scoring.

Trains a model to predict the probability that a failed transaction
will be recovered, using features available BEFORE we know the outcome.

We deliberately keep `failure_reason` as a feature here (in production this
comes from bank error codes / gateway response, so it IS available at
prediction time - we're not leaking the target).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("../data/failed_transactions.csv", parse_dates=["timestamp"])

# ---- Feature engineering ----
df["month"] = df["timestamp"].dt.month
df["is_salary_window"] = ((df["day_of_month"] <= 5) | (df["day_of_month"] >= 28)).astype(int)
df["high_value_txn"] = (df["amount_inr"] > 20000).astype(int)
df["repeat_failer"] = (df["customer_past_failures"] >= 3).astype(int)

feature_cols = [
    "amount_inr", "payment_method", "bank", "failure_reason",
    "day_of_month", "customer_past_failures", "customer_tenure_days",
    "is_salary_window", "high_value_txn", "repeat_failer",
]
target_col = "recovered"

X = df[feature_cols]
y = df[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_cols = ["payment_method", "bank", "failure_reason"]
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough",
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model),
])

pipeline.fit(X_train, y_train)

# ---- Evaluation ----
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
ap = average_precision_score(y_test, y_pred_proba)

print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"ROC-AUC: {auc:.3f}")
print(f"Average Precision (PR-AUC): {ap:.3f}\n")
print(classification_report(y_test, y_pred, target_names=["Not Recovered", "Recovered"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                 Predicted No    Predicted Yes")
print(f"Actual No        {cm[0][0]:<15} {cm[0][1]}")
print(f"Actual Yes       {cm[1][0]:<15} {cm[1][1]}")

# ---- Feature importance ----
ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
all_feature_names = cat_feature_names + numeric_cols

importances = pipeline.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values("importance", ascending=False).head(15)

print("\n" + "=" * 60)
print("TOP FEATURE IMPORTANCES")
print("=" * 60)
print(importance_df.to_string(index=False))

plt.figure(figsize=(8, 6))
plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="#0d6efd")
plt.xlabel("Importance")
plt.title("Top 15 Features Driving Recovery Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
print("\nSaved feature_importance.png")

# ---- Business impact simulation ----
# Compare: what if we only chase the top X% highest-scored transactions
# vs. chasing everyone randomly (current "blind retry" baseline)?
test_df = X_test.copy()
test_df["actual_recovered"] = y_test.values
test_df["recovery_score"] = y_pred_proba
test_df["amount_inr"] = X_test["amount_inr"].values

test_df_sorted = test_df.sort_values("recovery_score", ascending=False)

total_failed_value = test_df["amount_inr"].sum()
actual_recovered_value = test_df.loc[test_df["actual_recovered"] == 1, "amount_inr"].sum()

# Simulate: if we only pursue top 40% by score, how much of the recoverable value do we capture?
top_40pct_n = int(len(test_df_sorted) * 0.4)
top_40_df = test_df_sorted.head(top_40pct_n)
captured_value_top40 = top_40_df.loc[top_40_df["actual_recovered"] == 1, "amount_inr"].sum()

print("\n" + "=" * 60)
print("BUSINESS IMPACT SIMULATION (on test set)")
print("=" * 60)
print(f"Total value of failed transactions:        ₹{total_failed_value:,.0f}")
print(f"Actually recoverable value (ground truth):  ₹{actual_recovered_value:,.0f}")
print(f"If we ONLY chase top 40% by AI score:")
print(f"  -> Recoverable value captured:            ₹{captured_value_top40:,.0f}")
print(f"  -> % of all recoverable revenue captured:  {captured_value_top40/actual_recovered_value:.1%}")
print(f"  -> While only working {0.4:.0%} of the transactions (efficiency gain)")

# Save model + test predictions for later use in the dashboard
import joblib
joblib.dump(pipeline, "recovery_model.pkl")
test_df_sorted.to_csv("test_predictions.csv", index=False)
print("\nModel saved to recovery_model.pkl")
print("Test predictions saved to test_predictions.csv")
