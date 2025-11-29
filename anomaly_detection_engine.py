# ===============================================================
# ANOMALY DETECTION ON TRANSACTIONS (Improved Version)
# ===============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/synthetic_full_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

# Extract time-based features
df["day_of_week"] = df["date"].dt.weekday
df["day_type_num"] = df["day_type"].map({"weekday": 0, "weekend": 1})

# Encode categorical values
cat_enc = LabelEncoder()
merchant_enc = LabelEncoder()

df["category_enc"] = cat_enc.fit_transform(df["category"])
df["merchant_enc"] = merchant_enc.fit_transform(df["merchant"])

# ===============================================================
# Exclude income from anomaly detection model
# ===============================================================
expense_df = df[df["type"] == "expense"].copy()

# ===============================================================
# 1. Z-SCORE for expenses only
# ===============================================================
user_stats = expense_df.groupby("user_id")["amount"].agg(["mean", "std"]).reset_index()
expense_df = expense_df.merge(user_stats, on="user_id", how="left")

expense_df["z_score"] = (expense_df["amount"] - expense_df["mean"]) / (expense_df["std"] + 1e-6)
expense_df["spike_flag"] = (expense_df["z_score"].abs() > 3).astype(int)

# ===============================================================
# 2. Isolation Forest on EXPENSE Transactions Only
# ===============================================================
features = [
    "amount",
    "category_enc",
    "merchant_enc",
    "day_of_week",
    "day_type_num"
]

iso = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42
)

expense_df["iso_score"] = iso.fit_predict(expense_df[features])
expense_df["iso_anomaly"] = expense_df["iso_score"].map({-1: 1, 1: 0})

# ===============================================================
# 3. Final anomaly decision for expenses
# ===============================================================
expense_df["final_anomaly"] = ((expense_df["spike_flag"] == 1) | (expense_df["iso_anomaly"] == 1)).astype(int)

# ===============================================================
# 4. Merge back income + expenses and force income as normal
# ===============================================================
df = df.merge(
    expense_df[["date", "amount", "final_anomaly"]],
    on=["date", "amount"],
    how="left"
)

df["final_anomaly"] = df["final_anomaly"].fillna(0)  # income becomes 0
df.loc[df["type"] == "income", "final_anomaly"] = 0

# Save results
df.to_csv("data/transaction_anomalies.csv", index=False)

print("Anomaly detection completed successfully.")
print("\nAnomaly Count:")
print(df["final_anomaly"].value_counts())
