import pandas as pd # type: ignore
import numpy as np # type: ignore

# === PATH to your cleaned dataset (will NOT be modified) ===
INPUT_PATH = "data/cleaned_synthetic_dataset.csv"
OUTPUT_PATH = "data/daily_features.csv"

# === Load dataset ===
df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

# Safety: ensure required columns exist (creates empty ones if missing, just in case)
for c in ["user_id", "date", "amount", "merchant", "category", "payment_mode"]:
    if c not in df.columns:
        df[c] = np.nan

# === Helpers ===
# Flag weekend transactions (used for weekend_vs_weekday_ratio)
df["is_weekend"] = df["date"].dt.weekday >= 5

# === Compute DAILY features per (user_id, date) ===
daily_grp = (
    df.groupby(["user_id", "date"], as_index=False)
      .agg(
          # 1) transaction_count_by_day
          daily_transaction_count = ("amount", "count"),

          # 2) total daily spending (sum of negative amounts, as positive value)
          daily_total_spending = ("amount", lambda s: s[s < 0].abs().sum()),

          # 3) total daily income (sum of positive amounts)
          daily_total_income = ("amount", lambda s: s[s > 0].sum()),

          # 4) average_transaction_amount (daily) using |amount|
          daily_average_transaction_amount = ("amount", lambda s: s.abs().mean()),

          # 5) weekend transaction count (for ratio)
          daily_weekend_tx_count = ("is_weekend", "sum")
      )
)

# 6) weekend_vs_weekday_ratio at day level:
#    here we store weekend_fraction = weekend_tx_count / total_tx_count
daily_grp["daily_weekend_fraction"] = (
    daily_grp["daily_weekend_tx_count"] /
    daily_grp["daily_transaction_count"].replace(0, np.nan)
).fillna(0.0)

# Optional: fill NaNs for numeric columns to keep file tidy
daily_grp = daily_grp.fillna({
    "daily_total_spending": 0.0,
    "daily_total_income": 0.0,
    "daily_average_transaction_amount": 0.0,
    "daily_weekend_tx_count": 0.0,
    "daily_weekend_fraction": 0.0
})

# === Save DAILY features to a separate file ===
daily_grp.to_csv(OUTPUT_PATH, index=False)

print(f"Daily features saved to: {OUTPUT_PATH}")
print(df.shape[0])  # total rows, should be 108

# how many rows are missing user_id or date?
print(df["user_id"].isna().sum(), "rows with missing user_id")
print(df["date"].isna().sum(), "rows with missing date")

# how many unique (user_id, date) pairs?
print(df[["user_id", "date"]].dropna().drop_duplicates().shape[0])
