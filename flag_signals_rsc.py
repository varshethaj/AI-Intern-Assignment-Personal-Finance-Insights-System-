import pandas as pd # type: ignore
import numpy as np # type: ignore
import ast

PATTERN_PATH = "data/user_month_patterns.csv"
OUT_PATH = "data/user_month_signals.csv"

df = pd.read_csv(PATTERN_PATH)

# -------------------------------------------------------
# 1. Parse category_spend_dict if stored as string
# -------------------------------------------------------
if "category_spend_dict" in df.columns:
    def safe_parse_dict(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            x = x.strip()
            if x.startswith("{") and x.endswith("}"):
                try:
                    return ast.literal_eval(x)
                except:
                    return {}
        return {}

    df["category_spend_dict"] = df["category_spend_dict"].apply(safe_parse_dict)
else:
    df["category_spend_dict"] = [{} for _ in range(len(df))]

# -------------------------------------------------------
# 2. Rule-Based Signals
# -------------------------------------------------------
df["rb_low_or_negative_savings"] = df["savings_level"].isin(["negative", "low"])
df["rb_high_income_volatility"] = df.get("income_volatility_level", "unknown") == "high"

df["rb_high_merchant_dependency"] = df.get("is_high_merchant_dependency", False).astype(bool)
df["rb_high_recurring_burden"] = df.get("recurring_burden_level", "unknown") == "high"
df["rb_weekend_heavy_pattern"] = df.get("weekend_pattern", "unknown") == "weekend_heavy"

df["rb_extreme_payment_preference"] = df.get("payment_preference", "unknown").isin(
    ["cash_heavy", "digital_heavy"]
)

df["rb_has_spending_spike"] = df.get("has_spike", False).astype(bool)
df["rb_has_category_overspend"] = df.get("overspend_category_count", 0) > 0

# -------------------------------------------------------
# 3. Statistical Signals (Z-Scores)
# -------------------------------------------------------
for col, zcol in [
    ("total_spending", "stat_spend_zscore"),
    ("total_income", "stat_income_zscore"),
    ("savings_rate", "stat_savings_zscore"),
]:
    if col in df.columns:
        g = df.groupby("user_id")[col]
        mean = g.transform("mean")
        std = g.transform("std").replace(0, np.nan)
        df[zcol] = (df[col] - mean) / std
    else:
        df[zcol] = np.nan

df["stat_spend_unusually_high"] = df["stat_spend_zscore"] > 2
df["stat_spend_unusually_low"] = df["stat_spend_zscore"] < -2
df["stat_savings_unusually_low"] = df["stat_savings_zscore"] < -1

# Spike intensity
if {"spike_count", "max_spike_spend", "mean_daily_spend"}.issubset(df.columns):
    ratio = df["max_spike_spend"] / df["mean_daily_spend"].replace([0, np.inf, -np.inf], np.nan)
    df["stat_spike_intensity"] = ratio.where(df["spike_count"] > 0, 0).fillna(0)
else:
    df["stat_spike_intensity"] = 0

# -------------------------------------------------------
# 4. Comparative Signals
# -------------------------------------------------------

# Ensure sorting for shift operations
if "year_month" in df.columns:
    df = df.sort_values(["user_id", "year_month"])

# ---- A. Spending MoM ----
df["comp_prev_spending"] = df.groupby("user_id")["total_spending"].shift(1)
df["comp_spending_change_pct"] = (
    (df["total_spending"] - df["comp_prev_spending"]) /
    df["comp_prev_spending"].replace(0, np.nan)
)
df["comp_spending_increase"] = df["comp_spending_change_pct"] > 0.10
df["comp_spending_decrease"] = df["comp_spending_change_pct"] < -0.10

# ---- B. Income MoM ----
df["comp_prev_income"] = df.groupby("user_id")["total_income"].shift(1)
df["comp_income_change_pct"] = (
    (df["total_income"] - df["comp_prev_income"]) /
    df["comp_prev_income"].replace(0, np.nan)
)
df["comp_income_increase"] = df["comp_income_change_pct"] > 0.10
df["comp_income_decrease"] = df["comp_income_change_pct"] < -0.10

# ---- C. Category-level MoM ----
def count_category_shifts(curr, prev):
    if not curr or not prev:
        return 0
    shifts = 0
    for cat in curr:
        p = prev.get(cat, 0)
        c = curr.get(cat, 0)
        if p > 0 and abs(c - p) / p > 0.20:
            shifts += 1
    return shifts

df["comp_category_shift_count"] = 0

for uid in df["user_id"].unique():
    indices = df[df["user_id"] == uid].index.tolist()
    for i in range(1, len(indices)):
        curr_idx = indices[i]
        prev_idx = indices[i - 1]
        df.at[curr_idx, "comp_category_shift_count"] = count_category_shifts(
            df.at[curr_idx, "category_spend_dict"],
            df.at[prev_idx, "category_spend_dict"],
        )

df["comp_has_category_shift"] = df["comp_category_shift_count"] > 0

# ---- D. Cash vs Digital MoM ----
df["comp_prev_cash_ratio"] = df.groupby("user_id")["cash_vs_digital_ratio"].shift(1)
df["comp_cash_ratio_change"] = df["cash_vs_digital_ratio"] - df["comp_prev_cash_ratio"]

df["comp_cash_usage_increase"] = df["comp_cash_ratio_change"] > 0.10
df["comp_cash_usage_decrease"] = df["comp_cash_ratio_change"] < -0.10

# ---- E. Weekend vs Weekday Shift ----
if {"weekend_spend", "weekday_spend"}.issubset(df.columns):
    denominator = (df["weekend_spend"] + df["weekday_spend"]).replace(0, np.nan)
    df["comp_weekend_share"] = df["weekend_spend"] / denominator
else:
    df["comp_weekend_share"] = np.nan

df["comp_prev_weekend_share"] = df.groupby("user_id")["comp_weekend_share"].shift(1)
df["comp_weekend_shift"] = (df["comp_weekend_share"] - df["comp_prev_weekend_share"]).abs() > 0.10

# ---- F. 3-Month Trend Strength ----
df["comp_trend_3mo"] = (
    df.groupby("user_id")["total_spending"]
      .transform(lambda s: s.rolling(3, min_periods=3).mean())
)
df["comp_trend_strength"] = (
    (df["total_spending"] - df["comp_trend_3mo"]) /
    df["comp_trend_3mo"]
).replace([np.inf, -np.inf], np.nan)

# -------------------------------------------------------
# 5. Save
# -------------------------------------------------------
df.to_csv(OUT_PATH, index=False)
print("Signals for rule-based, statistical, and comparative approaches saved to:", OUT_PATH)
