import pandas as pd # type: ignore
import numpy as np # type: ignore
import ast

# ===============================
# PATHS
# ===============================
CLEAN_PATH = "data/cleaned_synthetic_dataset.csv"
DAILY_PATH = "data/daily_features.csv"
MONTHLY_PATH = "data/monthly_features.csv"
CAT_SHARE_PATH = "data/monthly_category_share.csv"
OUT_USER_MONTH = "data/user_month_patterns.csv"

# ===============================
# LOAD DATA
# ===============================
tx = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
monthly = pd.read_csv(MONTHLY_PATH)
cat_share = pd.read_csv(CAT_SHARE_PATH)

monthly["year_month"] = monthly["year_month"].astype(str)
cat_share["year_month"] = cat_share["year_month"].astype(str)

# ===============================
# LIST PARSING HELPERS
# ===============================
def parse_list_column(series):
    """Safely convert stringified lists to Python lists."""
    out = []
    for v in series:
        if isinstance(v, list):
            out.append(v)
        elif pd.isna(v):
            out.append([])
        else:
            try:
                out.append(ast.literal_eval(str(v)))
            except Exception:
                out.append([])
    return out

# Normalize list columns in monthly
for col in ["top_merchants", "top_merchants_spend"]:
    if col not in monthly.columns:
        monthly[col] = [[] for _ in range(len(monthly))]
    else:
        monthly[col] = parse_list_column(monthly[col])

# ===============================
# BASE USER–MONTH TABLE
# ===============================
monthly = monthly.sort_values(["user_id", "year_month"])
base = monthly[[
    "user_id", "year_month",
    "total_spending", "total_income",
    "savings_rate", "monthly_avg_transaction_amount"
]].copy()

# ===============================
# 1. SAVINGS PATTERN
# ===============================
def savings_label(sr):
    if pd.isna(sr):
        return "unknown"
    if sr < 0:
        return "negative"
    if sr < 0.2:
        return "low"
    if sr < 0.4:
        return "medium"
    return "high"

base["savings_level"] = base["savings_rate"].apply(savings_label)

# ===============================
# 2. INCOME VOLATILITY
# ===============================
if "income_3m_cv" in monthly.columns:
    vol = monthly[["user_id", "year_month", "income_3m_cv"]].copy()
else:
    vol = base[["user_id", "year_month"]].copy()
    vol["income_3m_cv"] = np.nan

vol["income_volatility_level"] = pd.cut(
    vol["income_3m_cv"],
    bins=[-np.inf, 0.2, 0.5, np.inf],
    labels=["low", "medium", "high"]
).astype("object").fillna("unknown")

# ===============================
# 3. MERCHANT DEPENDENCY
# ===============================
def merchant_concentration(row):
    spends = row["top_merchants_spend"]
    total = row["total_spending"]
    if not isinstance(spends, list) or total <= 0:
        return 0
    return sum(spends) / total

md = monthly[["user_id","year_month","total_spending",
              "top_merchants","top_merchants_spend"]].copy()
md["merchant_concentration"] = md.apply(merchant_concentration, axis=1)
md["is_high_merchant_dependency"] = md["merchant_concentration"] > 0.6

# ===============================
# 4. RECURRING BURDEN
# ===============================
if "recurring_expense_total" in monthly.columns:
    rb = monthly[["user_id","year_month","total_spending",
                  "recurring_expense_total"]].copy()

    rb["recurring_to_spend_ratio"] = np.where(
        rb["total_spending"] > 0,
        rb["recurring_expense_total"] / rb["total_spending"],
        0
    )

    rb["recurring_burden_level"] = pd.cut(
        rb["recurring_to_spend_ratio"],
        bins=[-np.inf, 0.1, 0.3, np.inf],
        labels=["low", "medium", "high"]
    ).astype("object").fillna("unknown")

else:
    rb = base[["user_id","year_month"]].copy()
    rb["recurring_expense_total"] = 0
    rb["recurring_to_spend_ratio"] = 0
    rb["recurring_burden_level"] = "unknown"

# ===============================
# 5. PAYMENT PREFERENCE
# ===============================
if "cash_vs_digital_ratio" in monthly.columns:
    pp = monthly[["user_id", "year_month", "cash_vs_digital_ratio"]].copy()
else:
    pp = base[["user_id","year_month"]].copy()
    pp["cash_vs_digital_ratio"] = 0

def payment_pref(x):
    if pd.isna(x):
        return "unknown"
    if x > 0.6:
        return "cash_heavy"
    if x < 0.2:
        return "digital_heavy"
    return "mixed"

pp["payment_preference"] = pp["cash_vs_digital_ratio"].apply(payment_pref)

# ===============================
# 6. WEEKEND / WEEKDAY BEHAVIOR
# ===============================
daily["year_month"] = daily["date"].dt.to_period("M").astype(str)

weekend = (
    daily.groupby(["user_id","year_month"], as_index=False)
         .agg(
            avg_daily_spend=("daily_total_spending", "mean"),
            avg_weekend_fraction=("daily_weekend_fraction", "mean")
         )
)

def weekend_label(x):
    if pd.isna(x):
        return "unknown"
    if x > 0.6:
        return "weekend_heavy"
    if x < 0.4:
        return "weekday_heavy"
    return "balanced"

weekend["weekend_pattern"] = weekend["avg_weekend_fraction"].apply(weekend_label)

# ===============================
# 7. DAILY SPIKES (PER MONTH)
# ===============================
def detect_spikes(g):
    m = g["daily_total_spending"].mean()
    s = g["daily_total_spending"].std(ddof=0)
    thr = m * 2 if s == 0 else m + 2 * s
    mask = g["daily_total_spending"] > thr
    spikes = g.loc[mask, "daily_total_spending"]

    return pd.Series({
        "mean_daily_spend": m,
        "std_daily_spend": s,
        "spike_threshold": thr,
        "spike_count": mask.sum(),
        "max_spike_spend": spikes.max() if not spikes.empty else 0
    })

spike = (
    daily.groupby(["user_id","year_month"])
         .apply(lambda g: detect_spikes(g.drop(columns=["user_id","year_month"])))
         .reset_index()
)

spike["has_spike"] = spike["spike_count"] > 0

# ===============================
# 8. CATEGORY OVERSPEND DETECTION
# ===============================
cs = cat_share.sort_values(["user_id","category","year_month"]).copy()
cs["prev_share"] = cs.groupby(["user_id","category"])["category_spend_share_pct"].shift(1)
cs["share_change"] = cs["category_spend_share_pct"] - cs["prev_share"]

overs = cs[cs["share_change"] > 10]

if overs.empty:
    cat_pat = base[["user_id","year_month"]].copy()
    cat_pat["overspend_category_count"] = 0
    cat_pat["top_overspend_category"] = "None"
    cat_pat["top_overspend_share_change"] = 0
else:
    cat_pat = (
        overs.groupby(["user_id","year_month"])
             .apply(lambda g: pd.Series({
                 "overspend_category_count": len(g),
                 "top_overspend_category": g.loc[g["share_change"].idxmax(),"category"],
                 "top_overspend_share_change": g["share_change"].max()
             }))
             .reset_index()
    )

# ===============================
# 9. MONTH-OVER-MONTH SPENDING TREND
# ===============================
trend = base[["user_id","year_month","total_spending"]].copy()
trend = trend.sort_values(["user_id","year_month"])

trend["prev_total_spending"] = trend.groupby("user_id")["total_spending"].shift(1)
trend["prev_total_spending"] = trend["prev_total_spending"].fillna(0)

trend["spend_change_pct"] = np.where(
    trend["prev_total_spending"] == 0,
    0,
    (trend["total_spending"] - trend["prev_total_spending"]) /
    trend["prev_total_spending"]
)

def trend_label(x):
    if x == 0:
        return "no_prev_data"
    if x > 0.1:
        return "increase"
    if x < -0.1:
        return "decrease"
    return "stable"

trend["spending_trend"] = trend["spend_change_pct"].apply(trend_label)

# ===============================
# MERGE ALL PATTERNS
# ===============================

user_month = base.merge(vol, on=["user_id","year_month"], how="left")
user_month = user_month.merge(md, on=["user_id","year_month"], how="left")
user_month = user_month.merge(rb, on=["user_id","year_month"], how="left")
user_month = user_month.merge(pp, on=["user_id","year_month"], how="left")
user_month = user_month.merge(weekend, on=["user_id","year_month"], how="left")
user_month = user_month.merge(spike, on=["user_id","year_month"], how="left")
user_month = user_month.merge(cat_pat, on=["user_id","year_month"], how="left")
# Remove duplicate columns before merging
trend_clean = trend.drop(columns=["total_spending"])
user_month = user_month.merge(trend_clean, on=["user_id","year_month"], how="left")

# ===============================
# CLEAN FINAL OUTPUT
# ===============================
# Fill numeric NaNs
numeric_fill = {
    "income_3m_cv": 0,
    "merchant_concentration": 0,
    "recurring_expense_total": 0,
    "recurring_to_spend_ratio": 0,
    "avg_daily_spend": 0,
    "avg_weekend_fraction": 0,
    "mean_daily_spend": 0,
    "std_daily_spend": 0,
    "spike_threshold": 0,
    "spike_count": 0,
    "max_spike_spend": 0,
    "overspend_category_count": 0,
    "top_overspend_share_change": 0,
    "prev_total_spending": 0,
    "spend_change_pct": 0,
}

for col, v in numeric_fill.items():
    if col in user_month.columns:
        user_month[col] = user_month[col].fillna(v)

# Fill categorical NaNs
categorical_fill = {
    "income_volatility_level": "unknown",
    "recurring_burden_level": "unknown",
    "payment_preference": "unknown",
    "weekend_pattern": "unknown",
    "top_overspend_category": "None",
    "spending_trend": "no_prev_data",
}

for col, v in categorical_fill.items():
    if col in user_month.columns:
        user_month[col] = user_month[col].astype("object").fillna(v)

# Boolean
if "has_spike" in user_month.columns:
    user_month["has_spike"] = user_month["has_spike"].fillna(False)

# Ensure list columns are lists
for col in ["top_merchants", "top_merchants_spend"]:
    if col in user_month.columns:
        user_month[col] = user_month[col].apply(lambda x: x if isinstance(x, list) else [])

# ---- FIX DUPLICATE COLUMN ISSUE ----
# Remove total_spending from trend because base already has it
trend_rename = trend.drop(columns=["total_spending"])

# ===============================
# SAVE
# ===============================
user_month.to_csv(OUT_USER_MONTH, index=False)
print("User–month pattern table saved to:", OUT_USER_MONTH)