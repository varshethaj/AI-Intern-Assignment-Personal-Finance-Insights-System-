import pandas as pd # type: ignore
import numpy as np # type: ignore
from functools import reduce

CLEAN_PATH = "data/cleaned_synthetic_dataset.csv"
DAILY_PATH = "data/daily_features.csv"
OUT_MONTHLY = "data/monthly_features.csv"
OUT_CAT_SHARE = "data/monthly_category_share.csv"
OUT_GLOBAL_TOP3 = "data/global_top3_merchants.csv"

tx = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
# ensure columns
for c in ["user_id","date","amount","merchant","category","payment_mode"]:
    if c not in tx.columns:
        tx[c] = np.nan

tx["year_month"] = tx["date"].dt.to_period("M").astype(str)
tx["abs_amount"] = tx["amount"].abs()
tx["is_cash"] = tx.get("payment_mode", "").astype(str).str.strip().str.lower() == "cash"

# total_spending_by_category (per user-month)
cat_mon = (
    tx[tx["amount"] < 0]
    .groupby(["user_id","year_month","category"], as_index=False)["amount"]
    .agg(total_spend=lambda s: s.abs().sum())
)

# average_transaction_amount (monthly, per user-month)
avg_tx = (
    tx.groupby(["user_id","year_month"], as_index=False)
      .agg(monthly_avg_transaction_amount = ("amount", lambda s: s.abs().mean()))
)

# recurring_expense_total: merchant appears in >=3 distinct months for that user
merchant_months = tx.groupby(["user_id","merchant"])["year_month"].nunique().reset_index(name="months_count")
recurring = merchant_months[merchant_months["months_count"] >= 3][["user_id","merchant"]]
recurring_set = set(map(tuple, recurring.values))
tx["is_recurring_merchant"] = list(zip(tx["user_id"], tx["merchant"]))
tx["is_recurring_merchant"] = tx["is_recurring_merchant"].apply(lambda t: t in recurring_set)

recurring_mon = (
    tx[(tx["amount"] < 0) & (tx["is_recurring_merchant"])]
    .groupby(["user_id","year_month"], as_index=False)["amount"]
    .agg(recurring_expense_total = lambda s: s.abs().sum())
)

# cash_vs_digital_ratio (per user-month)
pm = (
    tx[tx["amount"] < 0]
    .groupby(["user_id","year_month","is_cash"], as_index=False)["amount"]
    .agg(spend=lambda s: s.abs().sum())
)
pm_pivot = pm.pivot_table(index=["user_id","year_month"], columns="is_cash", values="spend", fill_value=0)
pm_pivot = pm_pivot.rename(columns={False:"digital_spend", True:"cash_spend"}).reset_index()
pm_pivot["cash_vs_digital_ratio"] = pm_pivot["cash_spend"] / (pm_pivot["cash_spend"] + pm_pivot["digital_spend"]).replace(0, np.nan)

# top_merchants_by_spend per user-month (top-5)
topk = (
    tx[tx["amount"] < 0]
    .groupby(["user_id","year_month","merchant"], as_index=False)["amount"]
    .agg(merchant_spend=lambda s: s.abs().sum())
)
topk_sorted = topk.sort_values(["user_id","year_month","merchant_spend"], ascending=[True,True,False])
top5 = topk_sorted.groupby(["user_id","year_month"]).head(5)
top5_list = top5.groupby(["user_id","year_month"]).agg(
    top_merchants = ("merchant", lambda s: list(s)),
    top_merchants_spend = ("merchant_spend", lambda s: list(s))
).reset_index()

# global top-3 merchants overall
global_top3 = (
    topk.groupby("merchant", as_index=False)["merchant_spend"]
    .sum()
    .sort_values("merchant_spend", ascending=False)
    .head(3)
    .reset_index(drop=True)
)

# savings_rate per user-month
mon_spend = tx[tx["amount"] < 0].groupby(["user_id","year_month"], as_index=False)["amount"].agg(total_spending=lambda s: s.abs().sum())
mon_income = tx[tx["amount"] > 0].groupby(["user_id","year_month"], as_index=False)["amount"].agg(total_income=lambda s: s.sum())
monthly = pd.merge(mon_spend, mon_income, on=["user_id","year_month"], how="outer").fillna(0)
monthly["savings_rate"] = np.where(
    monthly["total_income"] != 0,
    (monthly["total_income"] - monthly["total_spending"]) / monthly["total_income"],
    np.nan
)

# income_volatility: rolling 3-month std and CV per user
income_ts = monthly[["user_id","year_month","total_income"]].copy().sort_values(["user_id","year_month"])
def add_vol(g):
    g = g.copy()
    g["income_3m_std"] = g["total_income"].rolling(window=3, min_periods=1).std().values
    g["income_3m_mean"] = g["total_income"].rolling(window=3, min_periods=1).mean().values
    g["income_3m_cv"] = (g["income_3m_std"] / g["income_3m_mean"]).replace([np.inf,-np.inf], np.nan)
    return g
income_vol = income_ts.groupby("user_id", group_keys=False).apply(add_vol).reset_index(drop=True)
income_vol = income_vol[["user_id","year_month","income_3m_std","income_3m_cv"]]

# category_spend_share_percentage
cat_totals = cat_mon.groupby(["user_id","year_month"], as_index=False)["total_spend"].sum().rename(columns={"total_spend":"month_total_spend"})
cat_share = cat_mon.merge(cat_totals, on=["user_id","year_month"], how="left")
cat_share["category_spend_share_pct"] = (cat_share["total_spend"] / cat_share["month_total_spend"]).fillna(0) * 100

# assemble monthly_features: merge pieces
parts = [
    monthly,
    avg_tx,
    recurring_mon,
    pm_pivot[["user_id", "year_month", "cash_vs_digital_ratio"]],
    top5_list,
    income_vol
]
monthly_features = reduce(
    lambda left, right: pd.merge(left, right, on=["user_id", "year_month"], how="left"),
    parts
)

# --- HANDLE MISSING VALUES PROPERLY ---

# numeric columns we want to be non-null
numeric_defaults = {
    "monthly_avg_transaction_amount": 0.0,
    "recurring_expense_total": 0.0,
    "cash_vs_digital_ratio": 0.0,
    "income_3m_std": 0.0,
    "income_3m_cv": 0.0,
}

for col, default in numeric_defaults.items():
    if col not in monthly_features.columns:
        monthly_features[col] = default
    else:
        monthly_features[col] = monthly_features[col].fillna(default)

# savings_rate: avoid empty cells (set NaN to 0)
if "savings_rate" in monthly_features.columns:
    monthly_features["savings_rate"] = monthly_features["savings_rate"].fillna(0.0)

# list-like columns (store [] instead of NaN in CSV)
list_cols = ["top_merchants", "top_merchants_spend"]
for col in list_cols:
    if col not in monthly_features.columns:
        monthly_features[col] = [[] for _ in range(len(monthly_features))]
    else:
        monthly_features[col] = monthly_features[col].apply(
            lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [v])
        )

monthly_features.to_csv(OUT_MONTHLY, index=False)
cat_share.to_csv(OUT_CAT_SHARE, index=False)
global_top3.to_csv(OUT_GLOBAL_TOP3, index=False)

print("Saved:", OUT_MONTHLY, OUT_CAT_SHARE, OUT_GLOBAL_TOP3)
