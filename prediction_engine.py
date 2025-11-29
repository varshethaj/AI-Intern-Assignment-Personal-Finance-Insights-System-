# ===============================================================
# PREDICTION PIPELINE USING XGBOOST – STEPS 1–5 (FIXED)
# ===============================================================
import pandas as pd  # type: ignore
import numpy as np   # type: ignore

from xgboost import XGBRegressor, XGBClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score  # type: ignore

# ---------------------------------------------------------------
# STEP 1 — BUILD PREDICTION DATASET FROM USER-MONTH SIGNALS
#        (lags + next-month targets)
# ---------------------------------------------------------------
df = pd.read_csv("data/user_month_signals.csv")

if "year_month" not in df.columns:
    raise ValueError("Missing 'year_month' column in dataset.")

# Ensure year_month is sortable as a time axis
df["year_month"] = pd.to_datetime(df["year_month"])

# Sort for proper lag operations
df = df.sort_values(["user_id", "year_month"]).reset_index(drop=True)

# ---------------------------------------------------------------
# STEP 1A — CREATE LAG FEATURES (based on your real dataset)
# ---------------------------------------------------------------
# We use only lag1 to avoid wiping out all rows when users have few months
lag_features = [
    "total_spending",
    "total_income",
    "savings_rate",
    "cash_vs_digital_ratio",
    "avg_weekend_fraction",
    "stat_spike_intensity",
]

for f in lag_features:
    df[f + "_lag1"] = df.groupby("user_id")[f].shift(1)
    df[f + "_lag3"] = df.groupby("user_id")[f].shift(3)
    df[f + "_lag6"] = df.groupby("user_id")[f].shift(6)

# STEP 1B — CREATE NEXT-MONTH TARGETS
df["target_next_spending"] = df.groupby("user_id")["total_spending"].shift(-1)
df["target_next_savings_rate"] = df.groupby("user_id")["savings_rate"].shift(-1)

# Category-level predictions (none in your current schema)
category_cols: list[str] = [c for c in df.columns if c.startswith("cat_")]

for c in category_cols:
    df[f"{c}_next"] = df.groupby("user_id")[c].shift(-1)

# STEP 1C — OVERSPEND RISK LABEL (classification target)
df["overspend_risk"] = (df["target_next_spending"] > df["total_income"]).astype(int)

# Keep only rows where we have all required lag and target values
cols_needed = [f + "_lag1" for f in lag_features] + [
    "target_next_spending",
    "target_next_savings_rate",
]

df_model = df.dropna(subset=cols_needed).reset_index(drop=True)

if len(df_model) == 0:
    raise ValueError(
        "No rows left after creating lags and targets. "
        "Likely not enough months per user to support lag1 + next-month targets."
    )

# ---------------------------------------------------------------
# STEP 2 — SELECT FEATURES & TRAIN XGBOOST MODELS
# ---------------------------------------------------------------
# Use lag1 features (past behaviour) + current category profile as inputs
feature_cols = [f + "_lag1" for f in lag_features]
feature_cols += category_cols  # (empty for now, unless cat_* columns exist)

X = df_model[feature_cols]
y_spend = df_model["target_next_spending"]
y_savings = df_model["target_next_savings_rate"]
y_risk = df_model["overspend_risk"]

# Single consistent train/test split for all targets
X_train, X_test, y_spend_train, y_spend_test, y_savings_train, y_savings_test, y_risk_train, y_risk_test = (
    train_test_split(
        X,
        y_spend,
        y_savings,
        y_risk,
        test_size=0.2,
        random_state=42,
    )
)

# 2A. XGBoost model for next-month total spending
model_spend = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)
model_spend.fit(X_train, y_spend_train)

# 2B. XGBoost model for next-month savings rate
model_savings = XGBRegressor(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)
model_savings.fit(X_train, y_savings_train)

# 2C. XGBoost model for overspend risk (classification)
model_risk = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    eval_metric="logloss",
    random_state=42,
)
model_risk.fit(X_train, y_risk_train)

# 2D. Category-level XGBoost models (one per category column, if any)
category_models: dict[str, XGBRegressor] = {}
for c in category_cols:
    target_cat = df_model[f"{c}_next"]
    X_train_cat, X_test_cat, y_cat_train, y_cat_test = train_test_split(
        X, target_cat, test_size=0.2, random_state=42
    )

    model_cat = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    model_cat.fit(X_train_cat, y_cat_train)
    category_models[c] = model_cat

# ---------------------------------------------------------------
# STEP 3 — MAKE PREDICTIONS (TEST SET + FULL DATA)
# ---------------------------------------------------------------
# 3A. Predictions on test set (for evaluation)
pred_spend_test = model_spend.predict(X_test)
pred_savings_test = model_savings.predict(X_test)
pred_risk_test = model_risk.predict(X_test)

category_test_preds = {}
for c, model_cat in category_models.items():
    X_train_cat, X_test_cat, y_cat_train, y_cat_test = train_test_split(
        X, df_model[f"{c}_next"], test_size=0.2, random_state=42
    )
    category_test_preds[c] = (model_cat.predict(X_test_cat), y_cat_test)

# 3B. Predictions on full dataset (for later Insight Engine use)
X_full = df_model[feature_cols]
df_model["pred_next_spending"] = model_spend.predict(X_full)
df_model["pred_next_savings"] = model_savings.predict(X_full)
df_model["overspend_risk_prob"] = model_risk.predict_proba(X_full)[:, 1]

for c, model_cat in category_models.items():
    df_model[f"pred_{c}"] = model_cat.predict(X_full)

# For future integration, keep only the latest row per user in memory
latest_pred = (
    df_model.sort_values("year_month")
            .groupby("user_id")
            .tail(1)
            .reset_index(drop=True)
)

# ---------------------------------------------------------------
# STEP 3C — SAVE LATEST PREDICTIONS PER USER (for Insight Engine)
# ---------------------------------------------------------------
PRED_OUT = "data/user_latest_predictions.csv"
latest_pred.to_csv(PRED_OUT, index=False)
print(f"\nSaved latest per-user predictions to: {PRED_OUT}")

# ---------------------------------------------------------------
# STEP 4 — EVALUATE MODEL ACCURACY
# ---------------------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("\n=== NEXT MONTH TOTAL SPENDING PREDICTION ===")
print("RMSE:", rmse(y_spend_test, pred_spend_test))
print("R²:", r2_score(y_spend_test, pred_spend_test))

print("\n=== NEXT MONTH SAVINGS RATE PREDICTION ===")
print("RMSE:", rmse(y_savings_test, pred_savings_test))
print("R²:", r2_score(y_savings_test, pred_savings_test))

print("\n=== OVESPENDING RISK PREDICTION ===")
print("Accuracy:", accuracy_score(y_risk_test, pred_risk_test))

for c, (pred_cat_test, y_cat_test) in category_test_preds.items():
    print(f"\n=== NEXT MONTH CATEGORY SPEND: {c} ===")
    print("RMSE:", rmse(y_cat_test, pred_cat_test))
    print("R²:", r2_score(y_cat_test, pred_cat_test))
