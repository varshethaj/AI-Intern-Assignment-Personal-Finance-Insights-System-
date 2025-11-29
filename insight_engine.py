# ===============================================================
# INSIGHT ENGINE — Consolidates + Transforms all engine outputs
# into final LLM-ready JSON according to your required format
# ===============================================================

import ast
import os
from calendar import monthrange
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd


# ===============================================================
# =============  UTILITY HELPERS ================================
# ===============================================================

def classify_level(value: float, global_mean: float, global_std: float) -> str:
    """Return 'low', 'moderate', or 'high' based on z-score-ish thresholds."""
    if pd.isna(value) or pd.isna(global_mean) or pd.isna(global_std) or global_std == 0:
        return "typical"
    z = (value - global_mean) / global_std
    if z >= 0.5:
        return "high"
    elif z <= -0.5:
        return "low"
    else:
        return "moderate"


def normalize_bool(value):
    """Convert TRUE/FALSE/Nan/empty/string to proper boolean."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    v = str(value).strip().lower()
    return v in ("true", "1", "yes")


def parse_list(val):
    """Parse strings like '[1,2]' or "['a','b']" into Python list."""
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or val.strip() == "":
        return []
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def parse_dict(val):
    """Parse '{...}' strings into Python dict."""
    if isinstance(val, dict):
        return val
    if not isinstance(val, str) or val.strip() == "":
        return {}
    try:
        return ast.literal_eval(val)
    except Exception:
        return {}


# ===============================================================
# =============  LOADERS WITH NORMALIZATION =====================
# ===============================================================

BOOL_COLUMNS = [
    "rb_low_or_negative_savings",
    "rb_high_income_volatility",
    "rb_high_merchant_dependency",
    "rb_high_recurring_burden",
    "rb_weekend_heavy_pattern",
    "rb_extreme_payment_preference",
    "rb_has_spending_spike",
    "rb_has_category_overspend",
    "stat_spend_unusually_high",
    "stat_spend_unusually_low",
    "stat_savings_unusually_low",
    "comp_has_category_shift",
    "comp_cash_usage_increase",
    "comp_cash_usage_decrease",
    "comp_weekend_shift",
    "has_spike",
]

# ensure uniqueness in case someone later adds duplicates by mistake
UNIQUE_BOOL_COLUMNS = list(dict.fromkeys(BOOL_COLUMNS))


def load_signals_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize boolean columns
    for col in UNIQUE_BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_bool)

    # Normalize lists + dicts
    if "top_merchants" in df.columns:
        df["top_merchants"] = df["top_merchants"].apply(parse_list)

    if "top_merchants_spend" in df.columns:
        df["top_merchants_spend"] = df["top_merchants_spend"].apply(parse_list)

    if "category_spend_dict" in df.columns:
        df["category_spend_dict"] = df["category_spend_dict"].apply(parse_dict)

    if "year_month" in df.columns:
        df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")

    return df


def load_personas_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_latest_predictions_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "year_month" in df.columns:
        df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")

    # Normalization for flags in predictions CSV (same issues)
    for col in UNIQUE_BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_bool)

    # Parse list/dicts
    if "top_merchants" in df.columns:
        df["top_merchants"] = df["top_merchants"].apply(parse_list)

    if "top_merchants_spend" in df.columns:
        df["top_merchants_spend"] = df["top_merchants_spend"].apply(parse_list)

    if "category_spend_dict" in df.columns:
        df["category_spend_dict"] = df["category_spend_dict"].apply(parse_dict)

    return df


def load_transaction_anomalies_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Normalize booleans
    if "final_anomaly" in df.columns:
        df["final_anomaly"] = df["final_anomaly"].apply(normalize_bool)
    if "spike_flag" in df.columns:
        df["spike_flag"] = df["spike_flag"].apply(normalize_bool)
    if "iso_anomaly" in df.columns:
        df["iso_anomaly"] = df["iso_anomaly"].apply(normalize_bool)

    return df


# ===============================================================
# =============  RULE/STATS SECTION =============================
# ===============================================================

def build_rule_stats_output_for_user_month(
    df: pd.DataFrame,
    user_id: str,
    year_month: Optional[str]
):
    user_rows = df[df["user_id"] == user_id].copy()
    if user_rows.empty:
        raise ValueError("User not found in signals CSV")

    ym = pd.to_datetime(year_month) if year_month else None

    if year_month:
        row = user_rows[user_rows["year_month"] == ym]
        if row.empty:
            raise ValueError("No data for this month")
        row = row.iloc[-1]
    else:
        # if no year_month given, use most recent row
        row = user_rows.sort_values("year_month").iloc[-1]

    # Income, Spending, Savings
    total_income = float(row["total_income"])
    total_expense = float(row["total_spending"])
    savings = total_income - total_expense
    savings_rate_pct = float(row["savings_rate"]) * 100

    # Temporal
    weekday_spend = float(row.get("mean_daily_spend", 0))
    weekend_fraction = float(row.get("avg_weekend_fraction", 0))
    weekend_spend = weekday_spend * weekend_fraction
    ratio = weekend_fraction

    # Recurring items
    recurring_items: List[Dict[str, Any]] = []
    rec_total = row.get("recurring_expense_total", np.nan)
    rec_total = float(rec_total) if pd.notna(rec_total) else 0.0

    # Convert top merchants + spend into recurring (if recurring flag TRUE)
    if "top_merchants" in row:
        for merchant, amt in zip(row["top_merchants"], row["top_merchants_spend"]):
            recurring_items.append({
                "name": merchant,
                "amount": float(amt),
                "category": merchant  # best approximation
            })

    # Flags dictionary
    flags = {col: bool(row.get(col, False)) for col in UNIQUE_BOOL_COLUMNS}

    return {
        "summary": {
            "income": {
                "total_income": total_income,
                "sources": [],
                "volatility_flag": row.get("income_volatility_level", "unknown")
            },
            "spending": {
                "total_expense": total_expense,
                "savings": savings,
                "savings_rate_pct": savings_rate_pct,
                "essential_share_pct": None,
                "discretionary_share_pct": None
            },
            "by_category": [],  # filled later
            "temporal": {
                "weekday_spend": weekday_spend,
                "weekend_spend": weekend_spend,
                "weekend_to_weekday_ratio": ratio,
            },
            "payment_modes": {},  # filled later
            "recurring": {
                "total_recurring": rec_total,
                "items": recurring_items
            }
        },
        "comparative": {
            "vs_prev_month": {
                "expense_change_pct": float(row.get("spend_change_pct", 0)),
                "income_change_pct": None,
                "savings_rate_change_pct_points": None
            }
        },
        "flags": flags
    }


# ===============================================================
# =============  CATEGORY BREAKDOWN (REBUILT FROM TRANSACTIONS) =
# ===============================================================

def compute_category_breakdown(trans_df, user_id, year_month):
    df_u = trans_df[trans_df["user_id"] == user_id].copy()
    if df_u.empty:
        return []

    ym = pd.to_datetime(year_month)
    df_u = df_u[df_u["date"].dt.to_period("M").dt.to_timestamp() == ym]

    df_exp = df_u[df_u["type"] == "expense"]
    if df_exp.empty:
        return []

    cat_totals = df_exp.groupby("category")["amount"].sum().abs()
    total = cat_totals.sum()

    results = []
    for cat, amt in cat_totals.items():
        pct = (amt / total) * 100
        results.append({
            "category": cat,
            "amount": round(float(amt), 2),
            "share_pct": round(float(pct), 2),
            "trend_vs_prev_pct": None
        })

    return results


# ===============================================================
# ============= PAYMENT MODES (FROM ALL TRANSACTIONS) ============
# ===============================================================

def compute_payment_modes(trans_df, user_id, year_month):
    df_u = trans_df[trans_df["user_id"] == user_id].copy()
    if df_u.empty:
        return {
            "UPI_pct": 0,
            "credit_card_pct": 0,
            "debit_card_pct": 0,
            "bank_transfer_pct": 0,
            "cash_pct": 0
        }

    ym = pd.to_datetime(year_month)
    df_u = df_u[df_u["date"].dt.to_period("M").dt.to_timestamp() == ym]

    df_exp = df_u[df_u["type"] == "expense"]
    if df_exp.empty:
        return {
            "UPI_pct": 0,
            "credit_card_pct": 0,
            "debit_card_pct": 0,
            "bank_transfer_pct": 0,
            "cash_pct": 0
        }

    df_exp = df_exp.copy()
    df_exp.loc[:, "abs_amt"] = df_exp["amount"].abs()

    total = df_exp["abs_amt"].sum()

    def pct(mode):
        amt = df_exp[df_exp["payment_mode"].str.lower() == mode]["abs_amt"].sum()
        return round(float(amt / total * 100), 2) if total > 0 else 0.0

    return {
        "UPI_pct": pct("upi"),
        "credit_card_pct": pct("credit card"),
        "debit_card_pct": pct("debit card"),
        "bank_transfer_pct": pct("bank transfer"),
        "cash_pct": pct("cash"),
    }


# ===============================================================
# =============  CLUSTERING SECTION =============================
# ===============================================================

def build_clustering_output(personas_df: pd.DataFrame, user_id: str) -> Dict[str, Any]:
    """
    Build clustering / persona info:
    - segment_id, segment_label
    - segment_description: auto-generated from cluster stats
    - peer_comparison: how this user compares to peers in same cluster
    """
    row_df = personas_df[personas_df["user_id"] == user_id]
    if row_df.empty:
        return {
            "segment_id": None,
            "segment_label": None,
            "segment_description": None,
            "peer_comparison": {}
        }

    row = row_df.iloc[0]
    cluster = int(row.get("persona_cluster", -1))

    # All users in this cluster
    cluster_df = personas_df[personas_df["persona_cluster"] == cluster].copy()
    cluster_size = len(cluster_df)

    # If something is weird and cluster is empty, fall back to simple description
    if cluster_size == 0:
        return {
            "segment_id": f"cluster_{cluster}",
            "segment_label": f"Persona Cluster {cluster}",
            "segment_description": "Automatically derived behavior persona.",
            "peer_comparison": {}
        }

    # ----- Choose a few key metrics from personas -----
    spend_col = "total_spending_mean"
    savings_col = "savings_rate_mean"
    vol_col = "income_3m_cv_mean"

    # Global stats across all users
    global_spend_mean = personas_df[spend_col].mean()
    global_spend_std = personas_df[spend_col].std()

    global_savings_mean = personas_df[savings_col].mean()
    global_savings_std = personas_df[savings_col].std()

    global_vol_mean = personas_df[vol_col].mean()
    global_vol_std = personas_df[vol_col].std()

    # Cluster-level averages
    cluster_spend_mean = cluster_df[spend_col].mean()
    cluster_savings_mean = cluster_df[savings_col].mean()
    cluster_vol_mean = cluster_df[vol_col].mean()

    # User-level values
    user_spend = float(row.get(spend_col, np.nan))
    user_savings = float(row.get(savings_col, np.nan))
    user_vol = float(row.get(vol_col, np.nan))

    # ----- Classify cluster's behaviour vs global -----
    spend_level = classify_level(cluster_spend_mean, global_spend_mean, global_spend_std)
    savings_level = classify_level(cluster_savings_mean, global_savings_mean, global_savings_std)
    vol_level = classify_level(cluster_vol_mean, global_vol_mean, global_vol_std)

    # Build a short natural-language-ish description
    segment_description = (
        f"Users in this persona are {spend_level} spenders with "
        f"{savings_level} savings rates and {vol_level} income volatility."
    )

    # ----- Peer comparison: percentiles within cluster -----
    def percentile_in_cluster(series: pd.Series, value: float) -> Optional[float]:
        if pd.isna(value) or series.empty:
            return None
        return float((series <= value).mean() * 100.0)

    spend_pct = percentile_in_cluster(cluster_df[spend_col], user_spend)
    savings_pct = percentile_in_cluster(cluster_df[savings_col], user_savings)
    vol_pct = percentile_in_cluster(cluster_df[vol_col], user_vol)

    peer_comparison = {
        "cluster_size": int(cluster_size),
        "cluster_avg_spending": float(cluster_spend_mean) if pd.notna(cluster_spend_mean) else None,
        "cluster_avg_savings_rate_pct": float(cluster_savings_mean * 100.0) if pd.notna(cluster_savings_mean) else None,
        "cluster_avg_income_volatility": float(cluster_vol_mean) if pd.notna(cluster_vol_mean) else None,
        "user_spending_percentile_in_cluster": spend_pct,
        "user_savings_rate_percentile_in_cluster": savings_pct,
        "user_income_volatility_percentile_in_cluster": vol_pct,
    }

    return {
        "segment_id": f"cluster_{cluster}",
        "segment_label": f"Persona Cluster {cluster}",
        "segment_description": segment_description,
        "peer_comparison": peer_comparison
    }


# ===============================================================
# =============  PREDICTION SECTION =============================
# ===============================================================

def build_predictions_output(latest_df, user_id):
    r = latest_df[latest_df["user_id"] == user_id]
    if r.empty:
        return {}
    r = r.iloc[0]

    pred_spend = float(r.get("pred_next_spending", 0))
    pred_save = float(r.get("pred_next_savings", 0))
    pred_rate = float(r.get("target_next_savings_rate", 0)) * 100

    risk = float(r.get("overspend_risk_prob", 0))
    if risk < 0.33:
        risk_level = "low"
    elif risk < 0.66:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "next_month": {
            "predicted_total_expense": pred_spend,
            "predicted_savings": pred_save,
            "predicted_savings_rate_pct": pred_rate,
            "risk_of_overspend_level": risk_level
        }
    }


# ===============================================================
# =============  ANOMALY SECTION ================================
# ===============================================================

def build_anomaly_output(anom_df, user_id, year_month):
    df_u = anom_df[anom_df["user_id"] == user_id].copy()
    if df_u.empty:
        return {"anomaly_count": 0, "total_anomalous_amount": 0, "top_anomalies": []}

    ym = pd.to_datetime(year_month)
    df_u["ym"] = df_u["date"].dt.to_period("M").dt.to_timestamp()
    df_u = df_u[df_u["ym"] == ym]

    # ✅ Correct boolean filtering – no "is True" here
    df_a = df_u[df_u["final_anomaly"] == True]  # noqa: E712
    if df_a.empty:
        return {"anomaly_count": 0, "total_anomalous_amount": 0, "top_anomalies": []}

    df_a = df_a.copy()
    df_a.loc[:, "abs_amount"] = df_a["amount"].abs()

    top = df_a.sort_values("abs_amount", ascending=False).head(3)

    top_list = []
    for _, r in top.iterrows():
        reason = []
        if r.get("spike_flag", False):
            reason.append("amount spike")
        if r.get("iso_anomaly", False):
            reason.append("isolation forest anomaly")
        if not reason:
            reason.append("flagged anomalous")

        top_list.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "amount": float(r["amount"]),
            "category": r["category"],
            "merchant": r["merchant"],
            "reason": "; ".join(reason)
        })

    return {
        "anomaly_count": int(df_a.shape[0]),
        "total_anomalous_amount": float(df_a["amount"].abs().sum()),
        "top_anomalies": top_list
    }


# ===============================================================
# =============  MAIN PAYLOAD BUILDER ===========================
# ===============================================================

def build_llm_payload(
    user_profile,
    period,
    rule_stats,
    category_breakdown,
    payment_modes,
    clustering_output,
    predictions_output,
    anomaly_output
):
    """
    Assemble the base payload and attach higher-level derived views like
    financial_health and insight_candidates (used by the LLM layer).
    """
    rule_stats["summary"]["by_category"] = category_breakdown
    rule_stats["summary"]["payment_modes"] = payment_modes

    payload = {
        "user_profile": user_profile,
        "period": period,
        "summary": rule_stats["summary"],
        "comparative": rule_stats["comparative"],
        "clustering": clustering_output,
        "predictions": predictions_output,
        "anomalies": anomaly_output,
        "flags": rule_stats["flags"],
    }

    # extra derived blocks for the mini-agent
    payload["financial_health"] = compute_financial_health(
        summary=payload["summary"],
        flags=payload["flags"],
    )
    payload["insight_candidates"] = build_insight_candidates(payload)

    return payload


# ----------------- USER PROFILE + PERIOD HELPERS ----------------

def build_user_profile(anom_df, user_id, year_month):
    """
    Extract age, occupation, monthly income for the correct month.
    Uses fallbacks if the month is not present in the anomalies file.
    """
    df_u = anom_df[anom_df["user_id"] == user_id].copy()

    if df_u.empty:
        return {
            "user_id": user_id,
            "age": None,
            "occupation": None,
            "monthly_income": None
        }

    # Convert year_month input to datetime
    ym = pd.to_datetime(year_month)

    # Normalize the CSV year_month column if not already
    if "year_month" in df_u.columns:
        df_u["year_month"] = pd.to_datetime(df_u["year_month"], errors="coerce")

    # 1️⃣ Exact match for this month
    row = df_u[df_u["year_month"] == ym]

    # 2️⃣ If not found, use most recent month available
    if row.empty:
        row = df_u.sort_values("year_month").iloc[-1:]

    row = row.iloc[0]

    return {
        "user_id": user_id,
        "age": int(row["age"]) if pd.notna(row.get("age")) else None,
        "occupation": row.get("occupation", None),
        "monthly_income": float(row["monthly_income"]) if pd.notna(row.get("monthly_income")) else None
    }


def build_period_from_year_month(year_month: str) -> dict:
    """
    year_month can be '2025-10' or '2025-10-01' – both are fine.
    """
    dt = pd.to_datetime(year_month)
    year = dt.year
    month = dt.month

    start_date = f"{year:04d}-{month:02d}-01"
    last_day = monthrange(year, month)[1]
    end_date = f"{year:04d}-{month:02d}-{last_day:02d}"

    return {
        "label": f"{year:04d}-{month:02d}",
        "start_date": start_date,
        "end_date": end_date
    }


# ===============================================================
# =============  HIGH-LEVEL DERIVED METRICS =====================
# ===============================================================

def compute_financial_health(summary: Dict[str, Any], flags: Dict[str, bool]) -> Dict[str, Any]:
    """
    Build a compact 'financial_health' view that the LLM can use directly.
    Includes savings rate, expense/income ratio, recurring burden, and a
    coarse risk label.

    Also sets a flag 'unrealistic_savings_suspected' when savings look
    abnormally high (useful for synthetic / incomplete data detection).
    """
    income = summary["income"]["total_income"] or 0.0
    spending = summary["spending"]["total_expense"] or 0.0
    savings = summary["spending"]["savings"] or 0.0
    savings_rate_pct = summary["spending"]["savings_rate_pct"] or 0.0
    recurring_total = summary["recurring"]["total_recurring"] or 0.0

    expense_to_income_pct = (spending / income * 100.0) if income > 0 else None
    recurring_as_income_pct = (recurring_total / income * 100.0) if income > 0 else None
    recurring_as_expense_pct = (recurring_total / spending * 100.0) if spending > 0 else None

    # Simple risk classification based on savings rate and recurring load
    risk_level = "low"
    if savings_rate_pct < 10 or flags.get("rb_low_or_negative_savings", False):
        risk_level = "high"
    elif savings_rate_pct < 25:
        risk_level = "medium"

    # If recurring is heavy relative to income, bump risk one level
    if recurring_as_income_pct and recurring_as_income_pct > 40:
        if risk_level == "low":
            risk_level = "medium"
        elif risk_level == "medium":
            risk_level = "high"

    # ---- BONUS: detect potentially unrealistic savings patterns ----
    unrealistic = False
    if savings_rate_pct is not None and savings_rate_pct > 70:
        unrealistic = True
    if expense_to_income_pct is not None and expense_to_income_pct < 20:
        unrealistic = True

    if unrealistic:
        flags["unrealistic_savings_suspected"] = True

    return {
        "savings": savings,
        "savings_rate_pct": savings_rate_pct,
        "expense_to_income_pct": expense_to_income_pct,
        "recurring_as_income_pct": recurring_as_income_pct,
        "recurring_as_expense_pct": recurring_as_expense_pct,
        "risk_level": risk_level,
        "unrealistic_savings_suspected": unrealistic,
    }


def build_insight_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pre-compute candidate insights with types and priority scores.
    This acts as a light 'agent planning' layer for the LLM.

    Each candidate looks like:
    {
        "id": "recurring_burden",
        "type": "recurring_expenses",
        "priority": 0.9,
        "data": {...}
    }
    """
    summary = payload["summary"]
    flags = payload["flags"]
    financial_health = payload.get("financial_health", {})
    anomalies = payload.get("anomalies", {})
    clustering = payload.get("clustering", {})
    predictions = payload.get("predictions", {})

    candidates: List[Dict[str, Any]] = []

    # ---- 1. Spending pattern / top categories ----
    by_cat = summary.get("by_category", []) or []
    if by_cat:
        sorted_cats = sorted(by_cat, key=lambda x: x["amount"], reverse=True)
        top_three = sorted_cats[:3]
        top_share = top_three[0]["share_pct"] if top_three else 0.0
        priority = 0.6 + min(top_share / 200.0, 0.3)  # boost if very concentrated
        candidates.append({
            "id": "spend_top_categories",
            "type": "spending_pattern",
            "priority": round(priority, 3),
            "data": {
                "top_categories": top_three,
                "total_expense": summary["spending"]["total_expense"],
            },
        })

    # ---- 2. Recurring burden ----
    income = summary["income"]["total_income"] or 0.0
    recurring_total = summary["recurring"]["total_recurring"] or 0.0
    if income > 0 and recurring_total > 0:
        rec_as_income = recurring_total / income * 100.0
        priority = 0.7
        if rec_as_income > 30:
            priority += 0.15
        if flags.get("rb_high_recurring_burden", False):
            priority += 0.1

        candidates.append({
            "id": "recurring_burden",
            "type": "recurring_expenses",
            "priority": round(min(priority, 1.0), 3),
            "data": {
                "recurring_total": recurring_total,
                "recurring_as_income_pct": rec_as_income,
                "items": summary["recurring"]["items"],
            },
        })

    # ---- 3. Payment concentration ----
    pm = summary.get("payment_modes", {}) or {}
    if pm:
        max_mode = max(pm, key=lambda k: pm[k])
        max_pct = pm[max_mode]
        priority = 0.6
        if max_pct > 70:
            priority += 0.2
        if flags.get("rb_extreme_payment_preference", False):
            priority += 0.15
        candidates.append({
            "id": "payment_concentration",
            "type": "payment_preferences",
            "priority": round(min(priority, 1.0), 3),
            "data": {
                "payment_modes_pct": pm,
                "dominant_mode": max_mode,
                "dominant_pct": max_pct,
            },
        })

    # ---- 4. Temporal / weekend vs weekday ----
    temporal = summary.get("temporal", {}) or {}
    ratio = temporal.get("weekend_to_weekday_ratio", 0.0) or 0.0
    if ratio is not None:
        priority = 0.4
        if ratio > 1:
            priority += 0.25
        if flags.get("rb_weekend_heavy_pattern", False):
            priority += 0.15
        candidates.append({
            "id": "temporal_pattern",
            "type": "temporal_behavior",
            "priority": round(min(priority, 1.0), 3),
            "data": {
                "weekday_spend": temporal.get("weekday_spend"),
                "weekend_spend": temporal.get("weekend_spend"),
                "weekend_to_weekday_ratio": ratio,
            },
        })

    # ---- 5. Financial health overview ----
    fh_priority = 0.8
    if financial_health.get("risk_level") == "high":
        fh_priority = 0.95
    elif financial_health.get("risk_level") == "medium":
        fh_priority = 0.85

    candidates.append({
        "id": "financial_health_overview",
        "type": "financial_health",
        "priority": round(fh_priority, 3),
        "data": financial_health,
    })

    # ---- 6. Anomalies / spikes ----
    if anomalies.get("anomaly_count", 0) > 0 or flags.get("has_spike", False) or flags.get("rb_has_spending_spike", False):
        priority = 0.9
        if flags.get("stat_spend_unusually_high", False):
            priority = 0.95
        candidates.append({
            "id": "anomalies_and_spikes",
            "type": "anomaly_detection",
            "priority": round(priority, 3),
            "data": anomalies,
        })

    # ---- 7. Peer comparison / persona ----
    if clustering.get("segment_id"):
        pc = clustering.get("peer_comparison", {}) or {}
        candidates.append({
            "id": "peer_comparison",
            "type": "comparative_analysis",
            "priority": 0.6,
            "data": {
                "segment_label": clustering.get("segment_label"),
                "segment_description": clustering.get("segment_description"),
                "peer_comparison": pc,
            },
        })

    # ---- 8. Predictions / next month ----
    nm = predictions.get("next_month", {}) or {}
    if nm:
        candidates.append({
            "id": "next_month_outlook",
            "type": "prediction",
            "priority": 0.55,
            "data": nm,
        })

    # Sort candidates by priority descending
    candidates = sorted(candidates, key=lambda c: c["priority"], reverse=True)
    return candidates

# ===============================================================
# =============  HIGH-LEVEL DERIVED METRICS =====================
# ===============================================================

def compute_financial_health(summary: Dict[str, Any], flags: Dict[str, bool]) -> Dict[str, Any]:
    """
    Build a compact 'financial_health' view that the LLM can use directly.
    Includes savings rate, expense/income ratio, recurring burden, and a
    coarse risk label.
    """
    income = summary["income"]["total_income"] or 0.0
    spending = summary["spending"]["total_expense"] or 0.0
    savings = summary["spending"]["savings"] or 0.0
    savings_rate_pct = summary["spending"]["savings_rate_pct"] or 0.0
    recurring_total = summary["recurring"]["total_recurring"] or 0.0

    expense_to_income_pct = (spending / income * 100.0) if income > 0 else None
    recurring_as_income_pct = (recurring_total / income * 100.0) if income > 0 else None
    recurring_as_expense_pct = (recurring_total / spending * 100.0) if spending > 0 else None

    # Simple risk classification based on savings rate and recurring load
    risk_level = "low"
    if savings_rate_pct < 10 or flags.get("rb_low_or_negative_savings", False):
        risk_level = "high"
    elif savings_rate_pct < 25:
        risk_level = "medium"

    # If recurring is heavy relative to income, bump risk one level
    if recurring_as_income_pct and recurring_as_income_pct > 40:
        if risk_level == "low":
            risk_level = "medium"
        elif risk_level == "medium":
            risk_level = "high"

    return {
        "savings": savings,
        "savings_rate_pct": savings_rate_pct,
        "expense_to_income_pct": expense_to_income_pct,
        "recurring_as_income_pct": recurring_as_income_pct,
        "recurring_as_expense_pct": recurring_as_expense_pct,
        "risk_level": risk_level,
    }


def build_insight_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pre-compute candidate insights with types and priority scores.
    This acts as a light 'agent planning' layer for the LLM.

    Each candidate looks like:
    {
        "id": "recurring_burden",
        "type": "recurring_expenses",
        "priority": 0.9,
        "data": {...}
    }
    """
    summary = payload["summary"]
    flags = payload["flags"]
    financial_health = payload.get("financial_health", {})
    anomalies = payload.get("anomalies", {})
    clustering = payload.get("clustering", {})
    predictions = payload.get("predictions", {})

    candidates: List[Dict[str, Any]] = []

    # ---- 1. Spending pattern / top categories ----
    by_cat = summary.get("by_category", []) or []
    if by_cat:
        sorted_cats = sorted(by_cat, key=lambda x: x["amount"], reverse=True)
        top_three = sorted_cats[:3]
        top_share = top_three[0]["share_pct"] if top_three else 0.0
        priority = 0.6 + min(top_share / 200.0, 0.3)  # boost if very concentrated
        candidates.append({
            "id": "spend_top_categories",
            "type": "spending_pattern",
            "priority": round(priority, 3),
            "data": {
                "top_categories": top_three,
                "total_expense": summary["spending"]["total_expense"],
            },
        })

    # ---- 2. Recurring burden ----
    income = summary["income"]["total_income"] or 0.0
    recurring_total = summary["recurring"]["total_recurring"] or 0.0
    if income > 0 and recurring_total > 0:
        rec_as_income = recurring_total / income * 100.0
        priority = 0.7
        if rec_as_income > 30:
            priority += 0.15
        if flags.get("rb_high_recurring_burden", False):
            priority += 0.1

        candidates.append({
            "id": "recurring_burden",
            "type": "recurring_expenses",
            "priority": round(min(priority, 1.0), 3),
            "data": {
                "recurring_total": recurring_total,
                "recurring_as_income_pct": rec_as_income,
                "items": summary["recurring"]["items"],
            },
        })

    # ---- 3. Payment concentration ----
    pm = summary.get("payment_modes", {}) or {}
    if pm:
        max_mode = max(pm, key=lambda k: pm[k])
        max_pct = pm[max_mode]
        priority = 0.6
        if max_pct > 70:
            priority += 0.2
        if flags.get("rb_extreme_payment_preference", False):
            priority += 0.15
        candidates.append({
            "id": "payment_concentration",
            "type": "payment_preferences",
            "priority": round(min(priority, 1.0), 3),
            "data": {
                "payment_modes_pct": pm,
                "dominant_mode": max_mode,
                "dominant_pct": max_pct,
            },
        })

    # ---- 4. Temporal / weekend vs weekday ----
    temporal = summary.get("temporal", {}) or {}
    ratio = temporal.get("weekend_to_weekday_ratio", 0.0) or 0.0
    priority = 0.4
    if ratio > 1:
        priority += 0.25
    if flags.get("rb_weekend_heavy_pattern", False):
        priority += 0.15
    candidates.append({
        "id": "temporal_pattern",
        "type": "temporal_behavior",
        "priority": round(min(priority, 1.0), 3),
        "data": {
            "weekday_spend": temporal.get("weekday_spend"),
            "weekend_spend": temporal.get("weekend_spend"),
            "weekend_to_weekday_ratio": ratio,
        },
    })

    # ---- 5. Financial health overview ----
    fh_priority = 0.8
    if financial_health.get("risk_level") == "high":
        fh_priority = 0.95
    elif financial_health.get("risk_level") == "medium":
        fh_priority = 0.85

    candidates.append({
        "id": "financial_health_overview",
        "type": "financial_health",
        "priority": round(fh_priority, 3),
        "data": financial_health,
    })

    # ---- 6. Anomalies / spikes ----
    if anomalies.get("anomaly_count", 0) > 0 or flags.get("has_spike", False) or flags.get("rb_has_spending_spike", False):
        priority = 0.9
        if flags.get("stat_spend_unusually_high", False):
            priority = 0.95
        candidates.append({
            "id": "anomalies_and_spikes",
            "type": "anomaly_detection",
            "priority": round(priority, 3),
            "data": anomalies,
        })

    # ---- 7. Peer comparison / persona ----
    if clustering.get("segment_id"):
        pc = clustering.get("peer_comparison", {}) or {}
        candidates.append({
            "id": "peer_comparison",
            "type": "comparative_analysis",
            "priority": 0.6,
            "data": {
                "segment_label": clustering.get("segment_label"),
                "segment_description": clustering.get("segment_description"),
                "peer_comparison": pc,
            },
        })

    # ---- 8. Predictions / next month ----
    nm = predictions.get("next_month", {}) or {}
    if nm:
        candidates.append({
            "id": "next_month_outlook",
            "type": "prediction",
            "priority": 0.55,
            "data": nm,
        })

    # Sort by descending priority
    candidates = sorted(candidates, key=lambda c: c["priority"], reverse=True)
    return candidates

# ===============================================================
# =============  HIGH-LEVEL ENGINE ENTRYPOINT ===================
# ===============================================================

def build_analytics_payload(
    user_id: str,
    year_month: str,
    data_dir: str = "data"
) -> Dict[str, Any]:
    """
    Convenience wrapper: load all CSVs, compute all sections,
    and return the final analytics payload (WITHOUT LLM calls).
    """
    signals_path = os.path.join(data_dir, "user_month_signals.csv")
    personas_path = os.path.join(data_dir, "user_personas.csv")
    latest_pred_path = os.path.join(data_dir, "user_latest_predictions.csv")
    anomalies_path = os.path.join(data_dir, "transaction_anomalies.csv")

    # Load datasets
    signals = load_signals_df(signals_path)
    personas = load_personas_df(personas_path)
    latest_pred = load_latest_predictions_df(latest_pred_path)
    anomalies = load_transaction_anomalies_df(anomalies_path)

    # Build data sections
    rule_stats = build_rule_stats_output_for_user_month(signals, user_id, year_month)
    category_breakdown = compute_category_breakdown(anomalies, user_id, year_month)
    payment_modes = compute_payment_modes(anomalies, user_id, year_month)
    clustering_output = build_clustering_output(personas, user_id)
    predictions_output = build_predictions_output(latest_pred, user_id)
    anomaly_output = build_anomaly_output(anomalies, user_id, year_month)

    # User profile + period
    user_profile = build_user_profile(anomalies, user_id, year_month)
    period = build_period_from_year_month(year_month)

    # Final payload
    payload = build_llm_payload(
        user_profile,
        period,
        rule_stats,
        category_breakdown,
        payment_modes,
        clustering_output,
        predictions_output,
        anomaly_output
    )

    return payload
