import pandas as pd  # type: ignore
import numpy as np   # type: ignore

from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.cluster import KMeans               # type: ignore

# ================================================================
# 1. Load USER-MONTH Signals
# ================================================================
DATA_PATH = "data/user_month_signals.csv"
df = pd.read_csv(DATA_PATH)

if "year_month" not in df.columns:
    raise ValueError("Missing year_month column → pattern generation incomplete.")

# ================================================================
# 2. AGGREGATE TO USER-LEVEL VECTORS (ONE ROW PER USER)
# ================================================================
# Take all numeric columns as behavioral signals
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Drop columns that are identifiers / previous helper values
drop_cols = [
    "week_id",
    "comp_prev_spending",
    "comp_prev_income",
    "comp_prev_cash_ratio",
    "comp_prev_weekend_share",
]

numeric_cols = [c for c in numeric_cols if c not in drop_cols]

# Aggregate per user: mean, median, std, max of each numeric feature
user_features = df.groupby("user_id")[numeric_cols].agg(
    ["mean", "median", "std", "max"]
)

# Flatten MultiIndex column names: (col, func) -> "col_func"
user_features.columns = [f"{col}_{func}" for col, func in user_features.columns]

# Fill NaNs and bring user_id back as a column
user_features = user_features.fillna(0)
user_features.reset_index(inplace=True)

print("Created USER-LEVEL dataset with shape:", user_features.shape)

# ================================================================
# 3. Select features and scale
# ================================================================
exclude = ["user_id"]
feature_cols = [c for c in user_features.columns if c not in exclude]

X = user_features[feature_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================================================
# 4. Fit FINAL USER-LEVEL CLUSTER MODEL with k = 3
# ================================================================
N_CLUSTERS = 3
print(f"Fitting KMeans with k = {N_CLUSTERS} (fixed personas)")

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
user_features["persona_cluster"] = kmeans.fit_predict(X_scaled)

print("\nPersona cluster distribution (user-level):")
print(user_features["persona_cluster"].value_counts().sort_index())

# ================================================================
# 5. (Optional but useful) Inspect simple key feature profile
#    This is NOT a "check", just a quick summary.
# ================================================================
key_cols = [
    c for c in user_features.columns
    if ("total_spending" in c or
        "savings_rate" in c or
        "income_3m_cv" in c or
        "stat_spike_intensity" in c or
        "cash_vs_digital_ratio" in c)
]

if key_cols:
    print("\n=== Persona Profiles (Key Features, User-Level Means) ===")
    print(
        user_features
        .groupby("persona_cluster")[key_cols]
        .mean()
        .round(2)
    )

# ================================================================
# 6. Save personas (one row per user)
# ================================================================
PERSONA_OUT = "data/user_personas.csv"
user_features.to_csv(PERSONA_OUT, index=False)
print("\nSaved user-level personas to:", PERSONA_OUT)
