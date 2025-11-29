import pandas as pd # type: ignore

# ====================== Load Dataset ======================
df = pd.read_csv("data/synthetic_full_dataset.csv")

# ====================== Clean Column Names ======================
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
)

# ====================== Fix Data Types ======================
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["recurring"] = df.get("recurring", False).astype(bool)

# ====================== Normalize Text Fields ======================
text_cols = ["category", "merchant", "payment_mode", "type", "occupation"]
for col in text_cols:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.title()
        )

# Force proper transaction type formatting
df["type"] = df["type"].str.capitalize()

# ====================== Remove Invalid / Duplicate Rows ======================
critical_cols = ["date", "amount", "type", "category"]
df.drop_duplicates(inplace=True)
df.dropna(subset=critical_cols, inplace=True)

# Fill missing non-critical values safely (no chained assignment)
df["merchant"] = df["merchant"].replace("", "Unknown Merchant")
df["merchant"] = df["merchant"].fillna("Unknown Merchant")
df["payment_mode"] = df["payment_mode"].fillna("Unknown")

# ====================== Standardize Categories ======================
category_map = {
    "Food": "Food",
    "Restaurant": "Food",
    "Shopping": "Shopping",
    "Entertainment": "Entertainment",
    "Rent": "Rent",
    "Utilities": "Utilities",
    "Subscriptions": "Subscriptions",
    "Travel": "Travel",
    "Transport": "Transport",
    "Investment": "Investment"
}
df["category"] = df["category"].replace(category_map)

# ====================== Parse Dates ======================
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df.dropna(subset=["date"], inplace=True)

# ====================== Extract Useful Date Features ======================
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.day_name()
df["is_weekend"] = df["date"].dt.weekday >= 5
df["day_type"] = df["is_weekend"].apply(lambda x: "Weekend" if x else "Weekday")

# ====================== Validate Transaction Amounts ======================
df.loc[df["type"] == "Income", "amount"] = df.loc[df["type"] == "Income", "amount"].abs()
df.loc[df["type"] == "Expense", "amount"] = -df.loc[df["type"] == "Expense", "amount"].abs()

# ====================== Add Running Balance ======================
df["running_balance"] = df.groupby("user_id")["amount"].cumsum()

# ====================== Save Cleaned Dataset ======================
df.to_csv("data/cleaned_synthetic_dataset.csv", index=False)
print("Preprocessing complete. Saved as cleaned_synthetic_dataset.csv.")
