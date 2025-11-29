"""synthesize_financial_dataset.py – Strong Personas for Insightful Clusters"""

import json, random
from datetime import datetime, timedelta
import pandas as pd  # type: ignore
from copy import deepcopy

BASE_JSON_PATH = "data/base_data.json"
OUT_CSV_PATH = "data/synthetic_full_dataset.csv"

months_to_synthesize = 10
NUM_VARIANTS_PER_BASE_USER = 10
DAILY_EXPENSE_PROB = 0.03
SAVINGS_RATE_RANGE = (0.0, 0.25)  # still used internally as "minimum savings" seed
SEED = 42
random.seed(SEED)

# >>> NEW: realistic spend ratio + inflation ranges <<<
SPEND_SHARE_RANGE = (0.5, 0.85)      # want spending to be 50–85% of income
INFLATION_RANGE = (1.02, 1.15)       # per-month inflation factor for expenses

CATEGORY_RANGES = {
    "Food": (50, 1200),
    "Shopping": (200, 4000),
    "Entertainment": (100, 2000),
    "Travel": (300, 8000),
    "Health": (100, 2000),
    "Transport": (20, 800),
    "Education": (200, 4000),
    "Investment": (500, 8000),
    "Subscriptions": (50, 500),
    "Utilities": (200, 2000),
    "Software": (100, 3000),
    "Rent": (5000, 20000),
    "Other": (50, 1500),
}

CATEGORY_PAYMENT_PREFERENCES = {
    "Food": ["UPI", "Cash", "Credit Card"],
    "Rent": ["Bank Transfer"],
    "Shopping": ["Credit Card", "Debit Card"],
    "Entertainment": ["Credit Card", "UPI", "Cash"],
    "Travel": ["Credit Card", "Bank Transfer"],
    "Utilities": ["UPI", "Bank Transfer"],
    "Subscriptions": ["UPI", "Credit Card"],
    "Investment": ["Bank Transfer", "UPI"],
    "Transport": ["UPI", "Cash"],
    "Health": ["UPI", "Credit Card"],
    "Education": ["Bank Transfer", "UPI"],
    "Software": ["Credit Card", "Bank Transfer"],
    "Other": ["UPI", "Cash"],
}

COMMON_CATEGORIES = [
    "Food",
    "Transport",
    "Shopping",
    "Entertainment",
    "Health",
    "Travel",
    "Utilities",
    "Subscriptions",
    "Investment",
    "Other",
]

ROLE_NAMES = {
    0: "low",
    1: "moderate",
    2: "impulsive",
    3: "cash",
    4: "subscription",
}

# -------- NEW: fixed salaries for specific occupations --------
FIXED_SALARY_BY_OCCUPATION = {
    "software engineer": 85000,
    "college student": 15000,
}


def get_fixed_salary_for_profile(prof):
    """
    Return a fixed salary if occupation matches one of our
    special cases (software engineer / college student),
    otherwise None.
    """
    occ = str(prof.get("occupation", "")).lower()
    for key, val in FIXED_SALARY_BY_OCCUPATION.items():
        if key in occ:
            return val
    return None


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d")


def fmt_date(dt):
    return dt.strftime("%Y-%m-%d")


def add_months(dt, months):
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    return dt.replace(year=y, month=m, day=min(dt.day, 28))


def get_numeric_monthly_income(user):
    """
    Prefer profile['monthly_income'] when numeric.
    Only infer from transactions when it's 'variable'
    or missing / non-numeric.
    """
    prof = user.get("profile", {})
    mi = prof.get("monthly_income", None)

    if isinstance(mi, str) and mi.lower() == "variable":
        incomes = [
            t.get("amount", 0)
            for t in user.get("transactions", [])
            if t.get("type") == "income" and isinstance(t.get("amount"), (int, float))
        ]
        return int(sum(incomes)) if incomes else 30000

    if isinstance(mi, (int, float)):
        return int(mi)

    try:
        return int(float(mi))
    except Exception:
        incomes = [
            t.get("amount", 0)
            for t in user.get("transactions", [])
            if t.get("type") == "income" and isinstance(t.get("amount"), (int, float))
        ]
        if incomes:
            return int(sum(incomes))
        return 30000


def classify_user_type(user):
    """Roughly classify base user as 'freelancer', 'student', or 'salaried'."""
    prof = user.get("profile", {})
    mi = prof.get("monthly_income")
    occ = str(prof.get("occupation", "")).lower()
    numeric_income = get_numeric_monthly_income(user)

    if isinstance(mi, str):
        return "freelancer"
    if "student" in occ or numeric_income < 20000:
        return "student"
    return "salaried"


def choose_payment_mode(cat, behavior=None):
    """
    Choose payment mode with optional behavior bias.
    """
    behavior = {
        "spend_multiplier": 1.0,
        "daily_expense_prob": DAILY_EXPENSE_PROB,
        "spike_prob": 0.0,
        "spike_multiplier": 1.0,
        "category_multipliers": {},
        "subscription_bias": 0.0,
        "food_bias": 0.0,
        "transport_bias": 0.0,
        "force_cash_for_discretionary": False,
        "force_digital_for_most": False,
        "income_volatility": 0.2,
        "category_profile": None,  # will be set below
        "weekend_spend_bias": 0.0,
        "min_savings_multiplier": 1.0,  # NEW
    }
    behavior["category_profile"] = _make_random_category_profile()
    force_cash = behavior.get("force_cash_for_discretionary", False)
    force_digital = behavior.get("force_digital_for_most", False)

    discretionary_cats = {"Food", "Shopping", "Entertainment", "Transport", "Other"}

    if force_cash and cat in discretionary_cats:
        # strongly cash
        if random.random() < 0.9:
            return "Cash"

    if force_digital and cat in discretionary_cats.union({"Subscriptions"}):
        # strongly digital
        if random.random() < 0.9:
            return random.choice(["UPI", "Credit Card", "Debit Card"])

    prefs = CATEGORY_PAYMENT_PREFERENCES.get(cat)
    if prefs:
        return random.choice(prefs)

    return random.choice(["UPI", "Cash", "Credit Card", "Debit Card", "Bank Transfer"])


def sample_expense_amount(cat, behavior=None):
    behavior = behavior or {}
    spend_mult = behavior.get("spend_multiplier", 1.0)
    category_mults = behavior.get("category_multipliers", {})
    spike_prob = behavior.get("spike_prob", 0.0)
    spike_multiplier = behavior.get("spike_multiplier", 1.0)

    low, high = CATEGORY_RANGES.get(cat, CATEGORY_RANGES["Other"])
    base = random.triangular(low, high, (low + high) / 4.0)
    amt = base * spend_mult * category_mults.get(cat, 1.0)

    if spike_prob > 0 and random.random() < spike_prob:
        amt *= spike_multiplier

    return int(max(20, amt))


def build_category_pool(expense_templates, behavior=None):
    """
    Base pool from expense templates or COMMON_CATEGORIES,
    then bias towards certain categories (e.g. subscriptions, food).
    Also nudged by a per-variant category profile.
    """
    behavior = behavior or {}
    base = [
        t["category"] for t in expense_templates if not t.get("recurring")
    ] or COMMON_CATEGORIES[:]

    sub_bias = behavior.get("subscription_bias", 0.0)
    food_bias = behavior.get("food_bias", 0.0)
    transport_bias = behavior.get("transport_bias", 0.0)
    cat_profile = behavior.get("category_profile", {})

    if sub_bias > 0:
        base += ["Subscriptions"] * int(4 * sub_bias + 1)

    if food_bias > 0:
        base += ["Food"] * int(3 * food_bias + 1)

    if transport_bias > 0:
        base += ["Transport"] * int(3 * transport_bias + 1)

    for cat, w in cat_profile.items():
        if w > 1.05:
            base += [cat] * int((w - 1.0) * 4)

    if not base:
        base = COMMON_CATEGORIES[:]
    return base


def _adjust_to_weekend_or_weekday(exp_date, weekend_bias):
    """
    weekend_bias > 0 : try to move to weekend
    weekend_bias < 0 : try to move to weekday
    """
    if weekend_bias == 0:
        return exp_date

    max_tries = 5
    for _ in range(max_tries):
        if weekend_bias > 0:
            if exp_date.weekday() >= 5:
                return exp_date
        else:
            if exp_date.weekday() < 5:
                return exp_date
        # resample day in same month
        new_day = random.randint(1, 28)
        exp_date = exp_date.replace(day=new_day)
    return exp_date


def _make_random_category_profile():
    """
    Build a per-variant random category preference profile that
    boosts a few random categories so clusters spread out.
    """
    profile = {}
    k = random.randint(3, 5)
    chosen = random.sample(COMMON_CATEGORIES, k=min(k, len(COMMON_CATEGORIES)))
    for cat in COMMON_CATEGORIES:
        if cat in chosen:
            profile[cat] = random.uniform(1.2, 1.7)
        else:
            profile[cat] = random.uniform(0.6, 1.0)
    return profile


# ------------------------------------------------------
# FINAL GENERATOR WITH FREELANCER-INCOME-FIRST & VARIANTS
# ------------------------------------------------------
def synthesize_for_user(user, months_to_add=2):
    synthesized = []
    templates = deepcopy(user["transactions"])
    income_templates = [t for t in templates if t["type"] == "income"]
    expense_templates = [t for t in templates if t["type"] == "expense"]

    numeric_mi = get_numeric_monthly_income(user)
    prof = user.get("profile", {})
    user_type = prof.get("user_type") or classify_user_type(user)
    is_freelancer = user_type == "freelancer"
    behavior = user.get("behavior", {})

    # NEW: fixed salary for certain occupations
    fixed_salary = get_fixed_salary_for_profile(prof)

    latest_date = max(parse_date(t["date"]) for t in templates)
    base_income_volatility = behavior.get("income_volatility", 0.2)

    for m in range(1, months_to_add + 1):
        month_start = add_months(latest_date, m).replace(day=1)
        base_min, base_max = SAVINGS_RATE_RANGE
        min_savings_mult = behavior.get("min_savings_multiplier", 1.0)
        min_savings = random.uniform(base_min, base_max) * min_savings_mult
        min_savings = max(0.0, min(min_savings, 0.8))

        # ------------------- Build incomes for month -------------------
        incomes = []
        if income_templates:
            # If we have a fixed salary (software engineer / college student),
            # make sure total monthly income equals that fixed salary and
            # is the same every month (no jitter).
            if fixed_salary is not None and not is_freelancer:
                n = len(income_templates)
                if n == 0:
                    base_amounts = [fixed_salary]
                else:
                    per_inc = fixed_salary // n
                    remainder = fixed_salary - per_inc * n
                    base_amounts = []
                    for idx in range(n):
                        amt = per_inc + (remainder if idx == n - 1 else 0)
                        base_amounts.append(amt)

                for inc, amt in zip(income_templates, base_amounts):
                    d = parse_date(inc["date"]).day
                    inc_date = month_start.replace(day=min(d, 28))
                    incomes.append(
                        {
                            "date": fmt_date(inc_date),
                            "amount": int(amt),
                            "category": inc["category"],
                            "merchant": inc["merchant"],
                            "payment_mode": inc.get("payment_mode", "Bank Transfer"),
                            "recurring": inc.get("recurring", True),
                        }
                    )
            else:
                for inc in income_templates:
                    d = parse_date(inc["date"]).day
                    inc_date = month_start.replace(day=min(d, 28))

                    if is_freelancer:
                        vol = base_income_volatility
                        factor_low = max(0.2, 1.0 - vol * 2.5)
                        factor_high = 1.0 + vol * 2.5
                        factor = random.uniform(factor_low, factor_high)
                    else:
                        factor = random.uniform(0.95, 1.05)

                    amt = int(inc["amount"] * factor)
                    incomes.append(
                        {
                            "date": fmt_date(inc_date),
                            "amount": amt,
                            "category": inc["category"],
                            "merchant": inc["merchant"],
                            "payment_mode": inc.get("payment_mode", "Bank Transfer"),
                            "recurring": inc.get("recurring", True),
                        }
                    )
        else:
            # No income templates: fall back to fixed salary if present
            base_amount = fixed_salary if fixed_salary is not None else numeric_mi
            incomes.append(
                {
                    "date": fmt_date(month_start.replace(day=1)),
                    "amount": int(base_amount),
                    "category": "Salary",
                    "merchant": prof.get("occupation", "Employer"),
                    "payment_mode": "Bank Transfer",
                    "recurring": True,
                }
            )

        incomes.sort(key=lambda x: x["date"])
        total_income = sum(inc["amount"] for inc in incomes)
        remaining = total_income
        savings_threshold = int(total_income * min_savings)

        income_ratio = total_income / max(1, numeric_mi)
        month_spend_mult = behavior.get("spend_multiplier", 1.0) * (income_ratio ** 0.7)

        month_behavior = dict(behavior)
        month_behavior["spend_multiplier"] = month_spend_mult

        # ---------------- FREELANCER: income-first, then expenses ----------------
        if is_freelancer:
            current_income = 0

            for inc in incomes:
                current_income += inc["amount"]
                synthesized.append(
                    {
                        "user_id": user["user_id"],
                        "age": prof.get("age"),
                        "occupation": prof.get("occupation"),
                        "monthly_income": current_income,
                        "date": inc["date"],
                        "amount": inc["amount"],
                        "type": "income",
                        "category": inc["category"],
                        "merchant": inc["merchant"],
                        "payment_mode": inc["payment_mode"],
                        "recurring": inc.get("recurring", True),
                        "day_type": "weekday",
                        "persona_label": prof.get("persona_label"),
                    }
                )

            rec_exp = [t for t in expense_templates if t.get("recurring")]
            for exp in rec_exp:
                exp_day = parse_date(exp["date"]).day
                exp_date = month_start.replace(day=min(exp_day, 28))
                amt = sample_expense_amount(exp["category"], month_behavior)
                if remaining - amt < savings_threshold:
                    allowed = remaining - savings_threshold
                    if allowed < 20:
                        continue
                    amt = allowed
                remaining -= amt
                synthesized.append(
                    {
                        "user_id": user["user_id"],
                        "age": prof.get("age"),
                        "occupation": prof.get("occupation"),
                        "monthly_income": current_income,
                        "date": fmt_date(exp_date),
                        "amount": -amt,
                        "type": "expense",
                        "category": exp["category"],
                        "merchant": exp["merchant"],
                        "payment_mode": choose_payment_mode(
                            exp["category"], month_behavior
                        ),
                        "recurring": True,
                        "day_type": "weekday"
                        if exp_date.weekday() < 5
                        else "weekend",
                        "persona_label": prof.get("persona_label"),
                    }
                )

            pool = build_category_pool(expense_templates, month_behavior)
            attempts = 0
            daily_prob = month_behavior.get("daily_expense_prob", DAILY_EXPENSE_PROB)
            weekend_bias = month_behavior.get("weekend_spend_bias", 0.0)
            while remaining > savings_threshold:
                attempts += 1
                if attempts > 400 or random.random() > daily_prob:
                    if attempts > 400:
                        break
                    continue
                exp_date = month_start.replace(day=random.randint(1, 28))
                exp_date = _adjust_to_weekend_or_weekday(exp_date, weekend_bias)
                cat = random.choice(pool)
                amt = sample_expense_amount(cat, month_behavior)
                if remaining - amt < savings_threshold:
                    allowed = remaining - savings_threshold
                    if allowed < 20:
                        break
                    amt = allowed
                remaining -= amt
                day_type = "weekend" if exp_date.weekday() >= 5 else "weekday"
                synthesized.append(
                    {
                        "user_id": user["user_id"],
                        "age": prof.get("age"),
                        "occupation": prof.get("occupation"),
                        "monthly_income": current_income,
                        "date": fmt_date(exp_date),
                        "amount": -amt,
                        "type": "expense",
                        "category": cat,
                        "merchant": f"{cat} Vendor",
                        "payment_mode": choose_payment_mode(cat, month_behavior),
                        "recurring": False,
                        "day_type": day_type,
                        "persona_label": prof.get("persona_label"),
                    }
                )

            user["profile"]["monthly_income"] = current_income
            continue

        # ---------------- SALARIED / STUDENT ----------------
        current_income = 0
        remaining = 0
        daily_prob = month_behavior.get("daily_expense_prob", DAILY_EXPENSE_PROB)
        weekend_bias = month_behavior.get("weekend_spend_bias", 0.0)

        if user_type == "student":
            rec_exp_student = []
            for exp in expense_templates:
                if not exp.get("recurring"):
                    continue
                if exp["category"] in {"Rent", "Subscriptions", "Education", "Transport"}:
                    if random.random() < 0.5:
                        rec_exp_student.append(exp)
            rec_expenses = rec_exp_student
        else:
            rec_expenses = [t for t in expense_templates if t.get("recurring")]

        for idx, inc in enumerate(incomes):
            current_income += inc["amount"]
            remaining += inc["amount"]
            synthesized.append(
                {
                    "user_id": user["user_id"],
                    "age": prof.get("age"),
                    "occupation": prof.get("occupation"),
                    "monthly_income": current_income,
                    "date": inc["date"],
                    "amount": inc["amount"],
                    "type": "income",
                    "category": inc["category"],
                    "merchant": inc["merchant"],
                    "payment_mode": inc["payment_mode"],
                    "recurring": inc.get("recurring", True),
                    "day_type": "weekday",
                    "persona_label": prof.get("persona_label"),
                }
            )

            this_date = parse_date(inc["date"])
            next_date = (
                parse_date(incomes[idx + 1]["date"]) - timedelta(days=1)
                if idx + 1 < len(incomes)
                else month_start.replace(day=28)
            )
            seg_start = this_date + timedelta(days=1)

            if idx == 0 and rec_expenses:
                for exp in rec_expenses:
                    exp_day = parse_date(exp["date"]).day
                    exp_date = month_start.replace(day=min(exp_day, 28))
                    if not (this_date <= exp_date <= next_date):
                        continue
                    amt = sample_expense_amount(exp["category"], month_behavior)
                    if remaining - amt < savings_threshold:
                        allowed = remaining - savings_threshold
                        if allowed < 20:
                            continue
                        amt = allowed
                    remaining -= amt
                    day_type = "weekend" if exp_date.weekday() >= 5 else "weekday"
                    synthesized.append(
                        {
                            "user_id": user["user_id"],
                            "age": prof.get("age"),
                            "occupation": prof.get("occupation"),
                            "monthly_income": current_income,
                            "date": fmt_date(exp_date),
                            "amount": -amt,
                            "type": "expense",
                            "category": exp["category"],
                            "merchant": exp["merchant"],
                            "payment_mode": choose_payment_mode(
                                exp["category"], month_behavior
                            ),
                            "recurring": True,
                            "day_type": day_type,
                            "persona_label": prof.get("persona_label"),
                        }
                    )

            attempts = 0
            pool = build_category_pool(expense_templates, month_behavior)
            while remaining > savings_threshold:
                attempts += 1
                if attempts > 400 or random.random() > daily_prob:
                    if attempts > 400:
                        break
                    continue
                if seg_start > next_date:
                    break
                offset = random.randint(0, (next_date - seg_start).days)
                exp_date = seg_start + timedelta(days=offset)
                exp_date = _adjust_to_weekend_or_weekday(exp_date, weekend_bias)
                cat = random.choice(pool)
                amt = sample_expense_amount(cat, month_behavior)
                if remaining - amt < savings_threshold:
                    allowed = remaining - savings_threshold
                    if allowed < 20:
                        break
                    amt = allowed
                remaining -= amt
                day_type = "weekend" if exp_date.weekday() >= 5 else "weekday"
                synthesized.append(
                    {
                        "user_id": user["user_id"],
                        "age": prof.get("age"),
                        "occupation": prof.get("occupation"),
                        "monthly_income": current_income,
                        "date": fmt_date(exp_date),
                        "amount": -amt,
                        "type": "expense",
                        "category": cat,
                        "merchant": f"{cat} Vendor",
                        "payment_mode": choose_payment_mode(cat, month_behavior),
                        "recurring": False,
                        "day_type": day_type,
                        "persona_label": prof.get("persona_label"),
                    }
                )

    return synthesized


# ------------------------------------------------------
def load_base_users(path):
    with open(path, "r") as f:
        return json.load(f)


def expand_base_users(base_users, num_variants=20):
    """
    Expand base user templates into many synthetic users per type,
    adding behavior differences by user-type and variant role.
    """
    expanded = []

    for u in base_users:
        base_id = u["user_id"]
        base_profile = u.get("profile", {})
        base_age = base_profile.get("age")
        base_income_numeric = get_numeric_monthly_income(u)
        base_occupation = base_profile.get("occupation", "Professional")

        user_type = classify_user_type(u)
        base_mi = base_profile.get("monthly_income")

        # NEW: detect fixed salary from base profile occupation
        base_fixed_salary = get_fixed_salary_for_profile(base_profile)

        for i in range(num_variants):
            v = deepcopy(u)
            role_idx = i % 5
            role_name = ROLE_NAMES[role_idx]

            v_id = f"{base_id}_v{i+1}"
            v["user_id"] = v_id

            v.setdefault("profile", {})
            v["profile"]["user_type"] = user_type
            v["profile"]["variant_role"] = role_name
            v["profile"]["persona_label"] = f"{user_type}_{role_name}"

            if isinstance(base_age, (int, float)):
                jitter_age = base_age + random.randint(-2, 2)
                v["profile"]["age"] = max(18, jitter_age)
            else:
                v["profile"]["age"] = base_age

            behavior = {
                "spend_multiplier": 1.0,
                "daily_expense_prob": DAILY_EXPENSE_PROB,
                "spike_prob": 0.0,
                "spike_multiplier": 1.0,
                "category_multipliers": {},
                "subscription_bias": 0.0,
                "food_bias": 0.0,
                "transport_bias": 0.0,
                "force_cash_for_discretionary": False,
                "force_digital_for_most": False,
                "income_volatility": 0.2,
                "category_profile": _make_random_category_profile(),
                "weekend_spend_bias": 0.0,
                "min_savings_multiplier": 1.0,
            }

            # ----- User-type base behavior + fixed salary override -----
            if base_fixed_salary is not None:
                # For software engineer / college student: hard-set salary
                v["profile"]["monthly_income"] = int(base_fixed_salary)
                if user_type == "freelancer":
                    behavior["income_volatility"] = 0.0
            else:
                if user_type == "salaried":
                    base_factor = random.uniform(0.9, 1.1)
                    v["profile"]["monthly_income"] = int(base_income_numeric * base_factor)
                elif user_type == "freelancer":
                    v["profile"]["monthly_income"] = base_mi or "variable"
                    behavior["income_volatility"] = 0.2
                    behavior["daily_expense_prob"] = DAILY_EXPENSE_PROB * 1.2
                elif user_type == "student":
                    base_factor = random.uniform(0.25, 0.5)
                    v["profile"]["monthly_income"] = int(base_income_numeric * base_factor)
                    behavior["category_multipliers"] = {
                        "Food": 1.8,
                        "Transport": 1.8,
                        "Rent": 0.25,
                        "Investment": 0.1,
                        "Travel": 0.4,
                    }
                    behavior["food_bias"] = 1.2
                    behavior["transport_bias"] = 1.4
                    behavior["force_digital_for_most"] = True
                    behavior["daily_expense_prob"] = DAILY_EXPENSE_PROB * 1.3

            # ----- Variant role behavior -----
            if role_idx == 0:  # low
                behavior["spend_multiplier"] *= 0.4
                behavior["daily_expense_prob"] *= 0.5
                behavior["weekend_spend_bias"] = -0.2
                behavior["min_savings_multiplier"] = 1.5   # higher savings
            elif role_idx == 1:  # moderate
                behavior["min_savings_multiplier"] = 1.0
            elif role_idx == 2:  # impulsive
                behavior["spend_multiplier"] *= 1.5
                behavior["daily_expense_prob"] *= 2.0
                behavior["spike_prob"] = 0.25
                behavior["spike_multiplier"] = 4.0
                behavior["weekend_spend_bias"] = 0.6
                behavior["min_savings_multiplier"] = 0.3   # MUCH lower required savings
                behavior.setdefault("category_multipliers", {})
                for c in ["Shopping", "Entertainment", "Travel"]:
                    behavior["category_multipliers"][c] = (
                        behavior["category_multipliers"].get(c, 1.0) * 1.8
                    )
            elif role_idx == 3:  # cash
                behavior["force_cash_for_discretionary"] = True
                behavior["weekend_spend_bias"] = 0.1
                behavior["min_savings_multiplier"] = 1.0
            elif role_idx == 4:  # subscription
                behavior["subscription_bias"] = 2.5
                behavior.setdefault("category_multipliers", {})
                behavior["category_multipliers"]["Subscriptions"] = 3.0
                behavior["weekend_spend_bias"] = -0.1
                behavior["min_savings_multiplier"] = 0.8

            v["profile"]["occupation"] = base_occupation
            v["behavior"] = behavior
            expanded.append(v)

    return expanded


def cap_monthly_overspend(df):
    """
    Ensure that, for each user_id + year_month, total expenses
    do not exceed total income. If they do, trim the last expense.
    """
    df = df.sort_values(["user_id", "year_month", "date"]).copy()
    grouped = df.groupby(["user_id", "year_month"])

    for (uid, ym), idx in grouped.groups.items():
        sub = df.loc[idx]

        income = sub.loc[sub["amount"] > 0, "amount"].sum()
        expense = -sub.loc[sub["amount"] < 0, "amount"].sum()

        if expense > income:
            overshoot = expense - income
            exp_idx = sub[sub["amount"] < 0].index
            if len(exp_idx) == 0:
                continue
            last_idx = exp_idx[-1]
            old_amt = df.at[last_idx, "amount"]
            new_amt = old_amt + overshoot
            if new_amt >= 0:
                df.drop(index=last_idx, inplace=True)
            else:
                df.at[last_idx, "amount"] = new_amt

    return df


def normalize_and_combine(base_users, synthesized):
    """
    Build final dataframe and compute balances.

    NEW:
    - Ensure spending per user+month is between ~50% and 85% of income.
    - Apply per-month inflation factor (1.02–1.15) to expenses.
    """
    rows = []
    user_income_map = {u["user_id"]: get_numeric_monthly_income(u) for u in base_users}
    user_is_freelancer = {
        u["user_id"]: (
            (u.get("profile", {}).get("user_type") == "freelancer")
            or isinstance(u.get("profile", {}).get("monthly_income"), str)
            or ("freelanc" in str(u.get("profile", {}).get("occupation", "")).lower())
        )
        for u in base_users
    }

    for u in base_users:
        uid = u["user_id"]
        prof = u.get("profile", {})
        mi = user_income_map.get(uid, 0)
        is_var = isinstance(prof.get("monthly_income"), str)
        persona_label = prof.get("persona_label", "base_template")
        for t in u.get("transactions", []):
            try:
                amt = float(t.get("amount", 0))
            except Exception:
                amt = 0.0
            amt = -abs(amt) if t.get("type") == "expense" else abs(amt)

            rows.append(
                {
                    "user_id": uid,
                    "age": prof.get("age"),
                    "occupation": prof.get("occupation"),
                    "monthly_income": int(mi),
                    "is_variable_income": bool(is_var),
                    "date": t.get("date"),
                    "amount": amt,
                    "type": t.get("type"),
                    "category": t.get("category"),
                    "merchant": t.get("merchant"),
                    "payment_mode": t.get("payment_mode"),
                    "recurring": bool(t.get("recurring", False)),
                    "day_type": t.get("day_type"),
                    "persona_label": persona_label,
                }
            )

    for t in synthesized:
        try:
            amt = float(t.get("amount", 0))
        except Exception:
            amt = 0.0
        amt = -abs(amt) if t.get("type") == "expense" else abs(amt)

        rows.append(
            {
                "user_id": t.get("user_id"),
                "age": t.get("age"),
                "occupation": t.get("occupation"),
                "monthly_income": t.get(
                    "monthly_income", user_income_map.get(t.get("user_id"), 0)
                ),
                "is_variable_income": bool(
                    user_is_freelancer.get(t.get("user_id"), False)
                ),
                "date": t.get("date"),
                "amount": amt,
                "type": t.get("type"),
                "category": t.get("category"),
                "merchant": t.get("merchant"),
                "payment_mode": t.get("payment_mode"),
                "recurring": bool(t.get("recurring", False)),
                "day_type": t.get("day_type"),
                "persona_label": t.get("persona_label"),
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # ---------- Option 1: realistic expense/income ratio ----------
    def _adjust_spend_ratio(group: pd.DataFrame) -> pd.DataFrame:
        income = group.loc[group["amount"] > 0, "amount"].sum()
        expense = -group.loc[group["amount"] < 0, "amount"].sum()
        if income <= 0 or expense <= 0:
            return group

        current_ratio = expense / income
        min_ratio, max_ratio = SPEND_SHARE_RANGE

        # already realistic enough
        if current_ratio >= min_ratio:
            return group

        target_ratio = random.uniform(min_ratio, max_ratio)
        scale = target_ratio / max(current_ratio, 1e-3)

        mask = group["amount"] < 0
        group.loc[mask, "amount"] = (group.loc[mask, "amount"] * scale).astype(int)
        # ensure expenses remain negative and at least -20
        group.loc[mask, "amount"] = group.loc[mask, "amount"].clip(upper=-20)
        return group

    df = df.groupby(["user_id", "year_month"], group_keys=False).apply(_adjust_spend_ratio)

    # ---------- Option 3: inflation-like monthly scaling ----------
    unique_months = df["year_month"].dropna().unique().tolist()
    inflation_factors = {
        ym: random.uniform(*INFLATION_RANGE) for ym in unique_months
    }

    def _apply_inflation(row):
        amt = row["amount"]
        if amt < 0:  # only scale expenses
            factor = inflation_factors.get(row["year_month"], 1.0)
            return int(amt * factor)
        return amt

    df["amount"] = df.apply(_apply_inflation, axis=1)

    # After scaling, ensure we don't wildly overshoot income
    df = cap_monthly_overspend(df)

    # --------- rest of the pipeline (unchanged) ----------
    income_by_user_month = (
        df[df["amount"] > 0]
        .groupby(["user_id", "year_month"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "month_income"})
    )
    df = df.merge(income_by_user_month, on=["user_id", "year_month"], how="left")
    df["month_income"] = df["month_income"].fillna(0).astype(int)
    df["monthly_income"] = df["month_income"]

    df["is_freelancer_user"] = df["user_id"].map(user_is_freelancer).fillna(False)

    def priority_fn(row):
        if row["is_freelancer_user"]:
            return 0 if (row["amount"] > 0) else 1
        return 0

    df["priority"] = df.apply(priority_fn, axis=1)

    df.sort_values(["user_id", "year_month", "priority", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype(int)
    df["running_balance"] = df.groupby("user_id")["amount"].cumsum().astype(int)

    df.drop(
        columns=["month_income", "is_freelancer_user", "priority"],
        inplace=True,
        errors="ignore",
    )

    return df


def main():
    base_users = load_base_users(BASE_JSON_PATH)
    expanded_users = expand_base_users(
        base_users, num_variants=NUM_VARIANTS_PER_BASE_USER
    )

    synthesized = []
    for u in expanded_users:
        synthesized.extend(synthesize_for_user(u, months_to_add=months_to_synthesize))

    df = normalize_and_combine(expanded_users, synthesized)
    df.to_csv(OUT_CSV_PATH, index=False)
    print(f"Synthesized dataset saved to: {OUT_CSV_PATH}")
    print(f"Total users: {df['user_id'].nunique()}, total rows: {len(df)}")


if __name__ == "__main__":
    main()
