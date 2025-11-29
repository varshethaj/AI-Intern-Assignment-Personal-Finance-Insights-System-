"""
streamlit_app.py

Minimal Streamlit UI for the Personal Finance Insights System.

Features:
- Pick a user_id and month from available data.
- Run the full analytics pipeline + LLM insights.
- Show key metrics, AI-generated insights, and follow-up Q&A.
"""

import os
import json

import streamlit as st
import pandas as pd

import config
from insight_engine import build_analytics_payload
from llm_integration import generate_llm_insights, answer_followup_question


# ---------------------- FORMAT HELPERS ---------------------- #

def format_money(amount) -> str:
    """Format a number as Indian Rupee with commas, safe for None/bad values."""
    try:
        if amount is None:
            return "₹0"
        return f"₹{float(amount):,.0f}"
    except Exception:
        return "₹0"


import re

def normalize_bullets(text: str) -> str:
    """
    Convert Gemini output into nice bullet points.

    Handles:
    - Already bulleted text with •, -, or *.
    - Single paragraph: splits into sentences and bullets them.
    """
    t = text.strip()
    if not t:
        return t

    # Normalize common bullet markers into a single separator
    # e.g. "• text", "- text", "* text"
    pattern = r"(?:^|\n)[\-\*\u2022]\s*"   # \u2022 = •
    if re.search(pattern, t):
        # Text already has bullets -> split on them
        parts = re.split(pattern, t)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return "\n\n".join(f"• {p}" for p in parts)

    # Fallback: no clear bullets, so split into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    if len(sentences) > 1:
        return "\n\n".join(f"• {s}" for s in sentences)

    # If everything fails, just return original
    return t

def render_insights(insights_text: str):
    st.subheader("🤖 AI-Generated Insights")
    if not insights_text.strip():
        st.warning("No insights returned from the LLM.")
        return

    pretty_text = normalize_bullets(insights_text)
    st.markdown(pretty_text)


# ---------------------- DATA HELPERS ---------------------- #

@st.cache_data
def load_user_month_index(data_dir: str) -> pd.DataFrame:
    """
    Load user-month combinations from user_month_signals.csv
    so the UI can offer valid user_id + year_month options.
    """
    path = os.path.join(data_dir, "user_month_signals.csv")
    df = pd.read_csv(path)

    if "year_month" in df.columns:
        df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")
        df["year_month_str"] = df["year_month"].dt.to_period("M").astype(str)
    else:
        # Fallback: try to rebuild year_month from another date column if needed
        for col in ["date", "txn_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df["year_month_str"] = df[col].dt.to_period("M").astype(str)
                break

    if "year_month_str" not in df.columns:
        raise RuntimeError("Could not determine 'year_month' column for UI.")

    return df[["user_id", "year_month_str"]].dropna().drop_duplicates()


def get_available_users(df_index: pd.DataFrame) -> list[str]:
    return sorted(df_index["user_id"].unique().tolist())


def get_available_months_for_user(df_index: pd.DataFrame, user_id: str) -> list[str]:
    subset = df_index[df_index["user_id"] == user_id]
    months = sorted(subset["year_month_str"].unique().tolist())
    return months


# ---------------------- UI HELPERS ------------------------ #

def render_header():
    st.title("💰 AI Finance Insights – Demo UI")
    st.caption(
        "End-to-end personal finance analytics with anomaly detection, "
        "clustering, prediction, and LLM-based insight generation."
    )


def render_key_metrics(payload: dict):
    summary = payload.get("summary", {})
    income = summary.get("income", {})
    spending = summary.get("spending", {})
    fh = payload.get("financial_health", {})

    total_income = income.get("total_income", 0)
    total_expense = spending.get("total_expense", 0)
    savings = spending.get("savings", 0)
    savings_rate_pct = spending.get("savings_rate_pct", 0.0)

    st.subheader("💰 Financial Snapshot")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Total Income**")
        st.markdown(format_money(total_income))
    with c2:
        st.markdown("**Total Expense**")
        st.markdown(format_money(total_expense))
    with c3:
        st.markdown("**Savings**")
        st.markdown(format_money(savings))
    with c4:
        st.markdown("**Savings Rate**")
        st.markdown(f"{savings_rate_pct:.2f}%")

    # Financial health badge
    risk = fh.get("risk_level", "unknown")
    unrealistic = fh.get("unrealistic_savings_suspected", False)

    risk_label = {
        "low": "✅ Low",
        "medium": "⚠️ Medium",
        "high": "🔴 High",
    }.get(risk, f"❓ {risk}")

    st.write("")
    st.markdown(f"**Financial Health Risk:** {risk_label}")
    if unrealistic:
        st.info(
            "The system suspects your savings pattern may be unrealistically high. "
            "This could indicate missing expenses or incomplete tracking."
        )


def render_profile_and_period(payload: dict):
    up = payload.get("user_profile", {})
    period = payload.get("period", {})

    st.subheader("👤 User Profile")
    st.write(
        f"- **User ID:** {up.get('user_id')}\n"
        f"- **Age:** {up.get('age')}\n"
        f"- **Occupation:** {up.get('occupation')}\n"
        f"- **Monthly Income (reported):** {up.get('monthly_income')}"
    )

    st.subheader("📅 Period")
    st.write(
        f"- **Label:** {period.get('label')}\n"
        f"- **Start:** {period.get('start_date')}  \n"
        f"- **End:** {period.get('end_date')}"
    )


def render_insights(insights_text: str):
    st.subheader("🤖 AI-Generated Insights")
    if not insights_text.strip():
        st.warning("No insights returned from the LLM.")
        return

    pretty_text = normalize_bullets(insights_text)
    st.markdown(pretty_text)


def render_payload_debug(payload: dict):
    with st.expander("🔍 Advanced: View raw analytics payload (JSON)", expanded=False):
        st.json(payload)


def render_followup_qa(payload: dict):
    st.subheader("💬 Ask a follow-up question")

    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []

    question = st.text_input(
        "Type a question about this month's finances (e.g., "
        "'Which categories did I spend the most on?')"
    )

    col_q1, col_q2 = st.columns([1, 3])
    with col_q1:
        ask_clicked = st.button("Ask", use_container_width=True)
    with col_q2:
        clear_clicked = st.button("Clear history", use_container_width=True)

    if clear_clicked:
        st.session_state["qa_history"] = []

    if ask_clicked and question.strip():
        with st.spinner("Thinking..."):
            answer = answer_followup_question(payload, question.strip())
        st.session_state["qa_history"].append(
            {"question": question.strip(), "answer": answer}
        )

    # Display history
    if st.session_state["qa_history"]:
        st.markdown("#### Q&A History")
        for i, qa in enumerate(st.session_state["qa_history"], start=1):
            st.markdown(f"**Q{i}. {qa['question']}**")
            st.markdown(f"> {qa['answer']}")
            st.markdown("---")


# ---------------------- MAIN APP -------------------------- #

def main():
    st.set_page_config(
        page_title="AI Finance Insights",
        page_icon="💰",
        layout="wide",
    )
    render_header()

    # Load index of available user-month combinations
    try:
        df_index = load_user_month_index(config.DATA_DIR)
    except Exception as e:
        st.error(f"Error loading user index: {e}")
        return

    with st.sidebar:
        st.header("⚙️ Controls")

        users = get_available_users(df_index)
        if not users:
            st.error("No users found in data. Check user_month_signals.csv.")
            return

        selected_user = st.selectbox("User ID", users)

        months = get_available_months_for_user(df_index, selected_user)
        if not months:
            st.error("No months available for this user.")
            return

        selected_month = st.selectbox("Year-Month", months)

        st.markdown("---")
        run_button = st.button("🚀 Run Insights", use_container_width=True)

    if run_button:
        st.session_state["payload"] = None
        st.session_state["insights"] = ""

        with st.spinner("Building analytics payload and calling LLM..."):
            try:
                payload = build_analytics_payload(
                    user_id=selected_user,
                    year_month=selected_month,
                    data_dir=config.DATA_DIR,
                )
            except Exception as e:
                st.error(f"Error building analytics payload: {e}")
                return

            insights = generate_llm_insights(payload)

        st.session_state["payload"] = payload
        st.session_state["insights"] = insights
        st.success("Insights generated successfully!")

    # Display results if we already have them
    if "payload" in st.session_state and st.session_state["payload"] is not None:
        payload = st.session_state["payload"]
        insights = st.session_state.get("insights", "")

        col_left, col_right = st.columns([2, 3])

        with col_left:
            render_profile_and_period(payload)
            st.markdown("---")
            render_key_metrics(payload)

        with col_right:
            render_insights(insights)

        st.markdown("---")
        render_followup_qa(payload)
        render_payload_debug(payload)
    else:
        st.info("Select a user and month in the sidebar, then click **Run Insights**.")


if __name__ == "__main__":
    main()
