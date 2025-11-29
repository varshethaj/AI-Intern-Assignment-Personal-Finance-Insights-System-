"""
llm_integration.py

LLM integration layer:
- Reads analytics payload from `insight_engine.build_analytics_payload`.
- Calls Gemini to generate financial insights.
- Optionally supports follow-up Q&A over the same payload (CLI demo).

We support two styles:
1) generate_llm_insights_basic  -> pure workflow style (Approach 2)
2) generate_llm_insights        -> mini-agent style that uses insight_candidates
"""

# ===============================================================
# LLM INTEGRATION — Uses Gemini to turn analytics payload
# into natural-language insights + follow-up Q&A
# ===============================================================

import json
import os

from dotenv import load_dotenv
import google.generativeai as genai

from insight_engine import build_analytics_payload
import config  # <-- central configuration

# Load environment variables from .env (for GEMINI_API_KEY)
load_dotenv()


def _get_gemini_model() -> genai.GenerativeModel:
    """
    Configure and return a Gemini model instance.

    Reads:
        - GEMINI_API_KEY from environment
        - MODEL_NAME from config.MODEL_NAME
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("[ERROR] GEMINI_API_KEY not found in environment.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(config.MODEL_NAME)


# ===============================================================
# =============  INSIGHT GENERATION (BASIC – APPROACH 2) ========
# ===============================================================

def generate_llm_insights_basic(payload: dict, max_insights: int | None = None) -> str:
    """
    Basic workflow-style insight generation:
    - LLM reads the numeric summary and comparative sections.
    - It does NOT look at insight_candidates or priorities.
    This matches "Approach 2: LLM in workflow" from the assignment.
    """
    if max_insights is None:
        max_insights = config.MAX_INSIGHTS

    strict_clause = ""
    if config.STRICT_HALLUCINATION_MODE:
        strict_clause = """
- When you state a number (amount or percentage), copy it exactly from the JSON.
- If a specific number or fact is not present in the JSON, do NOT guess it.
- If you are unsure, say so briefly instead of inventing details.
"""

    prompt = f"""
You are a financial assistant analyzing a user's spending behavior.

Your task:
- Read the JSON provided below (focus especially on "summary" and "comparative").
- Generate ONLY {max_insights} short insights (1–2 lines each).
- Tone: helpful, concise, and actionable.
- Use bullet points (•).
- For each insight, you may also add a short recommendation if relevant.
- Use currency formatting like ₹71,657 (no trailing '.0') and percentages with at most two decimals.
- Do NOT hallucinate — only use information present in the JSON.
{strict_clause}

JSON Data:
{json.dumps(payload, indent=2)}

Now respond ONLY with the bullet point insights:
    """

    try:
        model = _get_gemini_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR calling Gemini]: {e}"


# ===============================================================
# =============  INSIGHT GENERATION (MINI AGENT) ================
# ===============================================================

def generate_llm_insights(payload: dict, max_insights: int | None = None) -> str:
    """
    Mini-agent style insight generation (Approach 1 on top of 2):

    - Uses the same analytics payload produced by the engine.
    - Additionally looks at:
        - payload["insight_candidates"]  (each has type + priority + data)
        - payload["financial_health"]
    - The LLM acts like a 'planner':
        - Chooses which candidate insights to surface (highest priority first)
        - Ensures coverage of key dimensions (esp. financial health)
        - Produces insight + one-line recommendation for each.

    This lets you say:
      "We implemented Approach 2 for analytics, and a light agentic layer (Approach 1)
       for insight selection and prioritisation."
    """
    if max_insights is None:
        max_insights = config.MAX_INSIGHTS

    strict_clause = ""
    if config.STRICT_HALLUCINATION_MODE:
        strict_clause = """
- When you state a number (amount or percentage), copy it exactly from the JSON.
- If a specific number or fact is not present in the JSON, do NOT guess it.
- If you are unsure, say so briefly instead of inventing details.
"""

    prompt = f"""
You are a financial assistant analyzing a user's finances.

You are given:
- An analytics payload with summary statistics.
- A list of "insight_candidates" that already include:
    - id (string)
    - type (e.g., "spending_pattern", "recurring_expenses", etc.)
    - priority (0–1, higher = more important)
    - data (structured fields used for that insight)
- A "financial_health" section that summarises overall health.

Act like a small analysis agent:
1) First, scan "insight_candidates" and treat them as a ranked list of what matters most.
2) Select the top {max_insights} distinct ideas to talk about (highest priority first).
   - If multiple candidates are about the same theme, combine them into one insight.
3) Always include at least ONE insight that summarises overall financial health.

For each chosen insight:
- Write ONE short sentence explaining the situation.
- Then write ONE short recommendation starting with a verb (e.g., "Consider...", "Review...", "Set a limit on...").

Output format (very important, follow exactly):

• <insight sentence>
  → Recommendation: <one short actionable recommendation>

Additional rules:
- Use currency formatting like ₹71,657 (no trailing ".0").
- Use at most two decimal places for percentages (e.g., 84.30%).
- Do NOT hallucinate — only use information present in the JSON.
{strict_clause}

JSON Data:
{json.dumps(payload, indent=2)}

Now respond ONLY with the list of bullet insights in the specified format:
    """

    try:
        model = _get_gemini_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR calling Gemini]: {e}"


# ===============================================================
# =============  FOLLOW-UP QUESTION ANSWERING ===================
# ===============================================================

def answer_followup_question(payload: dict, question: str) -> str:
    """
    Given the analytics payload and a natural-language question from the user,
    answer ONLY using the data in the payload.

    If the answer is not present, the model is allowed to infer high-level
    recommendations but NOT fabricate numbers.
    """

    strict_clause = ""
    if config.STRICT_HALLUCINATION_MODE:
        strict_clause = """
- Do NOT create or invent numbers that are not explicitly present in the JSON.
- If exact numeric information is missing, you MAY provide general reasoning-based advice or recommendations.
- Recommendations should be based ONLY on visible patterns (example: high recurring cost, low savings rate, category spikes, heavy payment dependency).
- Avoid hallucinating transactions or merchants that are not present in the payload.
"""

    prompt = f"""
You are a financial assistant.

You are given:
1) A JSON analytics payload describing a user's finances.
2) A follow-up question from the user.

Rules:
- Use the JSON as your primary source of truth.
- Be concise (2–4 sentences).
- Do NOT fabricate numbers or transactions not present in the payload.
{strict_clause}

JSON Data:
{json.dumps(payload, indent=2)}

User question:
{question}

Now provide your answer:
    """

    try:
        model = _get_gemini_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR calling Gemini]: {e}"


# ===============================================================
# =============  OPTIONAL: PROGRAMMATIC ENTRYPOINT ==============
# ===============================================================

def run_insights_for_user(
    user_id: str,
    year_month: str,
    data_dir: str | None = None,
    use_agentic: bool = True,
):
    """
    Convenience wrapper if you want to call this from other Python code
    (e.g., API / UI). Returns (payload, insights_string).

    Set use_agentic=False to use the basic workflow mode.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    payload = build_analytics_payload(
        user_id=user_id,
        year_month=year_month,
        data_dir=data_dir,
    )
    if use_agentic:
        insights = generate_llm_insights(payload)
    else:
        insights = generate_llm_insights_basic(payload)
    return payload, insights


# ===============================================================
# =============  MAIN EXECUTION (demo / CLI) ====================
# ===============================================================

def _print_cli_header(payload: dict) -> None:
    """Pretty-print a small header with key metrics before insights."""
    up = payload.get("user_profile", {})
    period = payload.get("period", {})
    summary = payload.get("summary", {})
    fh = payload.get("financial_health", {})

    print("\n=== USER PROFILE ===")
    print(f"- ID: {up.get('user_id')}")
    if up.get("age") is not None or up.get("occupation"):
        print(f"- Age: {up.get('age')}  |  Occupation: {up.get('occupation')}")
    if up.get("monthly_income") is not None:
        print(f"- Monthly Income: {up.get('monthly_income')}")

    print("\n=== PERIOD ===")
    print(f"- Label: {period.get('label')}")
    print(f"- From:  {period.get('start_date')}  To: {period.get('end_date')}")

    inc = summary.get("income", {})
    spend = summary.get("spending", {})
    print("\n=== KEY METRICS ===")
    print(f"- Total Income:  {inc.get('total_income')}")
    print(f"- Total Expense: {spend.get('total_expense')}")
    print(f"- Savings:       {spend.get('savings')}")
    print(f"- Savings Rate:  {round(spend.get('savings_rate_pct', 0.0), 2)}%")

    if fh:
        print(f"- Financial Health Risk: {fh.get('risk_level')}")


if __name__ == "__main__":
    # Get values from the user instead of hardcoding
    user_id = input("Enter user_id (e.g., user_002_v1): ").strip()
    year_month = input("Enter year_month (YYYY-MM, e.g., 2025-11): ").strip()

    # Basic validation for year_month format
    if len(year_month) != 7 or year_month[4] != "-":
        print("[ERROR] year_month must be in YYYY-MM format, e.g., 2025-11.")
        raise SystemExit(1)

    # Build analytics payload using the engine
    try:
        payload = build_analytics_payload(
            user_id=user_id,
            year_month=year_month,
            data_dir=config.DATA_DIR,  # centralised path
        )
    except Exception as e:
        print(f"[ERROR building analytics payload]: {e}")
        raise SystemExit(1)

    _print_cli_header(payload)

    print("\n=== AI-GENERATED INSIGHTS (Mini-Agent Mode) ===")
    insights = generate_llm_insights(payload)   # uses agentic version by default
    print(insights)

    # -------- Optional: interactive follow-up Q&A loop ----------
    if config.ENABLE_FOLLOWUP_QA:
        print("\nYou can now ask follow-up questions about this month's data.")
        print("Type 'exit' to quit.")

        conversation_log: list[str] = []
        conversation_log.append("=== AI-GENERATED INSIGHTS (Mini-Agent) ===\n" + insights + "\n")

        while True:
            q = input("\nAsk a follow-up question (or 'exit'): ").strip()
            if q.lower() in ("exit", "quit", "q"):
                print("Exiting follow-up Q&A.")
                break

            if not q:
                print("Please enter a question or type 'exit'.")
                continue

            answer = answer_followup_question(payload, q)
            print("\n[Answer]")
            print(answer)

            conversation_log.append(f"[Q] {q}\n[A] {answer}\n")

    else:
        print("\n[Follow-up Q&A is disabled via configuration.]")
