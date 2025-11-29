"""
config.py

Central configuration for the finance insights system.
This keeps all tunable parameters in one place so that
the behaviour of the system can be changed without
modifying the core logic.
"""

# Directory where all CSV data files are stored
DATA_DIR: str = "data"

# Which Gemini model to use for generation
MODEL_NAME: str = "gemini-2.5-flash"

# Maximum number of bullet-point insights to generate
MAX_INSIGHTS: int = 5

# Whether to allow interactive follow-up Q&A in CLI mode
ENABLE_FOLLOWUP_QA: bool = True

# If True, add stronger instructions in prompts to avoid hallucinations
STRICT_HALLUCINATION_MODE: bool = True
