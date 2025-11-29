# 💰 AI Intern Assignment — Personal Finance Insights System

A comprehensive data analytics and machine learning pipeline for personal finance pattern analysis, featuring automated insights generation powered by Google Gemini.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Features](#core-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
- [Data Pipeline](#data-pipeline)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

The **Personal Finance Insights System** is an end-to-end analytics platform that processes transaction data to generate actionable financial insights. The system combines traditional data engineering, machine learning models, and large language models to provide users with:

- **Automated pattern detection** in spending behavior
- **Anomaly identification** for unusual transactions
- **Predictive analytics** for future spending and savings
- **User persona clustering** based on financial behavior
- **AI-powered insights** with natural language explanations
- **Interactive Q&A** for follow-up financial questions

The system is built with a modular architecture, making it easy to extend and customize for different use cases.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Raw Transaction Data (CSV)                              │  │
│  │  • synthetic_full_dataset.csv                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PREPROCESSING & FEATURE ENGINEERING             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Data Cleanup │  │ Daily Features│  │ Monthly Features│     │
│  │ & Normalize  │→ │ Engineering  │→ │ Engineering   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYTICS ENGINES                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Pattern      │  │ Signals      │  │ Anomaly      │         │
│  │ Identification│→ │ Engine       │→ │ Detection    │         │
│  │              │  │ (Rule+Stats) │  │ (Z-score+IF) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Prediction   │  │ Clustering   │  │ Anomaly      │         │
│  │ (XGBoost)    │  │ (KMeans)     │  │ (Isolation   │         │
│  │              │  │              │  │  Forest)     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INSIGHT ENGINE                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Consolidates all engine outputs into structured JSON    │  │
│  │  • Financial health metrics                              │  │
│  │  • Insight candidates with priorities                   │  │
│  │  • Comparative analysis                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM INTEGRATION LAYER                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Google Gemini API                                        │  │
│  │  • Insight generation (mini-agent style)                 │  │
│  │  • Follow-up Q&A support                                 │  │
│  │  • Natural language explanations                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                                │
│  ┌──────────────┐              ┌──────────────┐               │
│  │ Streamlit UI │              │ CLI Mode     │               │
│  │ (Interactive)│              │ (Programmatic)│              │
│  └──────────────┘              └──────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Core Features

### 🔄 Automated Data Pipeline
- **Ingestion**: Processes raw transaction CSV files
- **Preprocessing**: Cleans, normalizes, and validates transaction data
- **Feature Engineering**: Generates daily and monthly aggregated features
- **Pattern Identification**: Extracts temporal and categorical spending patterns

### 🔍 Anomaly Detection
- **Z-Score Analysis**: Identifies transactions deviating significantly from user's normal behavior
- **Isolation Forest**: Detects multivariate anomalies in transaction patterns
- **Spike Detection**: Flags unusual spending spikes and outliers
- **Transaction-Level Flagging**: Marks anomalous transactions with reasons

### 📊 Predictive Analytics
- **Next-Month Spending Prediction**: XGBoost regression model for expense forecasting
- **Savings Rate Prediction**: Forecasts future savings behavior
- **Overspend Risk Assessment**: Binary classification for overspending probability
- **Lag-Based Features**: Uses historical patterns (1, 3, 6 months) for predictions

### 👥 User Persona Clustering
- **KMeans Clustering**: Groups users into behavioral personas (3 clusters)
- **User-Level Aggregation**: Creates feature vectors from monthly patterns
- **Peer Comparison**: Compares individual users against their cluster peers
- **Behavioral Segmentation**: Identifies spending, savings, and volatility patterns

### 🚦 Rule-Based & Statistical Signals
- **Rule-Based Flags**: Detects low savings, high volatility, merchant dependency, etc.
- **Statistical Signals**: Z-score based unusual spending/savings detection
- **Comparative Analysis**: Month-over-month changes in spending, income, categories
- **Temporal Patterns**: Weekend vs. weekday spending analysis

### 🤖 AI-Powered Insights
- **Gemini Integration**: Uses Google Gemini 2.5 Flash for natural language generation
- **Mini-Agent Architecture**: Prioritizes and selects insights from candidate pool
- **Actionable Recommendations**: Provides specific, data-driven financial advice
- **Follow-Up Q&A**: Supports conversational queries about financial data

### 🖥️ User Interface
- **Streamlit Web App**: Interactive dashboard for exploring insights
- **CLI Mode**: Command-line interface for programmatic access
- **Real-Time Analysis**: On-demand insights generation for any user-month combination

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd AI_Intern_Assignment
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
# .env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Important**: Never commit the `.env` file to version control. It should already be in `.gitignore`.

---

## ⚙️ Configuration

The system uses a central configuration file (`config.py`) for easy customization:

```python
# config.py
DATA_DIR = "data"                    # Directory for CSV data files
MODEL_NAME = "gemini-2.5-flash"      # Gemini model to use
MAX_INSIGHTS = 5                      # Maximum insights to generate
ENABLE_FOLLOWUP_QA = True             # Enable interactive Q&A
STRICT_HALLUCINATION_MODE = True      # Prevent LLM from inventing numbers
```

You can modify these values directly in `config.py` to adjust system behavior.

---

## 🚀 Usage

### Running the Streamlit UI

The easiest way to interact with the system is through the Streamlit web interface:

```bash
streamlit run streamlit_app.py
```

This will:
1. Open a web browser with the Streamlit interface
2. Allow you to select a user ID and month from available data
3. Generate and display insights in real-time
4. Support follow-up questions about the financial data

**Features in Streamlit UI:**
- User and month selection dropdowns
- Financial snapshot with key metrics
- AI-generated insights with recommendations
- Interactive Q&A section
- Raw JSON payload viewer (for debugging)

### Running via CLI

For programmatic access or batch processing:

```bash
python llm_integration.py
```

The CLI will prompt you for:
- **User ID**: e.g., `user_002_v1`
- **Year-Month**: e.g., `2025-11` (format: `YYYY-MM`)

After generating insights, you can ask follow-up questions interactively. Type `exit` to quit.

### Programmatic Usage

You can also import and use the system in your own Python code:

```python
from insight_engine import build_analytics_payload
from llm_integration import generate_llm_insights, answer_followup_question

# Build analytics payload
payload = build_analytics_payload(
    user_id="user_002_v1",
    year_month="2025-11",
    data_dir="data"
)

# Generate insights
insights = generate_llm_insights(payload)
print(insights)

# Answer follow-up questions
answer = answer_followup_question(
    payload, 
    "Which categories did I spend the most on?"
)
print(answer)
```

---

## 📁 Project Structure

```
AI_Intern_Assignment/
│
├── data/                              # Data directory
│   ├── synthetic_full_dataset.csv     # Raw transaction data
│   ├── cleaned_synthetic_dataset.csv  # Preprocessed data
│   ├── daily_features.csv             # Daily aggregated features
│   ├── monthly_features.csv           # Monthly aggregated features
│   ├── user_month_patterns.csv        # Identified patterns
│   ├── user_month_signals.csv         # Rule-based + statistical signals
│   ├── user_personas.csv              # Clustering results
│   ├── user_latest_predictions.csv    # Prediction outputs
│   └── transaction_anomalies.csv      # Anomaly detection results
│
├── data_preprocessing.py              # Data cleaning and normalization
├── feature_engineering_daily.py        # Daily feature generation
├── feature_engineering_monthly.py     # Monthly feature generation
├── pattern_identification.py          # Pattern extraction
├── flag_signals_rsc.py                # Rule-based + statistical signals
│
├── prediction_engine.py               # XGBoost prediction pipeline
├── clustering_engine.py               # KMeans clustering pipeline
├── anomaly_detection_engine.py        # Anomaly detection (Z-score + IF)
│
├── insight_engine.py                  # Core logic: assembles JSON payload
├── llm_integration.py                 # Gemini API integration
│
├── streamlit_app.py                   # Streamlit web UI
├── config.py                          # Central configuration
├── requirements.txt                    # Python dependencies
├── .env                               # Environment variables (create this)
└── README.md                          # This file
```

---

## 📚 Module Documentation

### Data Processing Modules

#### `data_preprocessing.py`
Cleans and normalizes raw transaction data:
- Standardizes column names and data types
- Validates transaction amounts (income positive, expenses negative)
- Normalizes text fields (categories, merchants, payment modes)
- Extracts date features (day of week, weekend flags)
- Removes duplicates and invalid rows
- **Output**: `data/cleaned_synthetic_dataset.csv`

#### `feature_engineering_daily.py`
Generates daily-level aggregated features:
- Daily spending totals
- Category-wise daily breakdowns
- Payment mode distributions
- Temporal patterns (weekday/weekend)
- **Output**: `data/daily_features.csv`

#### `feature_engineering_monthly.py`
Creates monthly aggregated features:
- Monthly income and expense totals
- Savings rate calculations
- Category spending shares
- Recurring expense identification
- Merchant dependency metrics
- **Output**: `data/monthly_features.csv`

#### `pattern_identification.py`
Identifies behavioral patterns:
- Spending spikes and anomalies
- Category overspending
- Payment mode preferences
- Weekend vs. weekday patterns
- Recurring expense patterns
- **Output**: `data/user_month_patterns.csv`

### Analytics Engines

#### `flag_signals_rsc.py`
Generates rule-based and statistical signals:
- **Rule-Based Flags**: Low savings, high volatility, merchant dependency, etc.
- **Statistical Signals**: Z-score based unusual spending/savings detection
- **Comparative Signals**: Month-over-month changes
- **Output**: `data/user_month_signals.csv`

#### `anomaly_detection_engine.py`
Detects anomalous transactions:
- **Z-Score Method**: Flags transactions with |z-score| > 3
- **Isolation Forest**: Multivariate anomaly detection (3% contamination)
- **Final Decision**: Combines both methods (union of flags)
- **Output**: `data/transaction_anomalies.csv`

#### `prediction_engine.py`
XGBoost-based prediction models:
- **Spending Prediction**: Regression model for next-month expenses
- **Savings Rate Prediction**: Forecasts future savings behavior
- **Overspend Risk**: Binary classifier (spending > income)
- **Features**: Lag-1, lag-3, lag-6 of key metrics
- **Output**: `data/user_latest_predictions.csv`

#### `clustering_engine.py`
KMeans clustering for user personas:
- **Aggregation**: User-level features from monthly patterns (mean, median, std, max)
- **Clustering**: KMeans with k=3 clusters
- **Features**: Spending, savings rate, income volatility, spike intensity, payment preferences
- **Output**: `data/user_personas.csv`

### Core Integration Modules

#### `insight_engine.py`
The central orchestrator that:
- Loads all processed CSV files
- Builds structured JSON payload for LLM
- Computes financial health metrics
- Generates insight candidates with priorities
- Combines outputs from all engines
- **Key Function**: `build_analytics_payload(user_id, year_month, data_dir)`

**Payload Structure:**
```json
{
  "user_profile": {...},
  "period": {...},
  "summary": {
    "income": {...},
    "spending": {...},
    "by_category": [...],
    "temporal": {...},
    "payment_modes": {...},
    "recurring": {...}
  },
  "comparative": {...},
  "clustering": {...},
  "predictions": {...},
  "anomalies": {...},
  "flags": {...},
  "financial_health": {...},
  "insight_candidates": [...]
}
```

#### `llm_integration.py`
Google Gemini API integration:
- **`generate_llm_insights(payload)`**: Generates prioritized insights using mini-agent approach
- **`generate_llm_insights_basic(payload)`**: Simple workflow-style insights
- **`answer_followup_question(payload, question)`**: Conversational Q&A
- **`run_insights_for_user(...)`**: Convenience wrapper for programmatic access

**Features:**
- Uses `gemini-2.5-flash` model (configurable)
- Strict hallucination prevention mode
- Supports both agentic and workflow styles
- Interactive CLI mode with follow-up Q&A

#### `streamlit_app.py`
Interactive web interface:
- User and month selection
- Real-time insights generation
- Financial metrics visualization
- Follow-up Q&A interface
- Raw payload viewer for debugging

---

## 🔄 Data Pipeline

The system processes data through the following stages:

### Stage 1: Data Ingestion & Preprocessing
```bash
# Run preprocessing
python data_preprocessing.py
```
- Input: `data/synthetic_full_dataset.csv`
- Output: `data/cleaned_synthetic_dataset.csv`

### Stage 2: Feature Engineering
```bash
# Generate daily features
python feature_engineering_daily.py

# Generate monthly features
python feature_engineering_monthly.py
```
- Input: `data/cleaned_synthetic_dataset.csv`
- Outputs: `data/daily_features.csv`, `data/monthly_features.csv`

### Stage 3: Pattern Identification
```bash
python pattern_identification.py
```
- Input: `data/monthly_features.csv`
- Output: `data/user_month_patterns.csv`

### Stage 4: Signal Generation
```bash
python flag_signals_rsc.py
```
- Input: `data/user_month_patterns.csv`
- Output: `data/user_month_signals.csv`

### Stage 5: ML Models
```bash
# Anomaly detection
python anomaly_detection_engine.py

# Predictions
python prediction_engine.py

# Clustering
python clustering_engine.py
```
- Inputs: Various processed CSVs
- Outputs: `data/transaction_anomalies.csv`, `data/user_latest_predictions.csv`, `data/user_personas.csv`

### Stage 6: Insights Generation
```bash
# Via Streamlit UI
streamlit run streamlit_app.py

# Or via CLI
python llm_integration.py
```
- Inputs: All processed CSVs
- Output: AI-generated insights + Q&A

**Note**: For a complete pipeline run, execute stages 1-5 in order. Stage 6 can be run independently once all intermediate files are generated.

---

## 📸 Screenshots

> **Note**: Screenshots will be added here to showcase:
> - Streamlit UI dashboard
> - Sample insights output
> - Follow-up Q&A interface
> - Financial metrics visualization

---

## 🔮 Future Improvements

### Short-Term Enhancements
- [ ] **Real-time Data Ingestion**: Support for live transaction feeds via API
- [ ] **Multi-Currency Support**: Handle transactions in different currencies
- [ ] **Budget Tracking**: Set and monitor budget goals per category
- [ ] [ ] **Export Functionality**: PDF/Excel export of insights and reports
- [ ] **User Authentication**: Multi-user support with secure login

### Medium-Term Enhancements
- [ ] **Advanced Visualizations**: Interactive charts using Plotly/Bokeh
- [ ] **Time Series Forecasting**: ARIMA/Prophet models for long-term predictions
- [ ] **Recommendation Engine**: Personalized financial product recommendations
- [ ] **Mobile App**: React Native or Flutter mobile interface
- [ ] **API Endpoints**: RESTful API for third-party integrations

### Long-Term Enhancements
- [ ] **Federated Learning**: Privacy-preserving model training across users
- [ ] **Blockchain Integration**: Immutable transaction ledger
- [ ] **Voice Interface**: Voice-activated financial queries
- [ ] **Multi-Language Support**: Localized insights in multiple languages
- [ ] **Advanced NLP**: Sentiment analysis of transaction descriptions

### Technical Improvements
- [ ] **Database Integration**: Replace CSV storage with PostgreSQL/MongoDB
- [ ] **Caching Layer**: Redis for faster payload generation
- [ ] **Async Processing**: Celery for background job processing
- [ ] **Model Versioning**: MLflow for model tracking and deployment
- [ ] **Unit Tests**: Comprehensive test coverage with pytest
- [ ] **CI/CD Pipeline**: Automated testing and deployment

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

**Note**: This is an assignment project. Please review licensing requirements before commercial use.

---

## 🙏 Acknowledgments

### Technologies & Libraries
- **Google Gemini API**: For natural language generation and conversational AI
- **XGBoost**: Gradient boosting framework for predictions
- **scikit-learn**: Machine learning utilities (KMeans, Isolation Forest, preprocessing)
- **Streamlit**: Rapid web app development framework
- **pandas & numpy**: Data manipulation and numerical computing
- **python-dotenv**: Environment variable management

### Inspiration
This project was developed as part of an AI Intern assignment, demonstrating end-to-end data science and machine learning capabilities in the personal finance domain.

---

## 📞 Support

For questions, issues, or contributions:
1. Check the documentation in this README
2. Review the code comments in individual modules
3. Open an issue on the repository (if applicable)
4. Contact the project maintainer

---

**Built with ❤️ using Python, XGBoost, and Google Gemini**

