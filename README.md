# 📊 CSE Market Intelligence 

An AI-powered equity research platform focused on the **Colombo Stock Exchange (CSE)**. This project leverages **Retrieval-Augmented Generation (RAG)** to analyze stock reports, company announcements, and portfolio risk.

## ✨ Key Features

-   **🔍 Stock Research**: Real-time market data integration with Yahoo Finance for full price history and performance metrics.
-   **💼 Portfolio Intelligence**: 
    -   Automated portfolio risk & concentration monitoring.
    -   Smart CSV parsing (with or without headers).
    -   AI-generated portfolio summaries and reviews.
-   **📑 Report & Announcement Intelligence**:
    -   Summarize and extract key financial facts from PDF reports.
    -   Track and analyze latest CSE disclosures using NLP.
    -   Structured event extraction (Materiality, Event Type, Positives/Risks).
-   **📈 Stock Screener**: Advanced filtering to identify market opportunities.
-   **🔔 Alerts & Monitoring**: Real-time tracking of market events and price movements.
-   **📝 AI Research Memos**: Auto-generate deep research memos combining market view and latest disclosures.

## 🛠️ Technology Stack

-   **Language**: Python 3.10+
-   **Web Framework**: [Streamlit](https://streamlit.io/)
-   **LLM Orchestration**: [LangChain](https://www.langchain.com/)
-   **local LLM Support**: [Ollama](https://ollama.ai/) (Llama3)
-   **Market Data**: `yfinance` & `requests` (CSE API)
-   **PDF Processing**: `PyMuPDF` / `pdfplumber`
-   **Data Analysis**: `pandas`, `plotly`

## 🚀 Getting Started

### 1. Prerequisites
-   Install [Ollama](https://ollama.ai/) and pull the required model (e.g., `llama3`).
-   Verify you have Python installed.

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/chirana07/cse-market-intelligence.git
cd cse-market-intelligence
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

## 📂 Project Structure

-   `src/`: Core logic, including price clients, RAG chains, and AI signal extractors.
-   `src/views/`: Individual Streamlit pages (Dashboard, Research, Portfolio, etc.).
-   `data/`: Configuration files and stock universes.
-   `app.py`: Main entry point for the Streamlit dashboard.

---

*This project was developed for the Sri Lankan Equity Market.*
