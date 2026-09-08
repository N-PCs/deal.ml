# deal.ml | Agentic RAG & Predictive ML System for Automated M&A Due Diligence & Valuation

> **An Enterprise Financial Machine Learning & RAG Platform for Global Investment Banking Analysts.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-5B46EB?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTUtMTAtNXpNMiAxN2wxMCA1IDEwLTVNMiAxMmwxMCA1IDEwLTUiLz48L3N2Zz4=)
![XGBoost](https://img.shields.io/badge/XGBoost-111111?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMSAydjQuMDdDOC42IDYuNTYgNi43OCA4LjQ0IDYuMjcgMTAuOUwzIDkuNCAxLjUgMTJsMy4yNyAxLjVjLjUxIDIuNDYgMi4zMyA0LjM0IDQuNzMgNC44M1YyMmgzLjV2LTMuMjdjMi40LS40OSA0LjIyLTIuMzcgNC43My00LjgzTDIxIDE1LjRsMS41LTMtMy4yNy0xLjVjLS41MS0yLjQ2LTIuMzMtNC4zNC00LjczLTQuODNWMmgtMy41em0xLjc1IDhhMiAyIDAgMSAxIDAgNCAyIDIgMCAwIDEgMC00eiIvPjwvc3ZnPg==)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

---

## Executive Summary

Investment bankers at top-tier financial institutions evaluate corporate acquisitions, build financial valuation models (DCF and Comps), and execute due diligence on public and private companies. Raw Large Language Models (LLMs) fail at financial modeling because LLMs hallucinate calculations and struggle with tabular accounting statements.

**`deal.ml`** solves this by uniting **Agentic Hybrid RAG** (for text and SEC table extraction) with a **Deterministic Python Mathematical Engine** (for zero-hallucination DCF, WACC, and M&A Accretion/Dilution models) and an **XGBoost Machine Learning Forecaster** (for predicting forward growth and margin trajectory based on peer market data).

---

## System Architecture & Workflow

```mermaid
flowchart TD
    %% Vibrant Neon Styling Definitions
    classDef mintGreen fill:#00FF99,stroke:#00CC77,color:#052e16,font-weight:bold;
    classDef neonYellow fill:#FFE600,stroke:#E6C200,color:#111827,font-weight:bold;
    classDef neonPink fill:#FF007F,stroke:#E60070,color:#FFFFFF,font-weight:bold;
    classDef neonOrange fill:#FF6600,stroke:#E65C00,color:#FFFFFF,font-weight:bold;
    classDef neonCyan fill:#00E5FF,stroke:#00B4D8,color:#0B0F19,font-weight:bold;

    subgraph DataIngestion["📁 1. Multi-Modal Financial Ingestion"]
        SEC["📄 SEC 10-K/10-Q (HTML)"]:::mintGreen
        Transcripts["🎙️ Earnings Call Transcripts"]:::mintGreen
        MarketData["📈 yfinance Fundamental Data"]:::mintGreen
    end

    subgraph ProcessingStorage["⚡ 2. Processing & Storage"]
        TableParser["Parsing: Table-Aware Parser"]:::neonYellow
        ChromaDB[("Vector Database: ChromaDB")]:::neonYellow
        BM25Index[("Sparse Index: BM25")]:::neonYellow
        FeatureMatrix["Feature Matrix: Pandas/NumPy"]:::neonYellow
    end

    subgraph CoreEngine["🧠 3. Intelligence & Math Engine"]
        RAG["Hybrid RAG: RRF + Reranker"]:::neonPink
        MLModel{"ML Engine: XGBoost Forecaster"}:::neonPink
        MathEngine["Math Engine: Pure Python Valuation & Merger"]:::neonPink
    end

    subgraph RESTAPI["🚀 4. API Orchestration"]
        FastAPI(["FastAPI REST Endpoints Services"]):::neonOrange
    end

    subgraph FrontendUI["💻 5. User Interface"]
        StreamlitDesk[["Streamlit Financial Workbench Terminal"]]:::neonCyan
        FootballField["Valuation Football Field Chart"]:::neonCyan
        AccretionSimulator["M&A Accretion/Dilution Engine"]:::neonCyan
        RAGWorkspace["Verified RAG Footnotes Workspace"]:::neonCyan
    end

    %% Workflow Connections
    SEC --> TableParser --> ChromaDB
    Transcripts --> BM25Index
    MarketData --> FeatureMatrix --> MLModel

    ChromaDB & BM25Index --> RAG
    RAG --> FastAPI
    MLModel --> MathEngine
    MathEngine --> FastAPI

    FastAPI --> StreamlitDesk
    StreamlitDesk --> FootballField & AccretionSimulator & RAGWorkspace
```

---

## 1. Core Financial Data Files

| Data Source | Format | Purpose | Extraction Strategy |
| :--- | :--- | :--- | :--- |
| **SEC 10-K / 10-Q Filings** | HTML / Markdown | Item 1A (Risk Factors) & Item 7 (MD&A) qualitative risk analysis | `SECTableAwareParser` isolates `<table>` tags to prevent splitting tables in half |
| **Earnings Call Transcripts** | Text / Markdown | Management sentiment and Wall Street analyst Q&A | Text chunking (~800 chars) with overlap and metadata indexing |
| **Historical Financial Statements** | Tabular CSV / `yfinance` | Balance Sheets, Income Statements, and Cash Flow Statements | Programmatic Extraction via `MarketDataPipeline` |
| **Market Valuation Multiples** | Tabular DataFrame | Peer P/E, EV/EBITDA, 52-Week High/Low trading ranges | Normalized feature vectors for XGBoost forecasting |

---

## 2. Technology Stack

- **Language**: 100% Python 3.10 (Zero R code)
- **Mathematical Execution**: Pure Python NumPy / Pandas (Deterministic, Audit-proof)
- **Machine Learning**: XGBoost, Scikit-Learn (Predictive metrics & parameter forecasting)
- **Agentic RAG & Vector Store**: ChromaDB, BM25 (`rank_bm25`), Reciprocal Rank Fusion (RRF)
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **Frontend Dashboard**: Streamlit, Plotly (Financial Football Fields & Pro-Forma Charts)
- **DevOps & Deployment**: Docker, Docker Compose, MLflow

---

## 3. Quick Start & Execution

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (Optional for containerized run)

### Running Locally with Python
```bash
# 1. Clone/Navigate to workspace
cd deal.ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch FastAPI Backend (Terminal 1)
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Launch Streamlit Financial Terminal (Terminal 2)
streamlit run frontend/app.py
```

### Running with Docker Compose (Modern Docker CLI v2)
```bash
docker compose up --build
```

*(Note: If using older legacy standalone docker-compose, use `docker-compose up --build`)*

Access the application at:
- **Streamlit Terminal**: `http://localhost:8501`
- **FastAPI API Docs**: `http://localhost:8000/docs`

---

 
