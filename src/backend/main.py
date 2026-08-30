"""
FastAPI Backend Server for deal.ml platform.
Exposes endpoints for Financial Math Models, ML Predictions, and Agentic RAG Due Diligence queries.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.ml_engine.valuation_math import InvestmentBankingMathEngine
from src.ml_engine.metrics_forecaster import FinancialMetricsForecaster
from src.data_pipeline.market_data import MarketDataPipeline
from src.rag_engine.hybrid_retriever import FinancialHybridRetriever
from src.data_pipeline.sec_ingestion import SECTableAwareParser

app = FastAPI(
    title="deal.ml - Investment Banking AI & ML Platform",
    description="Automated M&A Due Diligence, DCF Valuation, and Merger Consequences Engine",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend or web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global persistent instances
forecaster = FinancialMetricsForecaster()
forecaster.train_model()
retriever = FinancialHybridRetriever()

# --- Request / Response Models ---

class DCFRequest(BaseModel):
    free_cash_flows: List[float] = Field(..., example=[1000.0, 1150.0, 1300.0, 1450.0])
    wacc: float = Field(0.085, example=0.085)
    terminal_growth_rate: float = Field(0.025, example=0.025)
    net_debt: float = Field(500.0, example=500.0)
    shares_outstanding: float = Field(100.0, example=100.0)
    exit_multiple: Optional[float] = Field(None, example=12.5)
    last_ev_ebitda: Optional[float] = Field(None, example=2000.0)

class MergerSimulationRequest(BaseModel):
    acq_net_income: float = Field(..., example=5000.0)
    tgt_net_income: float = Field(..., example=1200.0)
    acq_shares: float = Field(..., example=500.0)
    tgt_shares: float = Field(..., example=150.0)
    acq_share_price: float = Field(..., example=120.0)
    tgt_share_price: float = Field(..., example=45.0)
    offer_premium_pct: float = Field(0.25, example=0.25)
    pre_tax_synergies: float = Field(300.0, example=300.0)
    cash_pct: float = Field(0.5, example=0.5)
    stock_pct: float = Field(0.5, example=0.5)

class RAGQueryRequest(BaseModel):
    query: str = Field(..., example="What are the key operational risk factors reported by the target?")
    top_k: int = Field(5, example=5)

class IngestSampleDataRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")

# --- Routes ---

@app.get("/")
def root():
    return {"message": "deal.ml Investment Banking Backend Active", "status": "healthy"}

@app.post("/api/v1/valuation/dcf")
def run_dcf_valuation(req: DCFRequest):
    try:
        results = InvestmentBankingMathEngine.calculate_dcf_valuation(
            free_cash_flows=req.free_cash_flows,
            wacc=req.wacc,
            terminal_growth_rate=req.terminal_growth_rate,
            net_debt=req.net_debt,
            shares_outstanding=req.shares_outstanding,
            exit_multiple=req.exit_multiple,
            last_ev_ebitda=req.last_ev_ebitda
        )
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/merger/simulate")
def run_merger_simulation(req: MergerSimulationRequest):
    try:
        results = InvestmentBankingMathEngine.simulate_merger_consequences(
            acq_net_income=req.acq_net_income,
            tgt_net_income=req.tgt_net_income,
            acq_shares=req.acq_shares,
            tgt_shares=req.tgt_shares,
            acq_share_price=req.acq_share_price,
            tgt_share_price=req.tgt_share_price,
            offer_premium_pct=req.offer_premium_pct,
            pre_tax_synergies=req.pre_tax_synergies,
            cash_pct=req.cash_pct,
            stock_pct=req.stock_pct
        )
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/ticker/fetch/{ticker}")
def fetch_ticker_data(ticker: str):
    try:
        pipeline = MarketDataPipeline(ticker)
        info = pipeline.fetch_company_info()
        dcf_inputs = pipeline.extract_dcf_inputs()
        return {"status": "success", "info": info, "dcf_inputs": dcf_inputs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch ticker {ticker}: {str(e)}")

@app.post("/api/v1/rag/query")
def query_financial_rag(req: RAGQueryRequest):
    try:
        results = retriever.hybrid_search(query=req.query, top_k=req.top_k)
        
        # Build synthesis response
        if not results:
            synthesis = "No indexed financial documents found in RAG store. Please ingest SEC 10-K or transcripts first."
        else:
            top_sources = [f"Source: {res['metadata'].get('source', 'Document')} (Score: {res['score']})" for res in results[:3]]
            synthesis = f"Extracted Relevant Context from SEC Filings:\n" + "\n\n".join([r['content'] for r in results[:2]])

        return {
            "status": "success",
            "synthesis": synthesis,
            "citations": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/ml/predict")
def predict_metrics(hist_growth: float = 0.08, volatility: float = 0.20, debt_to_equity: float = 0.8, op_margin: float = 0.25):
    try:
        predictions = forecaster.predict_forward_metrics(hist_growth, volatility, debt_to_equity, op_margin)
        return {"status": "success", "data": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
