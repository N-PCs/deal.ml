import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from fastapi.testclient import TestClient
from src.backend.main import app
from src.ml_engine.valuation_math import InvestmentBankingMathEngine
from src.ml_engine.metrics_forecaster import FinancialMetricsForecaster
from src.data_pipeline.sec_ingestion import SECTableAwareParser
from src.data_pipeline.market_data import MarketDataPipeline


def test_wacc_calculation():
    # 80% equity at 10%, 20% debt at 5% with 25% tax rate
    wacc = InvestmentBankingMathEngine.calculate_wacc(
        cost_of_equity=0.10,
        cost_of_debt=0.05,
        equity_value=80.0,
        debt_value=20.0,
        tax_rate=0.25
    )
    assert wacc == 0.0875


def test_dcf_valuation():
    fcfs = [1000.0, 1150.0, 1300.0, 1450.0]
    result = InvestmentBankingMathEngine.calculate_dcf_valuation(
        free_cash_flows=fcfs,
        wacc=0.085,
        terminal_growth_rate=0.025,
        net_debt=500.0,
        shares_outstanding=100.0,
        exit_multiple=12.5,
        last_ev_ebitda=2000.0
    )
    assert "gordon_growth" in result
    assert result["gordon_growth"]["enterprise_value"] > 0
    assert result["gordon_growth"]["implied_share_price"] > 0
    assert "exit_multiple" in result
    assert result["exit_multiple"]["enterprise_value"] > 0
    assert result["exit_multiple"]["implied_share_price"] > 0


def test_comps_valuation():
    result = InvestmentBankingMathEngine.calculate_comps_valuation(
        peer_ev_ebitda_multiples=[10.0, 12.0, 14.0],
        peer_pe_multiples=[18.0, 20.0, 22.0],
        target_ebitda=1500.0,
        target_net_income=800.0,
        target_net_debt=300.0,
        target_shares=100.0
    )
    assert result["peer_median_ev_ebitda"] == 12.0
    assert result["peer_median_pe"] == 20.0
    assert result["implied_share_price_ev_ebitda"] > 0
    assert result["implied_share_price_pe"] > 0


def test_merger_simulation():
    res = InvestmentBankingMathEngine.simulate_merger_consequences(
        acq_net_income=5000.0,
        tgt_net_income=1200.0,
        acq_shares=500.0,
        tgt_shares=150.0,
        acq_share_price=120.0,
        tgt_share_price=45.0,
        offer_premium_pct=0.25,
        pre_tax_synergies=300.0,
        cash_pct=0.5,
        stock_pct=0.5
    )
    assert "pro_forma_eps" in res
    assert "eps_change_pct" in res
    assert res["status"] in ["Accretive", "Dilutive"]


def test_ml_forecaster():
    forecaster = FinancialMetricsForecaster()
    forecaster.train_model()
    preds = forecaster.predict_forward_metrics(
        hist_growth=0.08,
        volatility=0.20,
        debt_to_equity=0.8,
        op_margin=0.25
    )
    assert "predicted_forward_growth" in preds
    assert "predicted_ebitda_margin" in preds
    assert "recommended_terminal_growth" in preds
    assert 0.015 <= preds["recommended_terminal_growth"] <= 0.035


def test_sec_parser():
    sample_html = """
    <html>
      <body>
        <h1>Item 1A. Risk Factors</h1>
        <p>The company faces intense competition in global cloud markets and supply chain risks.</p>
        <table>
          <tr><th>Segment</th><th>Revenue ($M)</th></tr>
          <tr><td>Cloud</td><td>15000</td></tr>
          <tr><td>Hardware</td><td>8000</td></tr>
        </table>
      </body>
    </html>
    """
    chunks = SECTableAwareParser.parse_sec_html(sample_html, "sample.html")
    assert len(chunks) >= 2
    types = [c["chunk_type"] for c in chunks]
    assert "table" in types
    assert "text" in types


def test_market_data_pipeline_methods():
    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "marketCap": 3000000000000,
            "currentPrice": 180.0,
            "totalDebt": 100000000000,
            "totalCash": 50000000000,
            "sharesOutstanding": 15000000000
        }
        mock_instance.cashflow = pd.DataFrame(
            {"2023": [100000.0], "2022": [90000.0], "2021": [80000.0], "2020": [70000.0]},
            index=["Free Cash Flow"]
        )
        mock_ticker.return_value = mock_instance

        pipeline = MarketDataPipeline("AAPL")
        info = pipeline.fetch_company_info()
        assert info["company_name"] == "Apple Inc."
        dcf_inputs = pipeline.extract_dcf_inputs()
        assert dcf_inputs["ticker"] == "AAPL"
        assert len(dcf_inputs["historical_fcf"]) == 4


def test_fastapi_endpoints():
    client = TestClient(app)
    # Root check
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

    # DCF endpoint
    dcf_payload = {
        "free_cash_flows": [1000.0, 1150.0, 1300.0, 1450.0],
        "wacc": 0.085,
        "terminal_growth_rate": 0.025,
        "net_debt": 500.0,
        "shares_outstanding": 100.0
    }
    r = client.post("/api/v1/valuation/dcf", json=dcf_payload)
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    # Merger endpoint
    merger_payload = {
        "acq_net_income": 5000.0,
        "tgt_net_income": 1200.0,
        "acq_shares": 500.0,
        "tgt_shares": 150.0,
        "acq_share_price": 120.0,
        "tgt_share_price": 45.0
    }
    r = client.post("/api/v1/merger/simulate", json=merger_payload)
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    # ML predict endpoint
    r = client.post("/api/v1/ml/predict?hist_growth=0.08&volatility=0.20&debt_to_equity=0.8&op_margin=0.25")
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    # RAG query endpoint
    r = client.post("/api/v1/rag/query", json={"query": "test", "top_k": 2})
    assert r.status_code == 200
    assert r.json()["status"] == "success"
