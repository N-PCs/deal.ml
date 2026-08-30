"""
Market Data Ingestion Pipeline.
Pulls financial statements, historical valuation multiples, and stock price histories via yfinance.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataPipeline:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)

    def fetch_company_info(self) -> Dict[str, Any]:
        """Fetch general company metadata and valuation metrics."""
        info = self.stock.info
        return {
            "ticker": self.ticker,
            "company_name": info.get("longName", self.ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0.0)),
            "pe_ratio": info.get("trailingPE", 0.0),
            "forward_pe": info.get("forwardPE", 0.0),
            "ev_ebitda": info.get("enterpriseToEbitda", 0.0),
            "enterprise_value": info.get("enterpriseValue", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            "52_week_high": info.get("fiftyTwoWeekHigh", 0.0),
            "52_week_low": info.get("fiftyTwoWeekLow", 0.0)
        }

    def fetch_financial_statements(()) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch balance sheet, income statement, and cash flow statement."""
        income_stmt = self.stock.financials
        balance_sheet = self.stock.balance_sheet
        cash_flow = self.stock.cashflow

        return income_stmt, balance_sheet, cash_flow

    def extract_dcf_inputs(()) -> Dict[str, Any]:
        """Extract baseline cash flows, debt, and cash balances for DCF calculation."""
        info = self.fetch_company_info()
        cash_flow = self.stock.cashflow
        balance_sheet = self.stock.balance_sheet

        # Free Cash Flow approximation: Operating Cash Flow - Capital Expenditures
        fcf_series = []
        try:
            free_cf = cash_flow.loc['Free Cash Flow']
            fcf_series = [float(val) for val in free_cf.values[:4]][::-1] # 4 years ascending
        except KeyError:
            try:
                op_cf = cash_flow.loc['Operating Cash Flow']
                cap_ex = cash_flow.loc['Capital Expenditure']
                fcf_calculated = op_cf + cap_ex # cap_ex is usually negative
                fcf_series = [float(val) for val in fcf_calculated.values[:4]][::-1]
            except Exception as e:
                logger.warning(f"Could not automatically compute FCF for {self.ticker}: {e}")
                fcf_series = [1e9, 1.1e9, 1.25e9, 1.4e9] # Fallback estimation baseline

        # Total Debt & Cash
        total_debt = info.get("totalDebt", 0.0)
        total_cash = info.get("totalCash", 0.0)
        net_debt = total_debt - total_cash

        return {
            "ticker": self.ticker,
            "historical_fcf": fcf_series,
            "net_debt": net_debt,
            "shares_outstanding": info.get("sharesOutstanding", 1.0),
            "current_price": info.get("currentPrice", 0.0)
        }

    def generate_ml_features((self, years: str = "5y") -> pd.DataFrame:
        """Create structured tabular dataset for ML predictive modeling."""
        history = self.stock.history(period=years)
        if history.empty:
            return pd.DataFrame()

        df = pd.DataFrame()
        df['Close'] = history['Close']
        df['Volume'] = history['Volume']
        df['Daily_Return'] = history['Close'].pct_change()
        df['Volatility_30D'] = df['Daily_Return'].rolling(30).std()
        df['MA_50'] = history['Close'].rolling(50).mean()
        df['MA_200'] = history['Close'].rolling(200).mean()

        return df.dropna()
