"""
Deterministic Financial Mathematics Engine for Investment Banking.
Handles DCF Valuation, WACC, Comps Pricing, and M&A Accretion/Dilution Analysis.
"""

from typing import List, Dict, Any, Optional

class InvestmentBankingMathEngine:
    @staticmethod
    def calculate_wacc(
        cost_of_equity: float,
        cost_of_debt: float,
        equity_value: float,
        debt_value: float,
        tax_rate: float
    ) -> float:
        """
        Calculates Weighted Average Cost of Capital (WACC).
        WACC = (E/V * Ke) + (D/V * Kd * (1 - t))
        """
        total_value = equity_value + debt_value
        if total_value == 0:
            raise ValueError("Total Capital Value (Equity + Debt) cannot be zero.")
            
        weight_equity = equity_value / total_value
        weight_debt = debt_value / total_value
        
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
        return round(wacc, 4)

    @staticmethod
    def calculate_dcf_valuation(
        free_cash_flows: List[float],
        wacc: float,
        terminal_growth_rate: float,
        net_debt: float = 0.0,
        shares_outstanding: float = 1.0,
        exit_multiple: Optional[float] = None,
        last_ev_ebitda: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates Enterprise Value (EV) and Implied Share Price using Discounted Cash Flow.
        Supports both Gordon Growth Model and Exit Multiple Method for Terminal Value.
        """
        if wacc <= terminal_growth_rate:
            raise ValueError("WACC must be strictly greater than terminal growth rate.")

        n_years = len(free_cash_flows)
        discount_factors = [(1 + wacc) ** i for i in range(1, n_years + 1)]
        pv_fcf = [fcf / df for fcf, df in zip(free_cash_flows, discount_factors)]
        pv_fcf_sum = sum(pv_fcf)

        # Terminal Value - Gordon Growth Model
        last_fcf = free_cash_flows[-1]
        terminal_value_gordon = (last_fcf * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)
        pv_terminal_gordon = terminal_value_gordon / discount_factors[-1]

        enterprise_value_gordon = pv_fcf_sum + pv_terminal_gordon
        equity_value_gordon = enterprise_value_gordon - net_debt
        implied_share_price_gordon = equity_value_gordon / shares_outstanding if shares_outstanding > 0 else 0.0

        # Optional Exit Multiple Method
        enterprise_value_exit = None
        implied_share_price_exit = None
        if exit_multiple is not None and last_ev_ebitda is not None:
            terminal_value_exit = last_ev_ebitda * exit_multiple
            pv_terminal_exit = terminal_value_exit / discount_factors[-1]
            enterprise_value_exit = pv_fcf_sum + pv_terminal_exit
            equity_value_exit = enterprise_value_exit - net_debt
            implied_share_price_exit = equity_value_exit / shares_outstanding if shares_outstanding > 0 else 0.0

        return {
            "pv_fcf_sum": round(pv_fcf_sum, 2),
            "gordon_growth": {
                "terminal_value": round(terminal_value_gordon, 2),
                "pv_terminal_value": round(pv_terminal_gordon, 2),
                "enterprise_value": round(enterprise_value_gordon, 2),
                "equity_value": round(equity_value_gordon, 2),
                "implied_share_price": round(implied_share_price_gordon, 2)
            },
            "exit_multiple": {
                "enterprise_value": round(enterprise_value_exit, 2) if enterprise_value_exit else None,
                "implied_share_price": round(implied_share_price_exit, 2) if implied_share_price_exit else None
            }
        }

    @staticmethod
    def calculate_comps_valuation(
        peer_ev_ebitda_multiples: List[float],
        peer_pe_multiples: List[float],
        target_ebitda: float,
        target_net_income: float,
        target_net_debt: float,
        target_shares: float
    ) -> Dict[str, Any]:
        """
        Calculates Target Valuation using Comparable Company Analysis (Comps).
        """
        median_ev_ebitda = float(sorted(peer_ev_ebitda_multiples)[len(peer_ev_ebitda_multiples) // 2])
        median_pe = float(sorted(peer_pe_multiples)[len(peer_pe_multiples) // 2])

        implied_ev = target_ebitda * median_ev_ebitda
        implied_equity_from_ev = implied_ev - target_net_debt
        implied_price_ev = implied_equity_from_ev / target_shares if target_shares > 0 else 0.0

        implied_equity_pe = target_net_income * median_pe
        implied_price_pe = implied_equity_pe / target_shares if target_shares > 0 else 0.0

        return {
            "peer_median_ev_ebitda": round(median_ev_ebitda, 2),
            "peer_median_pe": round(median_pe, 2),
            "comps_implied_ev": round(implied_ev, 2),
            "implied_share_price_ev_ebitda": round(implied_price_ev, 2),
            "implied_share_price_pe": round(implied_price_pe, 2)
        }

    @staticmethod
    def simulate_merger_consequences(
        acq_net_income: float,
        tgt_net_income: float,
        acq_shares: float,
        tgt_shares: float,
        acq_share_price: float,
        tgt_share_price: float,
        offer_premium_pct: float,
        pre_tax_synergies: float,
        cash_pct: float = 0.5,
        stock_pct: float = 0.5,
        debt_interest_rate: float = 0.05,
        tax_rate: float = 0.25
    ) -> Dict[str, Any]:
        """
        Executes Merger Consequences Analysis (Accretion / Dilution).
        Evaluates offer value, financing mix (Cash vs Debt vs Stock), synergy impact, and Pro-Forma EPS.
        """
        offer_price_per_share = tgt_share_price * (1 + offer_premium_pct)
        total_deal_value = offer_price_per_share * tgt_shares

        cash_needed = total_deal_value * cash_pct
        stock_value_issued = total_deal_value * stock_pct

        # New shares issued by acquirer for stock consideration
        new_shares_issued = stock_value_issued / acq_share_price if acq_share_price > 0 else 0.0
        pro_forma_shares = acq_shares + new_shares_issued

        # Interest expense from debt component (assuming cash portion is funded by debt if cash_pct > 0)
        annual_interest_expense = cash_needed * debt_interest_rate
        after_tax_interest_expense = annual_interest_expense * (1 - tax_rate)

        # Synergies net of tax
        after_tax_synergies = pre_tax_synergies * (1 - tax_rate)

        # Combined Pro-Forma Net Income
        pro_forma_net_income = (
            acq_net_income + 
            tgt_net_income + 
            after_tax_synergies - 
            after_tax_interest_expense
        )

        standalone_eps = acq_net_income / acq_shares if acq_shares > 0 else 0.0
        pro_forma_eps = pro_forma_net_income / pro_forma_shares if pro_forma_shares > 0 else 0.0

        eps_change = pro_forma_eps - standalone_eps
        eps_change_pct = (eps_change / standalone_eps) * 100 if standalone_eps > 0 else 0.0
        status = "Accretive" if eps_change > 0 else "Dilutive"

        return {
            "offer_price_per_share": round(offer_price_per_share, 2),
            "total_deal_value": round(total_deal_value, 2),
            "new_shares_issued": round(new_shares_issued, 2),
            "pro_forma_shares": round(pro_forma_shares, 2),
            "standalone_eps": round(standalone_eps, 2),
            "pro_forma_eps": round(pro_forma_eps, 2),
            "eps_change_dollars": round(eps_change, 2),
            "eps_change_pct": round(eps_change_pct, 2),
            "status": status
        }
