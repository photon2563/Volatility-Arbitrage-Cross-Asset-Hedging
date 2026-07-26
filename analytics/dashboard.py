"""
Explainability Framework & Transparency Dashboard.

Implements the Trade Justification Matrix to convert complex state-space mathematics into
human-readable, diagnostic trading logs in real-time, answering:
1. What is the precise directional edge? (The Signal State)
2. How is the inherent market risk being mitigated? (The Hedge State)
3. What is the mathematical confidence in the current model? (The Uncertainty Metric)

Also formats executive comparison tables for institutional presentation.
"""

from typing import List, Dict, Any
from tabulate import tabulate
from analytics.metrics import PerformanceMetrics
from engine.signals import TradeSignal
from engine.kalman import KalmanState


class TransparencyDashboard:
    """
    Generates real-time diagnostic logs and formatted performance comparison tables.
    """
    
    @staticmethod
    def generate_trade_justification_log(
        signal: TradeSignal,
        kalman: KalmanState,
        is_scaled: bool
    ) -> str:
        """
        Outputs a comprehensive 3-part diagnostic log entry for a specific trading step.
        """
        log_lines = [
            f"=== [TRADE JUSTIFICATION MATRIX | DATE: {signal.date}] ===",
            f"1. DIRECTIONAL EDGE (The Signal State):",
            f"   {signal.justification}",
            f"",
            f"2. RISK MITIGATION (The Hedge State):",
            f"   A directional position in VIX futures introduces substantial equity beta exposure. To neutralize this,",
            f"   an offsetting position in S&P 500 E-mini (ES) futures is algorithmically maintained. The Kalman filter",
            f"   processed today's market observations and updated the dynamic hedge ratio (beta_t|t) to {kalman.beta_post:.4f}.",
            f"   Required delta-neutral offset: {kalman.es_contracts_req:.2f} ES contracts per 100 VX contracts.",
            f"",
            f"3. MODEL CONFIDENCE & UNCERTAINTY DIAGNOSTICS:",
            f"   State covariance matrix (P_t|t): {kalman.p_post:.6f} | Measurement innovation (e_t): {kalman.innovation:.4f}",
            f"   Innovation standard deviation: {np.sqrt(kalman.innovation_var):.4f} | Innovation Z-Score: {kalman.z_score:.2f} std devs."
        ]
        
        if is_scaled:
            log_lines.extend([
                f"   🚨 CRITICAL WARNING: Measurement innovation exceeded 3.5 standard deviations ({kalman.z_score:.2f}σ)!",
                f"   Historical covariance between VIX and ES is experiencing an acute structural break. In response,",
                f"   the Kalman Gain (K_t = {kalman.kalman_gain:.4f}) has aggressively adapted the state estimate.",
                f"   ALGORITHMIC ACTION: Position sizing automatically reduced by 50% to protect portfolio capital until stability returns."
            ])
        else:
            log_lines.append(
                f"   Status: Relationship stable. Measurement innovations fall within normal distributions. Execution authorized at 100% sizing."
            )
            
        log_lines.append("=" * 65)
        return "\n".join(log_lines)

    @staticmethod
    def format_metrics_table(metrics_list: List[PerformanceMetrics]) -> str:
        """
        Formats a clean, publication-ready ASCII/Markdown table of comparative metrics.
        """
        headers = [
            "Metric Category",
            "Quantitative Metric",
            metrics_list[0].name,  # Dynamic Kalman
            metrics_list[1].name,  # Static OLS
            metrics_list[2].name,  # Unhedged
            metrics_list[3].name   # S&P 500 Benchmark
        ]
        
        rows = [
            ["Return Profiles", "CAGR (%)", f"{metrics_list[0].cagr_pct:.2f}%", f"{metrics_list[1].cagr_pct:.2f}%", f"{metrics_list[2].cagr_pct:.2f}%", f"{metrics_list[3].cagr_pct:.2f}%"],
            ["Return Profiles", "Sharpe Ratio", f"{metrics_list[0].sharpe_ratio:.2f}", f"{metrics_list[1].sharpe_ratio:.2f}", f"{metrics_list[2].sharpe_ratio:.2f}", f"{metrics_list[3].sharpe_ratio:.2f}"],
            ["Tail-Risk Profiles", "Sortino Ratio (Downside)", f"{metrics_list[0].sortino_ratio:.2f}", f"{metrics_list[1].sortino_ratio:.2f}", f"{metrics_list[2].sortino_ratio:.2f}", f"{metrics_list[3].sortino_ratio:.2f}"],
            ["Tail-Risk Profiles", "Max Drawdown (MDD %)", f"-{metrics_list[0].max_drawdown_pct:.2f}%", f"-{metrics_list[1].max_drawdown_pct:.2f}%", f"-{metrics_list[2].max_drawdown_pct:.2f}%", f"-{metrics_list[3].max_drawdown_pct:.2f}%"],
            ["Tail-Risk Profiles", "Calmar Ratio", f"{metrics_list[0].calmar_ratio:.2f}", f"{metrics_list[1].calmar_ratio:.2f}", f"{metrics_list[2].calmar_ratio:.2f}", f"{metrics_list[3].calmar_ratio:.2f}"],
            ["Tail-Risk Profiles", "CVaR 99% (Expected Shortfall)", f"-{metrics_list[0].cvar_99_pct:.2f}%", f"-{metrics_list[1].cvar_99_pct:.2f}%", f"-{metrics_list[2].cvar_99_pct:.2f}%", f"-{metrics_list[3].cvar_99_pct:.2f}%"],
            ["Execution Stats", "Hit Rate (%)", f"{metrics_list[0].hit_rate_pct:.1f}%", f"{metrics_list[1].hit_rate_pct:.1f}%", f"{metrics_list[2].hit_rate_pct:.1f}%", "N/A (Hold)"],
            ["Execution Stats", "Win / Loss Ratio", f"{metrics_list[0].win_loss_ratio:.2f}", f"{metrics_list[1].win_loss_ratio:.2f}", f"{metrics_list[2].win_loss_ratio:.2f}", "N/A (Hold)"],
            ["Execution Stats", "Profit Factor", f"{metrics_list[0].profit_factor:.2f}", f"{metrics_list[1].profit_factor:.2f}", f"{metrics_list[2].profit_factor:.2f}", "N/A (Hold)"],
            ["Execution Stats", "S&P 500 Correlation Beta", f"{metrics_list[0].spx_beta:.2f}", f"{metrics_list[1].spx_beta:.2f}", f"{metrics_list[2].spx_beta:.2f}", "1.00 (Ref)"],
            ["System Diagnostics", "Algorithmic Risk Scaling Events", str(metrics_list[0].risk_scaling_events), "0 (Static)", "0 (Naked)", "0 (Hold)"],
            ["System Diagnostics", "Total Frictions Paid ($)", f"${metrics_list[0].total_frictions:,.0f}", f"${metrics_list[1].total_frictions:,.0f}", f"${metrics_list[2].total_frictions:,.0f}", f"${metrics_list[3].total_frictions:,.0f}"]
        ]
        
        return tabulate(rows, headers=headers, tablefmt="github")

import numpy as np # import needed for np.sqrt in docstring/eval above
