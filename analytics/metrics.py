"""
Quantitative Evaluation Metrics Suite.

Implements rigorous, objective performance evaluation categorized into:
1. Return Profiles: Compound Annual Growth Rate (CAGR), Sharpe Ratio.
2. Downside & Tail-Risk Profiles:
   - Sortino Ratio (utilizing downside semivariance in denominator for asymmetric short-vol returns).
   - Maximum Drawdown (MDD) & Calmar Ratio.
   - Conditional Value at Risk (CVaR 99% / Expected Shortfall) to uncover hidden left-tail risks.
3. Execution & Trade Statistics: Hit Rate, Average Win / Average Loss Ratio, Profit Factor, and S&P 500 Beta.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

from backtest.backtester import StrategyResult, TradeRecord


@dataclass
class PerformanceMetrics:
    name: str
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    cvar_99_pct: float
    hit_rate_pct: float
    win_loss_ratio: float
    profit_factor: float
    spx_beta: float
    total_trades: int
    risk_scaling_events: int
    total_frictions: float


def evaluate_strategy(
    result: StrategyResult,
    rf_rate_series: np.ndarray,
    spx_returns: np.ndarray,
    trading_days_per_year: float = 252.0
) -> PerformanceMetrics:
    """
    Computes the full suite of quantitative evaluation metrics for a simulated strategy.
    """
    equity = result.equity_curve
    ret = result.daily_returns
    n = len(ret)
    years = max(n / trading_days_per_year, 0.1)
    
    # 1. Compound Annual Growth Rate (CAGR)
    tot_return = (equity[-1] / equity[0]) if equity[0] != 0 else 1.0
    cagr = (tot_return ** (1.0 / years) - 1.0) * 100.0 if tot_return > 0 else -100.0
    
    # Daily risk-free rate
    rf_daily = rf_rate_series / trading_days_per_year
    excess_ret = ret - rf_daily
    mean_excess = np.mean(excess_ret)
    std_ret = np.std(ret, ddof=1)
    
    # 2. Sharpe Ratio
    sharpe = (mean_excess / std_ret) * np.sqrt(trading_days_per_year) if std_ret > 0 else 0.0
    
    # 3. Sortino Ratio (Downside Semivariance)
    # Penalizes only negative excess returns (downside volatility)
    downside_diff = np.minimum(0.0, excess_ret)
    downside_semivar = np.sqrt(np.mean(downside_diff ** 2))
    sortino = (mean_excess / downside_semivar) * np.sqrt(trading_days_per_year) if downside_semivar > 0 else 0.0
    
    # 4. Maximum Drawdown (MDD)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    mdd = abs(np.min(drawdown)) * 100.0
    
    # 5. Calmar Ratio
    calmar = cagr / mdd if mdd > 0 else 0.0
    
    # 6. Conditional Value at Risk (CVaR 99% / Expected Shortfall)
    # Probability-weighted average of losses beyond the 99th percentile VaR threshold
    sorted_ret = np.sort(ret)
    idx_99 = max(int(np.floor(0.01 * n)), 1)
    tail_losses = sorted_ret[:idx_99]
    cvar_99 = abs(np.mean(tail_losses)) * 100.0 if len(tail_losses) > 0 else 0.0
    
    # 7. Trade Execution Statistics
    trades = result.trades
    total_trades = len(trades)
    if total_trades > 0:
        winning_trades = [t.net_pnl for t in trades if t.net_pnl > 0]
        losing_trades = [t.net_pnl for t in trades if t.net_pnl < 0]
        
        hit_rate = (len(winning_trades) / total_trades) * 100.0
        
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
        avg_loss = abs(np.mean(losing_trades)) if len(losing_trades) > 0 else 0.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else (avg_win if avg_win > 0 else 0.0)
        
        gross_win = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
        gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
        profit_factor = gross_win / gross_loss if gross_loss > 0 else (99.9 if gross_win > 0 else 0.0)
    else:
        hit_rate = 0.0
        win_loss_ratio = 0.0
        profit_factor = 0.0
        
    # 8. Beta to S&P 500 Benchmark
    cov_spx = np.cov(ret, spx_returns)[0, 1]
    var_spx = np.var(spx_returns, ddof=1)
    spx_beta = cov_spx / var_spx if var_spx > 0 else 0.0
    
    return PerformanceMetrics(
        name=result.name,
        cagr_pct=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=mdd,
        calmar_ratio=calmar,
        cvar_99_pct=cvar_99,
        hit_rate_pct=hit_rate,
        win_loss_ratio=win_loss_ratio,
        profit_factor=profit_factor,
        spx_beta=spx_beta,
        total_trades=total_trades,
        risk_scaling_events=result.risk_scaling_events,
        total_frictions=result.total_frictions_paid
    )
