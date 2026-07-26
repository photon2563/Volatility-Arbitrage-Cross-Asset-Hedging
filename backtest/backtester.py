"""
Institutional Multi-Strategy Backtesting Simulator.

Simulates and compares 4 distinct strategies over the identical historical evaluation period:
1. Dynamic State-Space Strategy (Kalman Hedged VIX Basis w/ Algorithmic Risk Scaling)
2. Statically Hedged Basis Strategy (Rolling window OLS regression baseline)
3. Unhedged Basis Strategy (Naked short/long VIX futures)
4. Buy-and-Hold Market Benchmark (S&P 500 E-mini Index Futures)

Accounts for institutional round-trip commissions, dynamic VIX-scaled slippage, and cash interest.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from data.data_pipeline import MarketDataSeries
from engine.signals import TradeSignal
from engine.kalman import KalmanState


@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    direction: int         # -1 (Short VX), +1 (Long VX)
    vx_pnl: float          # Gross PnL from VIX futures
    es_pnl: float          # Gross PnL from S&P 500 E-mini hedge
    frictions: float       # Total commissions and slippage paid
    net_pnl: float         # Net trade PnL
    is_scaled: bool        # Whether Algorithmic Risk Scaling was active during the trade


@dataclass
class StrategyResult:
    name: str
    dates: pd.DatetimeIndex
    equity_curve: np.ndarray
    daily_returns: np.ndarray
    vx_positions: np.ndarray
    es_positions: np.ndarray
    trades: List[TradeRecord]
    risk_scaling_events: int
    total_frictions_paid: float


class Backtester:
    """
    Executes comparative institutional backtests across all 4 strategy baselines.
    """
    def __init__(
        self,
        initial_capital: float = 5_000_000.0,
        base_vx_contracts: float = 50.0,        # Institutional base position sizing ($1M notional)
        risk_scaling_z_threshold: float = 3.5,  # Innovation z-score threshold to halve size
        risk_scaling_factor: float = 0.5        # Size reduction factor during extreme uncertainty
    ):
        self.initial_capital = initial_capital
        self.base_vx_contracts = base_vx_contracts
        self.z_threshold = risk_scaling_z_threshold
        self.scale_factor = risk_scaling_factor

    def run_all_benchmarks(
        self,
        data: MarketDataSeries,
        signals: List[TradeSignal],
        kalman_states: List[KalmanState],
        ols_history: List[Dict[str, Any]]
    ) -> Dict[str, StrategyResult]:
        """
        Runs the simulation across all 4 benchmarks and returns a structured dictionary of results.
        """
        results = {}
        
        # 1. Dynamic Kalman Hedged Strategy w/ Risk Scaling
        results["Dynamic Kalman (Proposed)"] = self._simulate_strategy(
            data=data,
            signals=signals,
            hedge_contracts_req=[k.es_contracts_req for k in kalman_states],
            z_scores=[k.z_score for k in kalman_states],
            name="Dynamic Kalman (Proposed)",
            enable_risk_scaling=True,
            is_hedged=True
        )
        
        # 2. Statically Hedged OLS Strategy
        results["Static OLS Benchmark"] = self._simulate_strategy(
            data=data,
            signals=signals,
            hedge_contracts_req=[o["es_contracts_req"] for o in ols_history],
            z_scores=[0.0] * len(signals),
            name="Static OLS Benchmark",
            enable_risk_scaling=False,
            is_hedged=True
        )
        
        # 3. Unhedged Basis Strategy (Naked VIX)
        results["Unhedged Basis Strategy"] = self._simulate_strategy(
            data=data,
            signals=signals,
            hedge_contracts_req=[0.0] * len(signals),
            z_scores=[0.0] * len(signals),
            name="Unhedged Basis Strategy",
            enable_risk_scaling=False,
            is_hedged=False
        )
        
        # 4. Buy-and-Hold S&P 500 Benchmark
        results["S&P 500 Buy & Hold"] = self._simulate_buy_and_hold_es(
            data=data,
            name="S&P 500 Buy & Hold"
        )
        
        return results

    def _simulate_strategy(
        self,
        data: MarketDataSeries,
        signals: List[TradeSignal],
        hedge_contracts_req: List[float],
        z_scores: List[float],
        name: str,
        enable_risk_scaling: bool,
        is_hedged: bool
    ) -> StrategyResult:
        n = len(data.dates)
        equity = np.zeros(n)
        equity[0] = self.initial_capital
        daily_ret = np.zeros(n)
        
        vx_pos = np.zeros(n)
        es_pos = np.zeros(n)
        
        trades = []
        scaling_events = 0
        total_frictions = 0.0
        
        # Track open trade state
        in_trade = False
        trade_entry_idx = 0
        trade_dir = 0
        trade_vx_pnl = 0.0
        trade_es_pnl = 0.0
        trade_frictions = 0.0
        trade_scaled = False
        
        for t in range(1, n):
            sig = signals[t].signal
            prev_vx_pos = vx_pos[t-1]
            prev_es_pos = es_pos[t-1]
            
            # --- ALGORITHMIC RISK SCALING ---
            # If previous day's Kalman innovation z-score exceeded 3.5 std dev, halve size today
            is_scaled_today = False
            curr_scale = 1.0
            if enable_risk_scaling and abs(z_scores[t-1]) > self.z_threshold:
                curr_scale = self.scale_factor
                is_scaled_today = True
                scaling_events += 1
                
            # Determine target positions for day t
            target_vx = sig * self.base_vx_contracts * curr_scale
            if is_hedged and sig != 0:
                # Required ES contracts offset is scaled by current sizing factor
                target_es = hedge_contracts_req[t] * self.base_vx_contracts * curr_scale
            else:
                target_es = 0.0
                
            # --- PnL ACCOUNTING FOR DAY t ---
            # 1. Gross trading gains/losses on yesterday's held position
            vx_pnl_t = prev_vx_pos * data.vx_returns[t] * (data.unadj_vx_price[t-1] * 1000.0)
            es_pnl_t = prev_es_pos * data.es_returns[t] * (data.es_price[t-1] * 50.0)
            
            # 2. Calculate execution frictions (Commissions + Dynamic VIX-Scaled Slippage + ES Slippage)
            delta_vx = abs(target_vx - prev_vx_pos)
            delta_es = abs(target_es - prev_es_pos)
            
            comm_cost = (delta_vx + delta_es) * data.commission_per_contract
            # Dynamic slippage applies to VIX futures executions (in index points * $1000 multiplier)
            vx_slip_cost = delta_vx * data.slippage_pts[t] * 1000.0
            # ES slippage is 1 tick ($12.50 per contract)
            es_slip_cost = delta_es * 12.50
            frictions_t = comm_cost + vx_slip_cost + es_slip_cost
            total_frictions += frictions_t
            
            # 3. Cash interest earned on unencumbered capital
            interest_t = equity[t-1] * (data.rf_rate[t] / 252.0)
            
            # 4. Total Equity Update
            net_day_pnl = vx_pnl_t + es_pnl_t - frictions_t + interest_t
            equity[t] = equity[t-1] + net_day_pnl
            daily_ret[t] = net_day_pnl / equity[t-1] if equity[t-1] != 0 else 0.0
            
            # Save actual active position for today
            vx_pos[t] = target_vx
            es_pos[t] = target_es
            
            # --- TRADE LOGGING ---
            if not in_trade and target_vx != 0:
                # Initiate new trade
                in_trade = True
                trade_entry_idx = t
                trade_dir = int(np.sign(target_vx))
                trade_vx_pnl = vx_pnl_t
                trade_es_pnl = es_pnl_t
                trade_frictions = frictions_t
                trade_scaled = is_scaled_today
            elif in_trade:
                # Accumulate PnL on existing trade
                trade_vx_pnl += vx_pnl_t
                trade_es_pnl += es_pnl_t
                trade_frictions += frictions_t
                if is_scaled_today:
                    trade_scaled = True
                
                # Check if trade closed or reversed
                if target_vx == 0 or np.sign(target_vx) != trade_dir:
                    trades.append(TradeRecord(
                        entry_date=str(data.dates[trade_entry_idx])[:10],
                        exit_date=str(data.dates[t])[:10],
                        direction=trade_dir,
                        vx_pnl=trade_vx_pnl,
                        es_pnl=trade_es_pnl,
                        frictions=trade_frictions,
                        net_pnl=trade_vx_pnl + trade_es_pnl - trade_frictions,
                        is_scaled=trade_scaled
                    ))
                    in_trade = False
                    if target_vx != 0:  # Reversal entry
                        in_trade = True
                        trade_entry_idx = t
                        trade_dir = int(np.sign(target_vx))
                        trade_vx_pnl = 0.0
                        trade_es_pnl = 0.0
                        trade_frictions = frictions_t
                        trade_scaled = is_scaled_today

        # Close out any remaining trade at end of backtest
        if in_trade:
            trades.append(TradeRecord(
                entry_date=str(data.dates[trade_entry_idx])[:10],
                exit_date=str(data.dates[n-1])[:10],
                direction=trade_dir,
                vx_pnl=trade_vx_pnl,
                es_pnl=trade_es_pnl,
                frictions=trade_frictions,
                net_pnl=trade_vx_pnl + trade_es_pnl - trade_frictions,
                is_scaled=trade_scaled
            ))
            
        return StrategyResult(
            name=name,
            dates=data.dates,
            equity_curve=equity,
            daily_returns=daily_ret,
            vx_positions=vx_pos,
            es_positions=es_pos,
            trades=trades,
            risk_scaling_events=scaling_events,
            total_frictions_paid=total_frictions
        )

    def _simulate_buy_and_hold_es(
        self,
        data: MarketDataSeries,
        name: str
    ) -> StrategyResult:
        n = len(data.dates)
        equity = np.zeros(n)
        equity[0] = self.initial_capital
        daily_ret = np.zeros(n)
        
        # Calculate constant notional ES contracts equivalent to initial capital
        es_contracts_notional = self.initial_capital / (data.es_price[0] * 50.0)
        es_pos = np.ones(n) * es_contracts_notional
        vx_pos = np.zeros(n)
        
        total_frictions = es_contracts_notional * data.commission_per_contract  # initial buy commission
        equity[0] -= total_frictions
        
        for t in range(1, n):
            es_pnl_t = es_pos[t-1] * data.es_returns[t] * (data.es_price[t-1] * 50.0)
            equity[t] = equity[t-1] + es_pnl_t
            daily_ret[t] = es_pnl_t / equity[t-1] if equity[t-1] != 0 else 0.0
            
        return StrategyResult(
            name=name,
            dates=data.dates,
            equity_curve=equity,
            daily_returns=daily_ret,
            vx_positions=vx_pos,
            es_positions=es_pos,
            trades=[],
            risk_scaling_events=0,
            total_frictions_paid=total_frictions
        )
