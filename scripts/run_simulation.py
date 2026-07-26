"""
Master Execution Script for VIX VRP Harvester Simulation.

Executes:
1. Institutional Data Engineering Pipeline (5-year synthetic daily data w/ dual continuous series).
2. Simon & Campasano Basis Signal Generation (Contango/Backwardation state machine).
3. Dynamic State-Space Kalman Filtering vs Static Rolling OLS Regression.
4. Multi-Strategy Institutional Backtest across 4 benchmarks w/ Algorithmic Risk Scaling.
5. Printout of Quantitative Evaluation Metrics Table and Trade Justification Matrix logs.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data_pipeline import DataPipeline
from engine.signals import SignalEngine
from engine.kalman import KalmanHedgeEngine
from engine.ols_benchmark import OLSHedgeEngine
from backtest.backtester import Backtester
from analytics.metrics import evaluate_strategy
from analytics.dashboard import TransparencyDashboard


def main():
    print("====================================================================================")
    print("   DA VINCI DERIVATIVES: QUANTITATIVE TRADING SYSTEM SIMULATION ENGINE")
    print("   Project: Harvesting the Volatility Risk Premium via State-Space Hedged VIX Futures")
    print("====================================================================================\n")
    
    # 1. Initialize Data Pipeline
    print("[1/5] Initializing Institutional Data Pipeline & Simulating Market Dynamics...")
    pipeline = DataPipeline(
        roll_threshold_days=10,
        base_slippage_pts=0.05,
        commission_per_contract=2.50,
        random_seed=42
    )
    data = pipeline.generate_synthetic_dataset(start_date="2020-01-02", trading_days=1260)
    print(f"      -> Simulated {len(data.dates)} business days ({data.dates[0].strftime('%Y-%m-%d')} to {data.dates[-1].strftime('%Y-%m-%d')}).")
    print(f"      -> Built Unadjusted Nominal Series (for signals) & Ratio-Adjusted Return Series (for Kalman/PnL).")
    
    # 2. Generate Trading Signals
    print("\n[2/5] Running Simon & Campasano Signal Engine (Normalized Daily Roll)...")
    signal_engine = SignalEngine(tau_upper=0.08, tau_lower=-0.05)
    signals = signal_engine.generate_signals(
        dates=data.dates,
        unadj_vx_price=data.unadj_vx_price,
        spot_vix=data.spot_vix,
        unadj_tts=data.unadj_tts
    )
    positions = np.array([s.signal for s in signals])
    short_days = np.sum(positions == -1)
    long_days = np.sum(positions == 1)
    neutral_days = np.sum(positions == 0)
    print(f"      -> Signal State Breakdown: {short_days} Days Short Contango | {long_days} Days Long Backwardation | {neutral_days} Days Neutral/Standby.")

    # 3. Execute Dynamic Kalman & Static OLS Hedging Engines
    print("\n[3/5] Executing Kalman Filter State-Space Model vs Static OLS Regression...")
    kalman_engine = KalmanHedgeEngine(v_e=1e-3, v_w=1e-5, init_beta=-0.75)
    kalman_states = kalman_engine.filter_series(
        dates=data.dates,
        vx_returns=data.vx_returns,
        es_returns=data.es_returns,
        es_prices=data.es_price,
        vx_prices=data.unadj_vx_price,
        vx_positions=positions
    )
    
    ols_engine = OLSHedgeEngine(window_days=60, init_beta=-0.75)
    ols_history = ols_engine.filter_series(
        dates=data.dates,
        vx_returns=data.vx_returns,
        es_returns=data.es_returns,
        es_prices=data.es_price,
        vx_prices=data.unadj_vx_price,
        vx_positions=positions
    )
    avg_kalman_beta = np.mean([k.beta_post for k in kalman_states])
    avg_ols_beta = np.mean([o["beta_ols"] for o in ols_history])
    print(f"      -> Kalman Filter Mean Posterior Beta: {avg_kalman_beta:.4f} | OLS Mean Static Beta: {avg_ols_beta:.4f}")

    # 4. Run Multi-Strategy Institutional Backtest
    print("\n[4/5] Running Institutional Backtest w/ Dynamic Slippage & Algorithmic Risk Scaling...")
    backtester = Backtester(
        initial_capital=5_000_000.0,
        base_vx_contracts=50.0,
        risk_scaling_z_threshold=3.5,
        risk_scaling_factor=0.5
    )
    benchmarks = backtester.run_all_benchmarks(data, signals, kalman_states, ols_history)
    
    # 5. Evaluate Quantitative Performance Metrics
    print("\n[5/5] Computing Quantitative Evaluation Metrics Suite...\n")
    metrics_list = []
    for name, res in benchmarks.items():
        m = evaluate_strategy(res, data.rf_rate, data.es_returns)
        metrics_list.append(m)
        
    table_str = TransparencyDashboard.format_metrics_table(metrics_list)
    print(table_str)
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    with open("results/metrics_summary.txt", "w") as f:
        f.write(table_str)
    print("\n -> Metrics table successfully exported to `results/metrics_summary.txt`.")

    # Print Sample Real-Time Trade Justification Logs
    print("\n====================================================================================")
    print("   SAMPLE TRANSPARENCY DASHBOARD: TRADE JUSTIFICATION MATRIX LOGS")
    print("====================================================================================\n")
    
    # Find sample entry short, sample risk scaling event, and sample reversal
    logged_count = 0
    for t in range(1, len(signals)):
        s = signals[t]
        k = kalman_states[t]
        is_scaled = abs(k.z_score) > 3.5
        
        if s.transition in ["ENTRY_SHORT", "ENTRY_LONG", "REVERSAL_LONG", "REVERSAL_SHORT"] or is_scaled:
            print(TransparencyDashboard.generate_trade_justification_log(s, k, is_scaled))
            print("")
            logged_count += 1
            if logged_count >= 3:
                break
                
    print("Simulation completed successfully. To generate charts, run: `python3 scripts/generate_plots.py`")


if __name__ == "__main__":
    main()
