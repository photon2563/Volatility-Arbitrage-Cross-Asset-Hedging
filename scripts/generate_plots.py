"""
Professional Visualization Suite for VIX VRP Harvester.

Generates and saves 4 high-resolution institutional charts in the `results/` directory:
1. Cumulative Equity Curves (Kalman vs OLS vs Unhedged vs S&P 500 Benchmark).
2. Dynamic Kalman Hedge Ratio vs Rolling OLS Beta w/ Market Stress Regimes.
3. Normalized VIX Daily Roll & Trade Signal State Machine.
4. Drawdown Profiles & Tail-Risk Mitigation Analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data_pipeline import DataPipeline
from engine.signals import SignalEngine
from engine.kalman import KalmanHedgeEngine
from engine.ols_benchmark import OLSHedgeEngine
from backtest.backtester import Backtester


def generate_all_plots():
    print("Initializing simulation to generate plot data...")
    pipeline = DataPipeline(roll_threshold_days=10, base_slippage_pts=0.05, commission_per_contract=2.50)
    data = pipeline.generate_synthetic_dataset("2020-01-02", 1260)
    
    signal_engine = SignalEngine(tau_upper=0.08, tau_lower=-0.05)
    signals = signal_engine.generate_signals(data.dates, data.unadj_vx_price, data.spot_vix, data.unadj_tts)
    positions = np.array([s.signal for s in signals])
    rolls = np.array([s.roll for s in signals])
    
    kalman_engine = KalmanHedgeEngine(v_e=1e-3, v_w=1e-5, init_beta=-0.75)
    kalman_states = kalman_engine.filter_series(data.dates, data.vx_returns, data.es_returns, data.es_price, data.unadj_vx_price, positions)
    
    ols_engine = OLSHedgeEngine(window_days=60, init_beta=-0.75)
    ols_history = ols_engine.filter_series(data.dates, data.vx_returns, data.es_returns, data.es_price, data.unadj_vx_price, positions)
    
    backtester = Backtester(initial_capital=5_000_000.0, base_vx_contracts=50.0)
    benchmarks = backtester.run_all_benchmarks(data, signals, kalman_states, ols_history)
    
    os.makedirs("results", exist_ok=True)
    
    # Set professional style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    
    # --- CHART 1: COMPARATIVE EQUITY CURVES ---
    print("Generating Chart 1: Cumulative Equity Curves...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    colors = {
        "Dynamic Kalman (Proposed)": "#1f77b4",  # Deep Blue
        "Static OLS Benchmark": "#ff7f0e",       # Orange
        "Unhedged Basis Strategy": "#d62728",    # Red
        "S&P 500 Buy & Hold": "#7f7f7f"          # Grey
    }
    
    for name, res in benchmarks.items():
        ax.plot(res.dates, res.equity_curve / 1_000_000.0, label=name, color=colors.get(name, "#333333"), linewidth=2.0 if "Kalman" in name else 1.5)
        
    ax.set_title("Institutional Portfolio Performance: State-Space Hedged VIX Basis vs Benchmarks ($5M Initial Capital)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Portfolio Value ($ Millions)", fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results/1_equity_curves.png")
    plt.close()
    
    # --- CHART 2: DYNAMIC BETA TRACKING ---
    print("Generating Chart 2: Dynamic Beta vs Static OLS Beta...")
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
    
    kalman_betas = [k.beta_post for k in kalman_states]
    ols_betas = [o["beta_ols"] for o in ols_history]
    
    ax1.plot(data.dates, kalman_betas, label="Kalman Dynamic Beta ($\\hat{\\beta}_{t|t}$)", color="#1f77b4", linewidth=2.0)
    ax1.plot(data.dates, ols_betas, label="Rolling 60-Day OLS Beta", color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax1.set_title("Kalman Filter vs Rolling OLS: Dynamic Hedge Ratio Adaptation During Market Shocks", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Date", fontweight="bold")
    ax1.set_ylabel("Hedge Ratio (Beta to S&P 500)", fontweight="bold")
    
    # Overlay Spot VIX on secondary axis to show stress regime alignment
    ax2 = ax1.twinx()
    ax2.plot(data.dates, data.spot_vix, label="Spot VIX Index (Right Axis)", color="#d62728", alpha=0.25, linewidth=1.0)
    ax2.set_ylabel("Spot VIX Level", color="#d62728", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True, facecolor="white")
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results/2_dynamic_beta_tracking.png")
    plt.close()

    # --- CHART 3: VIX ROLL SIGNAL STATE MACHINE ---
    print("Generating Chart 3: VIX Roll Signals & Thresholds...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    
    ax1.plot(data.dates, rolls, color="#2ca02c", linewidth=1.5, label="Normalized Daily Roll ($Roll_t$)")
    ax1.axhline(0.08, color="#1f77b4", linestyle="--", label="Contango Short Threshold ($\tau_{upper} = 0.08$)")
    ax1.axhline(-0.05, color="#d62728", linestyle="--", label="Backwardation Long Threshold ($\tau_{lower} = -0.05$)")
    ax1.axhline(0.00, color="black", linestyle=":", alpha=0.5)
    ax1.set_title("Simon & Campasano Normalized VIX Basis & Directional Signal Engine", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Normalized Daily Roll", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white")
    
    ax2.step(data.dates, positions, where="post", color="#333333", linewidth=2.0, label="Active Position (-1: Short, 0: Neutral, +1: Long)")
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["Short (-1)", "Neutral (0)", "Long (+1)"])
    ax2.set_xlabel("Date", fontweight="bold")
    ax2.set_ylabel("Position State", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white")
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results/3_vix_roll_signals.png")
    plt.close()

    # --- CHART 4: DRAWDOWN PROFILES ---
    print("Generating Chart 4: Drawdown Profiles...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    for name, res in benchmarks.items():
        peak = np.maximum.accumulate(res.equity_curve)
        dd = (res.equity_curve - peak) / peak * 100.0
        ax.plot(res.dates, dd, label=name, color=colors.get(name, "#333333"), linewidth=2.0 if "Kalman" in name else 1.5)
        
    ax.set_title("Tail-Risk Mitigation Analysis: Drawdown Percentage Profiles", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Drawdown (%)", fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results/4_drawdown_profiles.png")
    plt.close()
    
    print("\n -> All 4 high-resolution institutional charts successfully saved to `results/`.")


if __name__ == "__main__":
    generate_all_plots()
