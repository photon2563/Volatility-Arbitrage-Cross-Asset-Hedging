import numpy as np
from data.real_data_loader import RealDataLoader
from engine.signals import SignalEngine
from engine.kalman import KalmanHedgeEngine
from backtest.backtester import Backtester
from analytics.metrics import evaluate_strategy

def run_sweep():
    print("Loading 100% empirical historical data (Jan 2020 - Dec 2023)...")
    data = RealDataLoader.fetch_real_dataset('2020-01-01', '2023-12-31')
    
    signal_engine = SignalEngine(tau_upper=0.03, tau_lower=-0.01)
    signals = signal_engine.generate_signals(data.dates, data.unadj_vx_price, data.spot_vix, data.unadj_tts)
    positions = np.array([s.signal for s in signals])
    
    kalman_engine = KalmanHedgeEngine(v_e=1e-3, v_w=1e-5, init_beta=-0.75)
    kalman_states = kalman_engine.filter_series(data.dates, data.vx_returns, data.es_returns, data.es_price, data.unadj_vx_price, positions)
    
    print("\n=====================================================================================================")
    print("      INSTITUTIONAL CAPITAL UTILIZATION & LEVERAGE SWEEP (100% EMPIRICAL DATA: 2020-2023)")
    print("=====================================================================================================\n")
    print(f"{'Contracts':<12} | {'Notional Util':<15} | {'Kalman CAGR':<12} | {'Sharpe':<8} | {'Sortino':<8} | {'Max Drawdown':<14} | {'Win/Loss':<8}")
    print("-" * 95)
    
    for contracts in [50, 100, 150, 200, 250, 300]:
        bt = Backtester(
            initial_capital=5_000_000.0,
            base_vx_contracts=float(contracts),
            risk_scaling_z_threshold=3.5,
            risk_scaling_factor=0.5
        )
        res = bt._simulate_strategy(
            data=data,
            signals=signals,
            hedge_contracts_req=[k.es_contracts_req for k in kalman_states],
            z_scores=[k.z_score for k in kalman_states],
            name=f"Kalman ({contracts} Contracts)",
            enable_risk_scaling=True,
            is_hedged=True
        )
        m = evaluate_strategy(res, data.rf_rate, data.es_returns)
        
        # Approximate notional utilization assuming avg VIX futures price of ~20 ($20,000 notional per contract)
        util = (contracts * 20000.0) / 5_000_000.0 * 100
        print(f"{contracts:<12} | {util:>13.1f}% | {m.cagr_pct:>11.2f}% | {m.sharpe_ratio:>8.2f} | {m.sortino_ratio:>8.2f} | {m.max_drawdown_pct:>13.2f}% | {m.win_loss_ratio:>8.2f}")
    print("=====================================================================================================\n")

if __name__ == "__main__":
    run_sweep()
