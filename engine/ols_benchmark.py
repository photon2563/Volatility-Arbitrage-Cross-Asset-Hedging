"""
Static Ordinary Least Squares (OLS) Benchmark Hedging Engine.

Implements a rolling window OLS regression between VIX futures returns (y) and ES futures returns (x)
to represent the traditional academic/retail statistical arbitrage baseline.
Illustrates how static window models lag market reality during sudden volatility regime shifts.
"""

import numpy as np
from typing import List, Dict, Any


class OLSHedgeEngine:
    """
    Rolling window OLS regression estimator for static beta comparison.
    """
    def __init__(
        self,
        window_days: int = 60,
        init_beta: float = -0.75,
        vx_multiplier: float = 1000.0,
        es_multiplier: float = 50.0
    ):
        self.window_days = window_days
        self.init_beta = init_beta
        self.vx_multiplier = vx_multiplier
        self.es_multiplier = es_multiplier

    def filter_series(
        self,
        dates: Any,
        vx_returns: np.ndarray,
        es_returns: np.ndarray,
        es_prices: np.ndarray,
        vx_prices: np.ndarray,
        vx_positions: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Runs rolling OLS over the time series and computes static benchmark ES contract sizing.
        """
        n = len(vx_returns)
        history = []
        
        for t in range(n):
            date_str = str(dates[t])[:10]
            vx_pos = vx_positions[t]
            es_p = es_prices[t]
            vx_p = vx_prices[t]
            
            if t < self.window_days:
                beta_ols = self.init_beta
            else:
                y_win = vx_returns[t-self.window_days : t]
                x_win = es_returns[t-self.window_days : t]
                
                # OLS slope: Cov(y, x) / Var(x)
                x_mean = np.mean(x_win)
                y_mean = np.mean(y_win)
                cov = np.sum((x_win - x_mean) * (y_win - y_mean))
                var = np.sum((x_win - x_mean) ** 2)
                
                beta_ols = cov / var if var != 0 else self.init_beta
                
            vx_notional = vx_p * self.vx_multiplier
            es_notional = es_p * self.es_multiplier
            es_contracts = - vx_pos * beta_ols * (vx_notional / es_notional) if es_notional != 0 else 0.0
            
            history.append({
                "date": date_str,
                "beta_ols": beta_ols,
                "es_contracts_req": es_contracts
            })
            
        return history
