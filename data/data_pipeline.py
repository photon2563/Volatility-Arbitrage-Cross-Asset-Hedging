"""
Institutional Data Engineering Pipeline for VIX Futures and S&P 500 E-mini.

This module addresses two critical engineering requirements in quantitative derivatives trading:
1. Dual Continuous Series Construction:
   - Unadjusted Nominal Series: Preserves exact historical spot and futures prices to compute the
     true economic basis (Roll = (VX - VIX_spot) / TTS) without backward-adjustment distortion.
   - Ratio-Adjusted (Panama-Canal) Return Series: Eliminates artificial price jumps on contract roll dates,
     providing clean continuous returns for Kalman filter measurement updates and PnL accounting.
2. Dynamic Market Frictions Model:
   - Implements institutional round-trip commissions ($2.50/contract).
   - Simulates liquidity vacuums during market stress via dynamic slippage scaling linearly with spot VIX:
     Slippage(t) = Base_Slippage * (VIX_spot(t) / 20.0).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class MarketDataSeries:
    """Container for aligned historical or synthetic market data and processed continuous series."""
    dates: pd.DatetimeIndex
    spot_vix: np.ndarray          # VIX_S,t
    es_price: np.ndarray          # S&P 500 E-mini nominal price
    vx1_price: np.ndarray         # Front-month VIX future nominal price
    vx2_price: np.ndarray         # Second-month VIX future nominal price
    tts1: np.ndarray              # Time-to-settlement for VX1 (business days)
    tts2: np.ndarray              # Time-to-settlement for VX2 (business days)
    rf_rate: np.ndarray           # 3-month T-Bill annualized rate (e.g., 0.04 for 4%)
    
    # Processed Dual Continuous Series
    active_contract: np.ndarray   # 1 for VX1, 2 for VX2 (based on 10-day roll rule)
    unadj_vx_price: np.ndarray    # Unadjusted nominal active VX price (for basis signal)
    unadj_tts: np.ndarray         # Unadjusted active TTS (for basis signal)
    vx_returns: np.ndarray        # Ratio-adjusted continuous daily returns (for Kalman/PnL)
    es_returns: np.ndarray        # Continuous daily returns of ES futures
    
    # Dynamic Frictions
    slippage_pts: np.ndarray      # Estimated slippage per contract in index points
    commission_per_contract: float = 2.50  # Institutional round-trip commission ($)


class DataPipeline:
    """
    Constructs institutional-grade datasets and dual continuous series for VIX basis trading.
    """
    
    def __init__(
        self,
        roll_threshold_days: int = 10,
        base_slippage_pts: float = 0.05,
        commission_per_contract: float = 2.50,
        random_seed: int = 42
    ):
        """
        Args:
            roll_threshold_days: Business days to expiration when focus shifts to second-month contract.
            base_slippage_pts: Base slippage in index points at standard VIX=20.0 (0.05 pts = $50 on VX).
            commission_per_contract: Round-trip commission in dollars per contract.
            random_seed: Random seed for synthetic institutional data simulation.
        """
        self.roll_threshold_days = roll_threshold_days
        self.base_slippage_pts = base_slippage_pts
        self.commission_per_contract = commission_per_contract
        self.random_seed = random_seed

    def generate_synthetic_dataset(
        self,
        start_date: str = "2020-01-02",
        trading_days: int = 1260  # ~5 years of daily data
    ) -> MarketDataSeries:
        """
        Generates a high-fidelity synthetic dataset calibrated to empirical VIX/ES dynamics:
        - Persistent contango in calm regimes (VX > VIX_spot).
        - Sharp backwardation spikes during market shocks (COVID 2020, 2022 bear market).
        - Strong negative correlation between ES returns and VIX changes (~ -0.75).
        """
        np.random.seed(self.random_seed)
        dates = pd.date_range(start=start_date, periods=trading_days, freq="B")
        
        # 1. Simulate S&P 500 E-mini (ES) with regime jumps and volatility clustering
        es_price = np.zeros(trading_days)
        es_price[0] = 3250.0  # Jan 2020 starting level
        
        # We model 3 distinct market regimes over the 5-year span:
        # Regime 0: Normal Bull Market (low vol, positive drift, contango)
        # Regime 1: Severe Crisis / Crash (high vol, negative drift, backwardation - e.g., COVID March 2020)
        # Regime 2: Bear Market / High Inflation (elevated vol, choppy drift - e.g., 2022)
        regimes = np.zeros(trading_days, dtype=int)
        # 2020 COVID shock: days 35 to 80
        regimes[35:80] = 1
        # 2022 Bear market: days 500 to 750
        regimes[500:750] = 2
        
        # Generate correlated shocks for ES and VIX spot
        corr = -0.76  # Strong empirical negative correlation
        cov_matrix = [[1.0, corr], [corr, 1.0]]
        shocks = np.random.multivariate_normal([0, 0], cov_matrix, size=trading_days)
        
        spot_vix = np.zeros(trading_days)
        spot_vix[0] = 14.5  # Calm initial VIX
        
        rf_rate = np.linspace(0.015, 0.052, trading_days)  # T-Bill rate rising over 2020-2024
        
        for t in range(1, trading_days):
            reg = regimes[t]
            if reg == 0:    # Calm Bull Market
                mu_es, vol_es = 0.0005, 0.009
                vix_target, kappa, vol_vix = 15.0, 0.08, 0.06
            elif reg == 1:  # Crisis / Crash
                mu_es, vol_es = -0.0035, 0.035
                vix_target, kappa, vol_vix = 65.0, 0.05, 0.25
            else:           # Choppy Bear Market
                mu_es, vol_es = -0.0002, 0.016
                vix_target, kappa, vol_vix = 26.0, 0.06, 0.12
                
            # ES evolution (Geometric Brownian Motion w/ regime parameters)
            es_ret = mu_es + vol_es * shocks[t, 0]
            es_price[t] = es_price[t-1] * (1.0 + es_ret)
            
            # Spot VIX evolution (Mean-reverting w/ correlated jump)
            dvix = kappa * (vix_target - spot_vix[t-1]) + spot_vix[t-1] * vol_vix * shocks[t, 1]
            spot_vix[t] = np.clip(spot_vix[t-1] + dvix, 9.5, 85.0)

        # 2. Simulate Time-To-Settlement (TTS) and VIX Futures Prices
        # Monthly contract cycles (~21 business days)
        tts1 = np.zeros(trading_days, dtype=int)
        tts2 = np.zeros(trading_days, dtype=int)
        
        cycle_len = 21
        curr_tts = cycle_len
        for t in range(trading_days):
            tts1[t] = curr_tts
            tts2[t] = curr_tts + cycle_len
            curr_tts -= 1
            if curr_tts == 0:
                curr_tts = cycle_len

        # Simon & Campasano empirical basis: In calm regimes, VX is priced higher than spot (contango).
        # In distress, VX is priced lower than spot (backwardation w/ downward mean-reversion expectation).
        vx1_price = np.zeros(trading_days)
        vx2_price = np.zeros(trading_days)
        
        for t in range(trading_days):
            v_spot = spot_vix[t]
            reg = regimes[t]
            
            if reg == 0:  # Contango: futures overpriced by ~1.5 to 3.0 points amortized over time
                prem1 = 1.8 * (tts1[t] / 21.0) + np.random.normal(0, 0.15)
                prem2 = 3.2 * (tts2[t] / 42.0) + np.random.normal(0, 0.20)
            elif reg == 1:  # Backwardation: spot spikes above futures as market anticipates reversion
                prem1 = -12.0 * (tts1[t] / 21.0) + np.random.normal(0, 0.60)
                prem2 = -8.0 * (tts2[t] / 42.0) + np.random.normal(0, 0.50)
            else:  # Choppy / Mild Contango or Flat
                prem1 = 0.6 * (tts1[t] / 21.0) + np.random.normal(0, 0.25)
                prem2 = 1.4 * (tts2[t] / 42.0) + np.random.normal(0, 0.30)
                
            vx1_price[t] = np.clip(v_spot + prem1, 10.0, 80.0)
            vx2_price[t] = np.clip(v_spot + prem2, 10.5, 82.0)

        return self.build_continuous_series(
            dates=dates,
            spot_vix=spot_vix,
            es_price=es_price,
            vx1_price=vx1_price,
            vx2_price=vx2_price,
            tts1=tts1,
            tts2=tts2,
            rf_rate=rf_rate
        )

    def build_continuous_series(
        self,
        dates: pd.DatetimeIndex,
        spot_vix: np.ndarray,
        es_price: np.ndarray,
        vx1_price: np.ndarray,
        vx2_price: np.ndarray,
        tts1: np.ndarray,
        tts2: np.ndarray,
        rf_rate: np.ndarray
    ) -> MarketDataSeries:
        """
        Stitches front and second-month contracts into dual continuous series.
        """
        n = len(dates)
        active_contract = np.ones(n, dtype=int)
        unadj_vx_price = np.zeros(n)
        unadj_tts = np.zeros(n)
        vx_returns = np.zeros(n)
        es_returns = np.zeros(n)
        slippage_pts = np.zeros(n)
        
        for t in range(n):
            # Dynamic Roll Rule: Switch focus to second-month contract when front-month TTS <= threshold
            if tts1[t] <= self.roll_threshold_days:
                active_contract[t] = 2
                unadj_vx_price[t] = vx2_price[t]
                unadj_tts[t] = tts2[t]
            else:
                active_contract[t] = 1
                unadj_vx_price[t] = vx1_price[t]
                unadj_tts[t] = tts1[t]
                
            # Dynamic Slippage Scaling: increases linearly during volatility spikes
            # At VIX=20, slippage is base_slippage_pts (0.05). At VIX=60, slippage is 3x (0.15 pts = $150/contract).
            slippage_pts[t] = self.base_slippage_pts * (spot_vix[t] / 20.0)

        # Calculate continuous returns
        for t in range(1, n):
            # ES returns are straight percentage changes
            es_returns[t] = (es_price[t] - es_price[t-1]) / es_price[t-1]
            
            # Ratio-Adjusted (Panama-Canal style) return for VIX Futures:
            # On roll days (when active_contract changes from 1 to 2), we MUST compute return
            # using the SAME contract as yesterday (i.e., VX1 today vs VX1 yesterday) to prevent
            # recording an artificial price jump between VX1 and VX2 as a trading gain/loss!
            prev_contract = active_contract[t-1]
            if prev_contract == 1:
                p_today = vx1_price[t]
                p_prev = vx1_price[t-1]
            else:
                p_today = vx2_price[t]
                p_prev = vx2_price[t-1]
                
            vx_returns[t] = (p_today - p_prev) / p_prev

        return MarketDataSeries(
            dates=dates,
            spot_vix=spot_vix,
            es_price=es_price,
            vx1_price=vx1_price,
            vx2_price=vx2_price,
            tts1=tts1,
            tts2=tts2,
            rf_rate=rf_rate,
            active_contract=active_contract,
            unadj_vx_price=unadj_vx_price,
            unadj_tts=unadj_tts,
            vx_returns=vx_returns,
            es_returns=es_returns,
            slippage_pts=slippage_pts,
            commission_per_contract=self.commission_per_contract
        )
