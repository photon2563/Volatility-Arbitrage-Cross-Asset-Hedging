"""
Core Quantitative Engines for VIX VRP Harvester.
Includes Simon & Campasano basis signal generation, Kalman Filter state-space dynamic hedging,
and static OLS benchmark models.
"""

from .signals import SignalEngine, TradeSignal
from .kalman import KalmanHedgeEngine, KalmanState
from .ols_benchmark import OLSHedgeEngine

__all__ = ["SignalEngine", "TradeSignal", "KalmanHedgeEngine", "KalmanState", "OLSHedgeEngine"]
