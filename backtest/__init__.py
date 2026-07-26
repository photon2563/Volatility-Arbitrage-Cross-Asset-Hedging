"""
Backtesting Suite for VIX VRP Harvester.
Simulates event-driven / vectorized institutional execution across 4 benchmarks
with Algorithmic Risk Scaling based on Kalman Filter uncertainty diagnostics.
"""

from .backtester import Backtester, StrategyResult

__all__ = ["Backtester", "StrategyResult"]
