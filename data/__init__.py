"""
Data Engineering Module for VIX VRP Harvester.
Provides synthetic institutional market data simulation and dual continuous series construction
(Unadjusted Nominal for signal generation, Ratio-Adjusted for Kalman filtering and PnL).
"""

from .data_pipeline import DataPipeline, MarketDataSeries

__all__ = ["DataPipeline", "MarketDataSeries"]
