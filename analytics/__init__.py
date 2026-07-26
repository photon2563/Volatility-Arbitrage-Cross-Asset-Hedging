"""
Quantitative Analytics Suite for VIX VRP Harvester.
Provides return profiles, tail-risk profiles, and trade execution statistics.
"""

from .metrics import PerformanceMetrics, evaluate_strategy
from .dashboard import TransparencyDashboard

__all__ = ["PerformanceMetrics", "evaluate_strategy", "TransparencyDashboard"]
