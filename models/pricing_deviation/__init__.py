"""
Pricing Deviation Analysis Module

This module implements sophisticated analysis tools for understanding and
quantifying option pricing deviations from theoretical models, specifically
designed for quantitative risk management applications.

Key Components:
- QuantitativeAnalyzer: Advanced analytics for hedge effectiveness and tail risk
"""

from .quantitative_analyzer import QuantitativeAnalyzer, RiskMetrics

__all__ = [
    'QuantitativeAnalyzer', 'RiskMetrics'
]