"""
Quantitative Analyzer for Multi-Agent Option Pricing

Advanced quantitative analysis tools specifically designed for practical
risk management applications. Focuses on the key insights that quantitative
teams need:

1. Tail risk assessment and model risk quantification
2. Hedge effectiveness analysis under market frictions
3. Stress testing and regime identification
4. VaR/CVaR calibration with realistic deviations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
import warnings

from ..multi_agent.agent_interaction import MarketEquilibrium, MarketRegime


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for quantitative applications"""

    # Model risk assessment
    tail_risk_underestimation: float  # How much BS underestimates tail events
    model_var_adjustment: float  # VaR adjustment factor for model risk
    model_cvar_adjustment: float  # CVaR adjustment factor

    # Hedge effectiveness metrics
    delta_hedge_effectiveness: float  # Expected hedge ratio performance
    gamma_hedge_deterioration: float  # Gamma hedging degradation
    vega_hedge_reliability: float  # Volatility hedging reliability

    # Market stability indicators
    regime_stability_score: float  # How stable current regime is (0-1)
    systemic_fragility_indicator: float  # Early warning for market stress
    liquidity_risk_premium: float  # Premium required for liquidity risk

    # Practical trading metrics
    transaction_cost_impact: float  # Effective transaction cost multiplication
    market_impact_estimate: float  # Expected market impact per trade
    optimal_rebalancing_frequency: float  # Recommended hedge rebalancing


class QuantitativeAnalyzer:
    """
    Advanced quantitative analysis engine for multi-agent option markets.

    Provides sophisticated risk analytics that go beyond traditional models
    to account for realistic market frictions and agent behaviors.
    """

    def __init__(self):
        """Initialize quantitative analyzer."""
        self.historical_equilibria: List[MarketEquilibrium] = []
        self.regime_transitions: List[Tuple[float, MarketRegime, MarketRegime]] = []

        # Risk model parameters
        self.confidence_levels = [0.95, 0.99, 0.995]  # For VaR/CVaR calculations
        self.hedge_horizons = [1, 5, 21]  # Days for hedge effectiveness analysis

    def analyze_equilibrium_sequence(self, equilibria: List[MarketEquilibrium],
                                   underlying_prices: List[float],
                                   timestamps: List[float]) -> RiskMetrics:
        """
        Analyze a sequence of market equilibria to extract risk metrics.

        Parameters:
        -----------
        equilibria : List[MarketEquilibrium]
            Sequence of market states
        underlying_prices : List[float]
            Corresponding underlying asset prices
        timestamps : List[float]
            Time stamps for each equilibrium

        Returns:
        --------
        RiskMetrics
            Comprehensive risk analysis
        """
        self.historical_equilibria = equilibria

        # 1. Model risk assessment
        model_risk_metrics = self._assess_model_risk(equilibria, underlying_prices)

        # 2. Hedge effectiveness analysis
        hedge_metrics = self._analyze_hedge_effectiveness(equilibria, underlying_prices, timestamps)

        # 3. Market stability analysis
        stability_metrics = self._assess_market_stability(equilibria, timestamps)

        # 4. Trading cost analysis
        cost_metrics = self._analyze_trading_costs(equilibria)

        # Combine all metrics
        risk_metrics = RiskMetrics(
            **model_risk_metrics,
            **hedge_metrics,
            **stability_metrics,
            **cost_metrics
        )

        return risk_metrics

    def _assess_model_risk(self, equilibria: List[MarketEquilibrium],
                         underlying_prices: List[float]) -> Dict[str, float]:
        """
        Assess how much traditional models underestimate risk.

        Key insight: Multi-agent frictions create fat tails and skewed distributions
        that Black-Scholes fundamentally cannot capture.
        """

        if len(equilibria) < 10:
            return {
                'tail_risk_underestimation': 1.0,
                'model_var_adjustment': 1.0,
                'model_cvar_adjustment': 1.0
            }

        # Extract pricing deviations across all options and time
        all_deviations = []
        relative_deviations = []

        for eq in equilibria:
            for (strike, expiry), deviation in eq.pricing_deviations.items():
                theoretical_price = deviation + eq.equilibrium_prices.get((strike, expiry), deviation)
                if theoretical_price > 0:
                    all_deviations.append(deviation)
                    relative_deviations.append(deviation / theoretical_price)

        if not all_deviations:
            return {
                'tail_risk_underestimation': 1.0,
                'model_var_adjustment': 1.0,
                'model_cvar_adjustment': 1.0
            }

        deviations = np.array(all_deviations)
        rel_deviations = np.array(relative_deviations)

        # Assess tail behavior vs normal distribution
        # Black-Scholes assumes log-normal underlying, normal option P&L
        # Reality: fat tails due to liquidity, market microstructure

        # Compare empirical tail quantiles to normal
        normal_99 = stats.norm.ppf(0.99, loc=np.mean(rel_deviations), scale=np.std(rel_deviations))
        empirical_99 = np.percentile(rel_deviations, 99)

        normal_995 = stats.norm.ppf(0.995, loc=np.mean(rel_deviations), scale=np.std(rel_deviations))
        empirical_995 = np.percentile(rel_deviations, 99.5)

        # Tail risk underestimation factors
        tail_99_factor = abs(empirical_99 / normal_99) if normal_99 != 0 else 1.0
        tail_995_factor = abs(empirical_995 / normal_995) if normal_995 != 0 else 1.0

        tail_risk_underestimation = max(1.0, np.mean([tail_99_factor, tail_995_factor]))

        # VaR/CVaR adjustments
        # VaR needs multiplicative adjustment for fat tails
        var_adjustment = tail_99_factor

        # CVaR (expected shortfall) needs even larger adjustment
        cvar_adjustment = tail_995_factor

        return {
            'tail_risk_underestimation': tail_risk_underestimation,
            'model_var_adjustment': var_adjustment,
            'model_cvar_adjustment': cvar_adjustment
        }

    def _analyze_hedge_effectiveness(self, equilibria: List[MarketEquilibrium],
                                   underlying_prices: List[float],
                                   timestamps: List[float]) -> Dict[str, float]:
        """
        Analyze hedge effectiveness degradation due to market frictions.

        Key insight: Perfect delta hedging assumes continuous trading at zero cost.
        Multi-agent frictions make hedging less effective, especially for gamma/vega.
        """

        if len(equilibria) < 21:  # Need at least 21 periods for analysis
            return {
                'delta_hedge_effectiveness': 0.8,
                'gamma_hedge_deterioration': 0.2,
                'vega_hedge_reliability': 0.7
            }

        # Simulate hedge performance across different strategies
        delta_effectiveness = self._simulate_delta_hedging(equilibria, underlying_prices, timestamps)
        gamma_deterioration = self._assess_gamma_hedging(equilibria, underlying_prices)
        vega_reliability = self._assess_vega_hedging(equilibria)

        return {
            'delta_hedge_effectiveness': delta_effectiveness,
            'gamma_hedge_deterioration': gamma_deterioration,
            'vega_hedge_reliability': vega_reliability
        }

    def _simulate_delta_hedging(self, equilibria: List[MarketEquilibrium],
                              underlying_prices: List[float],
                              timestamps: List[float]) -> float:
        """
        Simulate delta hedging performance with realistic transaction costs and spreads.
        """

        # For each option, simulate delta hedging over the period
        hedge_errors = []

        # Focus on ATM options for representative analysis
        for eq in equilibria[:10]:  # Analyze first 10 periods to avoid overfitting

            for (strike, expiry), eq_price in eq.equilibrium_prices.items():
                if len(underlying_prices) < 5:
                    continue

                # Calculate theoretical delta (simplified)
                spot = underlying_prices[0] if underlying_prices else 100.0
                moneyness = spot / strike

                # Focus on near-ATM options (0.9 < S/K < 1.1)
                if not (0.9 <= moneyness <= 1.1):
                    continue

                # Simplified delta calculation
                time_to_expiry = max(0.01, expiry)
                d1 = (np.log(moneyness) + 0.5 * 0.2**2 * time_to_expiry) / (0.2 * np.sqrt(time_to_expiry))
                theoretical_delta = stats.norm.cdf(d1)

                # Simulate hedge over next few periods
                if len(underlying_prices) >= 5:
                    hedge_pnl = self._calculate_hedge_pnl(
                        theoretical_delta, underlying_prices[:5], eq_price
                    )
                    perfect_hedge_pnl = 0.0  # Perfect hedge has zero P&L

                    hedge_error = abs(hedge_pnl - perfect_hedge_pnl) / eq_price
                    hedge_errors.append(hedge_error)

        if not hedge_errors:
            return 0.8  # Default conservative estimate

        # Hedge effectiveness = 1 - average relative hedge error
        avg_error = np.mean(hedge_errors)
        effectiveness = max(0.1, 1.0 - min(avg_error, 0.9))

        return effectiveness

    def _calculate_hedge_pnl(self, delta: float, price_path: List[float], option_price: float) -> float:
        """Calculate P&L from delta hedging over a price path."""

        if len(price_path) < 2:
            return 0.0

        # Simplified hedge P&L calculation
        total_pnl = 0.0
        current_hedge = delta

        for i in range(1, len(price_path)):
            spot_change = price_path[i] - price_path[i-1]

            # P&L from hedge position
            hedge_pnl = current_hedge * spot_change

            # Option P&L (simplified as delta * spot change + gamma effect)
            gamma_effect = 0.5 * 0.1 * spot_change**2  # Simplified gamma
            option_pnl = delta * spot_change + gamma_effect

            # Net hedged P&L
            net_pnl = option_pnl - hedge_pnl
            total_pnl += net_pnl

            # Update hedge (simplified - assume static delta)
            # In reality, delta changes with spot and time

        return total_pnl

    def _assess_gamma_hedging(self, equilibria: List[MarketEquilibrium],
                            underlying_prices: List[float]) -> float:
        """
        Assess gamma hedging deterioration due to discrete rebalancing and transaction costs.
        """

        # Gamma hedging deterioration increases with:
        # 1. Wider bid-ask spreads
        # 2. Less frequent rebalancing
        # 3. Market volatility regime changes

        if not equilibria:
            return 0.2  # Default estimate

        # Calculate average spread impact
        spread_impacts = []
        for eq in equilibria[-10:]:  # Last 10 periods
            for (strike, expiry), (bid, ask) in eq.bid_ask_spreads.items():
                if ask > bid:
                    spread_impact = (ask - bid) / ((ask + bid) / 2)
                    spread_impacts.append(spread_impact)

        avg_spread_impact = np.mean(spread_impacts) if spread_impacts else 0.02

        # Gamma deterioration scales with spread impact
        deterioration = min(0.8, avg_spread_impact * 5)  # 2% spread -> 10% deterioration

        return deterioration

    def _assess_vega_hedging(self, equilibria: List[MarketEquilibrium]) -> float:
        """
        Assess vega hedging reliability given volatility smile dynamics.
        """

        if len(equilibria) < 5:
            return 0.7  # Default estimate

        # Analyze volatility surface stability
        iv_changes = []

        for i in range(1, len(equilibria)):
            prev_ivs = equilibria[i-1].implied_volatilities
            curr_ivs = equilibria[i].implied_volatilities

            for key in prev_ivs:
                if key in curr_ivs:
                    iv_change = abs(curr_ivs[key] - prev_ivs[key])
                    iv_changes.append(iv_change)

        if not iv_changes:
            return 0.7

        # Higher IV volatility -> less reliable vega hedging
        avg_iv_change = np.mean(iv_changes)
        vega_reliability = max(0.3, 1.0 - avg_iv_change * 2)  # Scale by factor of 2

        return vega_reliability

    def _assess_market_stability(self, equilibria: List[MarketEquilibrium],
                               timestamps: List[float]) -> Dict[str, float]:
        """
        Assess market regime stability and systemic risk indicators.
        """

        if len(equilibria) < 10:
            return {
                'regime_stability_score': 0.5,
                'systemic_fragility_indicator': 0.3,
                'liquidity_risk_premium': 0.02
            }

        # Regime transition frequency
        regime_changes = 0
        for i in range(1, len(equilibria)):
            if equilibria[i].market_regime != equilibria[i-1].market_regime:
                regime_changes += 1

        regime_stability = max(0.0, 1.0 - regime_changes / len(equilibria))

        # Systemic fragility based on concentration and arbitrage capacity
        recent_eq = equilibria[-10:]
        avg_arbitrage_capacity = np.mean([eq.arbitrage_capacity for eq in recent_eq])
        avg_systemic_risk = np.mean([eq.systemic_risk_indicator for eq in recent_eq])

        systemic_fragility = avg_systemic_risk * (1 - avg_arbitrage_capacity)

        # Liquidity risk premium from spreads and regime instability
        avg_liquidity_score = np.mean([eq.liquidity_score for eq in recent_eq])
        liquidity_risk_premium = max(0.001, (1 - avg_liquidity_score) * 0.1)

        return {
            'regime_stability_score': regime_stability,
            'systemic_fragility_indicator': systemic_fragility,
            'liquidity_risk_premium': liquidity_risk_premium
        }

    def _analyze_trading_costs(self, equilibria: List[MarketEquilibrium]) -> Dict[str, float]:
        """
        Analyze effective trading costs and market impact.
        """

        if not equilibria:
            return {
                'transaction_cost_impact': 1.2,
                'market_impact_estimate': 0.005,
                'optimal_rebalancing_frequency': 1.0
            }

        # Transaction cost multiplication factor
        # How much more expensive trading is vs theoretical
        avg_spreads = []
        for eq in equilibria[-20:]:  # Last 20 periods
            for (strike, expiry), (bid, ask) in eq.bid_ask_spreads.items():
                if ask > bid:
                    rel_spread = (ask - bid) / ((ask + bid) / 2)
                    avg_spreads.append(rel_spread)

        avg_spread = np.mean(avg_spreads) if avg_spreads else 0.02
        transaction_cost_impact = 1.0 + avg_spread  # Base cost + spread

        # Market impact estimate
        # How much prices move per unit of trading
        market_impact = avg_spread * 0.5  # Half the spread as market impact estimate

        # Optimal rebalancing frequency
        # Higher costs -> less frequent rebalancing
        if transaction_cost_impact > 1.05:  # >5% transaction costs
            optimal_frequency = 0.2  # Rebalance every 5 days instead of daily
        elif transaction_cost_impact > 1.02:  # >2% transaction costs
            optimal_frequency = 0.5  # Rebalance every 2 days
        else:
            optimal_frequency = 1.0  # Daily rebalancing OK

        return {
            'transaction_cost_impact': transaction_cost_impact,
            'market_impact_estimate': market_impact,
            'optimal_rebalancing_frequency': optimal_frequency
        }

    def generate_risk_report(self, risk_metrics: RiskMetrics,
                           underlying_symbol: str = "UNDERLYING") -> str:
        """
        Generate comprehensive risk report for quantitative team consumption.
        """

        report = f"""
# Multi-Agent Option Pricing Risk Analysis Report

## Executive Summary for {underlying_symbol}

**Key Risk Adjustments:**
- Traditional VaR should be multiplied by {risk_metrics.model_var_adjustment:.2f}
- Traditional CVaR should be multiplied by {risk_metrics.model_cvar_adjustment:.2f}
- Tail risk is underestimated by factor of {risk_metrics.tail_risk_underestimation:.2f}

**Hedging Effectiveness:**
- Delta hedging effectiveness: {risk_metrics.delta_hedge_effectiveness:.1%}
- Gamma hedging deterioration: {risk_metrics.gamma_hedge_deterioration:.1%}
- Vega hedging reliability: {risk_metrics.vega_hedge_reliability:.1%}

**Market Conditions:**
- Regime stability score: {risk_metrics.regime_stability_score:.2f}/1.0
- Systemic fragility indicator: {risk_metrics.systemic_fragility_indicator:.2f}
- Liquidity risk premium: {risk_metrics.liquidity_risk_premium:.2%}

## Quantitative Recommendations

### 1. Risk Model Adjustments
- **VaR Models:** Apply {risk_metrics.model_var_adjustment:.2f}x multiplier to account for fat tails
- **Stress Testing:** Include {risk_metrics.tail_risk_underestimation:.1f}x tail scenarios
- **Capital Requirements:** Consider {risk_metrics.systemic_fragility_indicator:.1%} systemic buffer

### 2. Hedging Strategy Optimization
- **Delta Hedging:** Expected {risk_metrics.delta_hedge_effectiveness:.1%} effectiveness
- **Rebalancing Frequency:** Optimal frequency {risk_metrics.optimal_rebalancing_frequency:.1f}x daily
- **Transaction Cost Budget:** {risk_metrics.transaction_cost_impact:.1%} premium over theoretical

### 3. Position Sizing Guidelines
- **Market Impact:** Expect {risk_metrics.market_impact_estimate:.2%} price impact per trade
- **Liquidity Premium:** Add {risk_metrics.liquidity_risk_premium:.2%} to discount rates
- **Regime Risk:** Monitor for stability score below 0.5

### 4. Early Warning Indicators
- **Systemic Fragility > 0.7:** Reduce position sizes by 50%
- **Liquidity Score < 0.3:** Increase hedging frequency
- **Tail Risk Multiplier > 2.0:** Review risk limits immediately

---
*Report generated by Multi-Agent Option Pricing Risk Engine*
*For quantitative risk management applications*
        """

        return report.strip()

    def get_var_cvar_adjustments(self, confidence_levels: List[float] = None) -> Dict[str, Dict[str, float]]:
        """
        Get specific VaR/CVaR adjustment factors for different confidence levels.

        Returns:
        --------
        Dict with adjustment factors for practical risk management use
        """

        if confidence_levels is None:
            confidence_levels = self.confidence_levels

        if not self.historical_equilibria:
            # Default conservative adjustments
            return {
                f"confidence_{cl:.1%}": {
                    "var_multiplier": 1.5,
                    "cvar_multiplier": 2.0,
                    "tail_probability_adjustment": cl * 1.2
                }
                for cl in confidence_levels
            }

        adjustments = {}

        # Extract all pricing deviations
        all_deviations = []
        for eq in self.historical_equilibria:
            for deviation in eq.pricing_deviations.values():
                all_deviations.append(deviation)

        if all_deviations:
            deviations = np.array(all_deviations)

            for cl in confidence_levels:
                # Compare empirical vs normal quantiles
                empirical_quantile = np.percentile(deviations, cl * 100)
                normal_quantile = stats.norm.ppf(cl, loc=np.mean(deviations), scale=np.std(deviations))

                var_multiplier = abs(empirical_quantile / normal_quantile) if normal_quantile != 0 else 1.5

                # CVaR requires expected shortfall beyond VaR
                tail_deviations = deviations[deviations >= empirical_quantile]
                cvar_empirical = np.mean(tail_deviations) if len(tail_deviations) > 0 else empirical_quantile
                cvar_normal = np.mean(deviations[deviations >= normal_quantile]) if normal_quantile != 0 else normal_quantile

                cvar_multiplier = abs(cvar_empirical / cvar_normal) if cvar_normal != 0 else 2.0

                adjustments[f"confidence_{cl:.1%}"] = {
                    "var_multiplier": max(1.0, var_multiplier),
                    "cvar_multiplier": max(1.0, cvar_multiplier),
                    "tail_probability_adjustment": cl * (1 + var_multiplier * 0.1)
                }

        return adjustments