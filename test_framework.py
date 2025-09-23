#!/usr/bin/env python3
"""
Comprehensive Test Suite for Derivatives Risk Management Framework

Tests all framework components to ensure proper functionality and integration.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test all framework module imports"""
    print("Testing framework module imports...")

    try:
        # Model imports
        from models.black_scholes import BlackScholesModel
        from models.heston import HestonModel
        from models.binomial_tree import BinomialTreeModel
        print("✓ Models module imports successful")

        # Risk imports
        from risk.var_models import VaRModel
        from risk.cvar_models import CVaRModel
        from risk.portfolio_risk import PortfolioRisk
        from risk.risk_measures import CoherentRiskMeasures
        print("✓ Risk module imports successful")

        # Data imports
        from data.market_data import YahooDataProvider, MockDataProvider
        from data.data_preprocessing import DataPreprocessor
        from data.sample_generators import SampleDataGenerator
        print("✓ Data module imports successful")

        # Evaluation imports
        from evaluation_modules.model_validation import ModelValidator, BacktestFramework, ValidationMetrics
        from evaluation_modules.performance_metrics import PerformanceAnalyzer, BenchmarkComparison, RiskAdjustedMetrics
        from evaluation_modules.statistical_tests import StatisticalTestSuite, GoodnessOfFitTests, ModelAdequacyTests
        print("✓ Evaluation module imports successful")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key classes"""
    print("\nTesting basic functionality...")

    try:
        # Import classes for testing
        from models.black_scholes import BlackScholesModel
        from risk.var_models import VaRModel
        from evaluation_modules.performance_metrics import PerformanceAnalyzer
        from evaluation_modules.statistical_tests import StatisticalTestSuite

        # Test Black-Scholes model
        from models.black_scholes import BSParameters
        bs_params = BSParameters(S0=100, K=105, T=0.25, r=0.05, sigma=0.2)
        bs_model = BlackScholesModel(bs_params)
        call_price = bs_model.call_price()
        put_price = bs_model.put_price()
        greeks = bs_model.greeks('call')
        print(f"✓ Black-Scholes model works - Call: ${call_price:.4f}, Put: ${put_price:.4f}")

        # Test VaR calculation
        returns = np.random.normal(0.001, 0.02, 252)  # Simulated daily returns
        var_model = VaRModel()
        var_95 = var_model.historical_var(returns, 0.05)
        print(f"✓ VaR model works - 95% VaR: {var_95:.4f}")

        # Test PerformanceAnalyzer
        returns_series = pd.Series(returns)
        analyzer = PerformanceAnalyzer(returns_series)
        metrics = analyzer.performance_summary()
        print(f"✓ PerformanceAnalyzer works - Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")

        # Test StatisticalTestSuite
        test_suite = StatisticalTestSuite()
        normality_results = test_suite.normality_tests(returns)
        print(f"✓ StatisticalTestSuite works - Normality tests completed: {len(normality_results)} tests")

        return True

    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def test_integration():
    """Test integration between different framework components"""
    print("\nTesting framework integration...")

    try:
        from models.black_scholes import BlackScholesModel
        from models.heston import HestonModel
        from risk.var_models import VaRModel
        from risk.portfolio_risk import PortfolioRisk
        from data.sample_generators import SampleDataGenerator
        from evaluation_modules.model_validation import BacktestFramework

        # Generate sample data
        generator = SampleDataGenerator()
        returns_data = {}
        for asset in ['STOCK1', 'STOCK2', 'STOCK3']:
            paths = generator.generate_gbm_paths(S0=100, mu=0.05, sigma=0.2, T=1.0, n_paths=1, n_steps=100)
            prices = paths[0]  # Take first path
            returns_data[asset] = np.diff(np.log(prices))

        returns_df = pd.DataFrame(returns_data)
        print(f"✓ Generated sample data: {returns_df.shape}")

        # Test portfolio risk calculation
        weights = np.array([0.4, 0.3, 0.3])
        portfolio_risk = PortfolioRisk(returns_df.values, weights)
        portfolio_vol = portfolio_risk.portfolio_volatility
        print(f"✓ Portfolio risk calculation: volatility = {portfolio_vol:.4f}")

        # Test model comparison
        from models.heston import HestonParameters
        from models.black_scholes import BSParameters
        bs_params = BSParameters(S0=100, K=105, T=0.25, r=0.05, sigma=0.2)
        bs_model = BlackScholesModel(bs_params)

        heston_params = HestonParameters(S0=100, K=105, T=0.25, r=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5)
        heston_model = HestonModel(heston_params)

        bs_price = bs_model.call_price()
        heston_price = heston_model.call_price()
        price_diff = abs(bs_price - heston_price)
        print(f"✓ Model comparison: BS=${bs_price:.4f}, Heston=${heston_price:.4f}, Diff=${price_diff:.4f}")

        # Test backtesting integration
        portfolio_returns = (returns_df * weights).sum(axis=1)
        var_model = VaRModel()

        backtest_framework = BacktestFramework()
        # Create simple mock risk model function
        def simple_var_model(data):
            return {'var_95': var_model.historical_var(data.values, 0.05)}

        backtest_results = backtest_framework.risk_model_backtest(
            simple_var_model,
            pd.DataFrame({'returns': portfolio_returns}),
            confidence_levels=[0.05]
        )
        print(f"✓ Backtesting integration successful")

        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("DERIVATIVES RISK MANAGEMENT FRAMEWORK - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    # Test imports
    import_success = test_imports()

    if import_success:
        # Test functionality
        func_success = test_basic_functionality()

        if func_success:
            # Test integration
            integration_success = test_integration()

            if integration_success:
                print("\n" + "=" * 70)
                print("✅ ALL TESTS PASSED - Framework is ready for use!")
                print("=" * 70)
                print("\n🚀 Next steps:")
                print("  1. Run 'python pipeline.py' for complete workflow demonstration")
                print("  2. Check examples/ directory for specific use cases")
                print("  3. Review config.yaml for customization options")
            else:
                print("\n" + "=" * 70)
                print("❌ INTEGRATION TESTS FAILED")
                print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("❌ FUNCTIONALITY TESTS FAILED")
            print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ IMPORT TESTS FAILED")
        print("=" * 70)

if __name__ == "__main__":
    main()