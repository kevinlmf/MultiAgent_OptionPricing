# Multi-Agent Options Pricing System

## Project Overview

A comprehensive options pricing and risk management system that integrates traditional financial engineering models (Black-Scholes, Heston, SABR, Local Volatility) with an innovative Multi-Agent modeling framework. The project provides complete functionality for pricing, risk assessment, model calibration, backtesting, and visualization.

**Core Innovation:** The Multi-Agent model understands options market microstructure through modeling market participant behaviors (market makers, arbitrageurs, noise traders), providing pricing mechanisms closer to market reality.

---

## Quick Start

### Bitcoin Options Model Comparison (Recommended for Beginners)

Compare the profitability of Multi-Agent model versus traditional stochastic volatility models in Bitcoin options trading:

```bash
# Navigate to project directory
cd "/Users/mengfanlong/Downloads/System/Quant/Options Pricing"

# Run Bitcoin options model comparison
python3 bitcoin/bitcoin_model_comparison.py
```

**Runtime:** 2-5 minutes
**Results saved in:** `./bitcoin_comparison_results/`

### Multi-Agent Market Simulation Demo

```bash
# Run three-layer measures demo (market microstructure analysis)
python3 models/multi_agent/demos/three_layer_demo.py
```

---

## Main Feature Modules

### 1. Pricing Models (models/)

#### Traditional Models
- **Black-Scholes** - Classic options pricing formula
- **Heston** - Stochastic volatility model (pricing via characteristic function)
- **SABR** - Stochastic alpha-beta-rho model (volatility smile calibration)
- **Local Volatility** - Local volatility model (Dupire formula)
- **Binomial Tree** - Binomial tree model (American options pricing)

#### Multi-Agent Framework (models/multi_agent/)
- **agents/** - Market participant modeling
  - `market_maker.py` - Market maker (provides liquidity, manages inventory)
  - `arbitrageur.py` - Arbitrageur (eliminates market inconsistencies)
  - `noise_trader.py` - Noise trader (introduces market volatility)
  - `agent_interaction.py` - Agent interaction mechanisms
- **models/** - Three-layer measures system
  - `three_layer_measures.py` - Mathematical measures, risk measures, market microstructure measures
- **demos/** - Demo programs
  - `three_layer_demo.py` - Complete market simulation demo

### 2. Risk Management (risk/)
- `risk_measures.py` - VaR, CVaR, Greeks
- `var_models.py` - VaR models (historical simulation, parametric, Monte Carlo)
- `cvar_models.py` - CVaR models and coherent risk measures
- `option_risk.py` - Options risk analysis
- `portfolio_risk.py` - Portfolio risk
- `delta_hedging.py` - Delta hedging strategies
- `integrated_risk.py` - Integrated risk management framework

### 3. Data Module (data/)
- `market_data.py` - Market data acquisition and processing
- `data_preprocessing.py` - Data cleaning and preprocessing
- `sample_generators.py` - Simulation data generators

### 4. Evaluation Modules (evaluation_modules/)
- `model_validation.py` - Model validation and backtesting
- `performance_metrics.py` - Performance metrics calculation
- `statistical_tests.py` - Statistical tests
- `trading_backtest.py` - Trading backtest framework
- `visualization.py` - Visualization tools

### 5. Bitcoin Options Application (bitcoin/)
- `bitcoin_data_fetcher.py` - Deribit market data acquisition
- `bitcoin_model_comparison.py` - Model comparison experiments

---

## Example Experimental Results

### Bitcoin Options Model Comparison Results

After running, the following will be generated:

```
bitcoin_comparison_results/
├── pnl_comparison.png              # P&L comparison chart
├── risk_metrics.png                # Risk metrics
├── return_distributions.png        # Return distributions
├── model_comparison_summary.csv    # Data table
└── model_comparison_report.md      # Complete report
```

**Typical Output:**
```
Model              Total P&L ($)  Sharpe Ratio  Win Rate (%)  Max Drawdown
MULTI_AGENT             245.67         2.34          67.5         -8.2%
HESTON                  198.42         1.89          62.1        -12.5%
SABR                    176.23         1.72          58.9        -15.3%
LOCAL_VOLATILITY        142.56         1.45          55.2        -18.7%
```

**Key Metrics Interpretation:**
- **Total P&L** - Net profit (higher is better)
- **Sharpe Ratio** - Risk-adjusted returns (>2.0 is excellent)
- **Win Rate** - Percentage of profitable trades (>60% is excellent)
- **Max Drawdown** - Maximum loss from peak (smaller absolute value is better)

---

## Complete Project Structure

```
Options Pricing/
├── bitcoin/                      # Bitcoin options application
│   ├── bitcoin_data_fetcher.py
│   └── bitcoin_model_comparison.py
├── models/                       # Pricing models
│   ├── black_scholes.py
│   ├── heston.py
│   ├── sabr.py
│   ├── local_volatility.py
│   ├── binomial_tree.py
│   ├── model_calibrator.py
│   ├── implied_volatility.py
│   ├── option_portfolio.py
│   └── multi_agent/             # Multi-Agent framework
│       ├── agents/              # Market participants
│       │   ├── base_agent.py
│       │   ├── market_maker.py
│       │   ├── arbitrageur.py
│       │   ├── noise_trader.py
│       │   └── agent_interaction.py
│       ├── models/              # Three-layer measures system
│       │   └── three_layer_measures.py
│       └── demos/               # Demo programs
│           └── three_layer_demo.py
├── risk/                        # Risk management
│   ├── risk_measures.py
│   ├── var_models.py
│   ├── cvar_models.py
│   ├── option_risk.py
│   ├── portfolio_risk.py
│   ├── delta_hedging.py
│   └── integrated_risk.py
├── data/                        # Data processing
│   ├── market_data.py
│   ├── data_preprocessing.py
│   └── sample_generators.py
├── evaluation_modules/          # Evaluation and backtesting
│   ├── model_validation.py
│   ├── performance_metrics.py
│   ├── statistical_tests.py
│   ├── trading_backtest.py
│   └── visualization.py
└── theory/                      # Theoretical documentation
    ├── Theoretical_Framework.md
    └── Detailed_Methods.md
```

---

## Advanced Usage

### 1. Using Real Market Data

```bash
# Use real Deribit data (requires API access)
python3 bitcoin/bitcoin_model_comparison.py --real-data

# Custom market parameters
python3 bitcoin/bitcoin_model_comparison.py --spot-price 45000 --risk-free-rate 0.05
```

### 2. Model Calibration

```python
from models.model_calibrator import ModelCalibrator
from models.heston import HestonModel

# Load market data
market_data = load_option_data()

# Calibrate Heston model
calibrator = ModelCalibrator(HestonModel())
params = calibrator.calibrate(market_data)
```

### 3. Risk Analysis

```python
from risk.delta_hedging import DeltaHedger
from risk.var_models import HistoricalVaR

# Delta hedging
hedger = DeltaHedger(option_portfolio)
hedged_pnl = hedger.run_strategy()

# VaR calculation
var_model = HistoricalVaR(confidence=0.95)
var = var_model.calculate(portfolio_returns)
```

### 4. Multi-Agent Simulation

```python
from models.multi_agent.agents import MarketMaker, Arbitrageur, NoiseTrader
from models.multi_agent.agents.agent_interaction import MarketEnvironment

# Create market environment
env = MarketEnvironment(spot_price=50000, volatility=0.8)

# Add market participants
env.add_agent(MarketMaker(inventory_limit=100))
env.add_agent(Arbitrageur(capital=1000000))
env.add_agent(NoiseTrader(trading_frequency=0.1))

# Run simulation
results = env.simulate(n_periods=1000)
```

---

## Installation and Dependencies

### Environment Requirements
- Python 3.8+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn (visualization)
- Requests (data fetching)

### Installing Dependencies

```bash
# Basic dependencies
pip install numpy scipy pandas matplotlib seaborn

# Additional dependencies for Bitcoin module
pip install requests

# Or use requirements file
pip install -r bitcoin/requirements_bitcoin.txt
```

---

## Common Issues

**Q: Module import errors?**

Ensure you run from the project root directory and PYTHONPATH is correct:
```bash
cd "/Users/mengfanlong/Downloads/System/Quant/Options Pricing"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Q: Slow execution?**

- This is normal, model calibration involves numerical optimization (2-5 minutes)
- Heston model characteristic function integration is time-consuming
- You can run simple tests first to verify the environment

**Q: Numerical instability?**

- Check parameter ranges (volatility > 0, reasonable interest rates, etc.)
- Verify Feller condition (Heston model: 2κθ > ξ²)
- Adjust numerical integration parameters

---

## Theoretical Background

### Mathematical Framework

The project is based on a complete financial mathematics theoretical system. Detailed documentation in the `theory/` directory:

1. **Detailed_Methods.md** - Detailed method derivations
   - Measure theory foundations (physical measure P and risk-neutral measure Q)
   - Black-Scholes model and PDE derivation
   - Heston model and characteristic function methods
   - VaR/CVaR and coherent risk measures
   - Monte Carlo methods and variance reduction techniques
   - PDE numerical methods (finite differences)
   - Model calibration and parameter estimation

2. **Theoretical_Framework.md** - Theoretical framework overview

### Three-Layer Measures System of Multi-Agent Model

The core innovation of this project is the **Three-Layer Measures** framework:

#### Layer 1: Mathematical Measures
- **Physical Measure P**: Describes real-world price dynamics
- **Risk-Neutral Measure Q**: Theoretical foundation for arbitrage-free pricing
- **Girsanov Theorem**: Mathematical tool for measure transformations

#### Layer 2: Risk Measures
- **VaR/CVaR**: Market risk quantification
- **Greeks**: Option risk sensitivity (Delta, Gamma, Vega, Theta, Rho)
- **Coherent Risk Measures**: Satisfying monotonicity, translation invariance, positive homogeneity, subadditivity

#### Layer 3: Market Microstructure Measures
- **Market Maker Behavior**: Inventory management, bid-ask spread optimization
- **Arbitrage Activities**: Price convergence mechanisms, arbitrage-free boundaries
- **Noise Trading**: Source of market volatility, liquidity provision

These three layers interact to form a complete options pricing and risk management system.

---

## Research Value and Applications

### Core Research Question

> **Can the Multi-Agent model provide better pricing accuracy and trading performance than traditional models?**

Validated through a complete empirical research process:

1. **Model Calibration** - Calibrate model parameters using market data
2. **Pricing Comparison** - Compare model prices with market prices
3. **Trading Strategies** - Construct trading strategies based on model pricing
4. **Risk Management** - Implement Delta hedging strategies
5. **Performance Evaluation** - Calculate P&L, Sharpe Ratio, maximum drawdown, etc.

### Practical Application Scenarios

- **Quantitative Trading**: Options arbitrage strategy development
- **Risk Management**: Options portfolio risk assessment for financial institutions
- **Product Design**: Structured product pricing and hedging
- **Academic Research**: Market microstructure and behavioral finance
- **Cryptocurrency**: Bitcoin/Ethereum options market analysis

---

## Contribution and Extension

### Extensible Directions

1. **New Pricing Models**
   - Jump-diffusion models (Merton, Kou)
   - Rough volatility models
   - Levy process models

2. **Enhanced Multi-Agent Framework**
   - More complex agent behaviors (learning, adaptation)
   - Market impact models
   - Order book dynamics

3. **More Market Applications**
   - Equity options
   - Foreign exchange options
   - Commodity options

4. **Machine Learning Integration**
   - Deep hedging
   - Reinforcement learning strategies
   - GAN-generated volatility surfaces

---

## Project Features

**Completeness** - Complete chain from theory to implementation
**Modularity** - Clear code architecture, easy to extend
**Reproducibility** - Detailed documentation and example code
**Practicality** - Can be directly applied to real market data
**Educational Value** - Suitable for learning financial engineering and quantitative trading

---

## Quick Reference

**Run complete experiment:**
```bash
cd "/Users/mengfanlong/Downloads/System/Quant/Options Pricing"
python3 bitcoin/bitcoin_model_comparison.py
```

**Run market simulation:**
```bash
python3 models/multi_agent/demos/three_layer_demo.py
```

**View results:**
```bash
open bitcoin_comparison_results/
```

---

## License

MIT License - Free for academic and commercial use

---

**Last Updated:** 2025-10-14
**Project Status:** Stable version, actively maintained
