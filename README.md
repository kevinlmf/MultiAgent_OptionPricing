# Multi-Agent Options Pricing System

An options pricing and risk management system combining traditional models (Black-Scholes, Heston, SABR, Local Volatility) with a Multi-Agent framework that models market microstructure through participant behaviors.

---

## Quick Start

```bash
git clone https://github.com/kevinlmf/Options_Pricing
cd Options_Pricing


# Compare Multi-Agent vs traditional models on Bitcoin options (2-5 min)
python3 bitcoin/bitcoin_model_comparison.py

# Results saved to: ./bitcoin_comparison_results/
```

---

## Key Modules

**Pricing Models** (`models/`)
- Traditional: Black-Scholes, Heston, SABR, Local Volatility, Binomial Tree
- Multi-Agent: Market makers, arbitrageurs, noise traders with three-layer measures system

**Risk Management** (`risk/`)
- VaR/CVaR, Greeks, delta hedging, portfolio risk

**Evaluation** (`evaluation_modules/`)
- Backtesting, performance metrics, statistical tests, visualization

**Bitcoin Application** (`bitcoin/`)
- Deribit data fetching and model comparison

---

## Sample Results

| Model            | Total P&L ($)     | Sharpe Ratio | Win Rate (%) |
|------------------|-------------------|--------------|--------------|
| LOCAL_VOLATILITY | 1,196,477.39      | 154.609      | 0.0          |
| MULTI_AGENT      | 1,437,693.98      | 51.479       | 0.5          |
| SABR             | 1,560,628.32      | 39.439       | 1.3          |
| HESTON           | 30,830,765,806.67 | 0.400        | 64.7         |

---

## C++ Acceleration (Optional)

```bash
cd cpp_accelerators
pip install pybind11
make  # 5-10x speedup for Heston/SABR
```

---

## Installation

```bash
pip install numpy scipy pandas matplotlib seaborn requests
# Or: pip install -r bitcoin/requirements_bitcoin.txt
```

**Requirements:** Python 3.8+

---

## Usage Examples

**Model Calibration**
```python
from models.model_calibrator import ModelCalibrator
from models.heston import HestonModel

calibrator = ModelCalibrator(HestonModel())
params = calibrator.calibrate(market_data)
```

**Risk Analysis**
```python
from risk.delta_hedging import DeltaHedger
from risk.var_models import HistoricalVaR

hedger = DeltaHedger(option_portfolio)
hedged_pnl = hedger.run_strategy()

var_model = HistoricalVaR(confidence=0.95)
var = var_model.calculate(portfolio_returns)
```

**Multi-Agent Simulation**
```python
from models.multi_agent.agents import MarketMaker, Arbitrageur, NoiseTrader
from models.multi_agent.agents.agent_interaction import MarketEnvironment

env = MarketEnvironment(spot_price=50000, volatility=0.8)
env.add_agent(MarketMaker(inventory_limit=100))
env.add_agent(Arbitrageur(capital=1000000))
env.add_agent(NoiseTrader(trading_frequency=0.1))
results = env.simulate(n_periods=1000)
```

---

## Theoretical Framework

**Three-Layer Measures System:**
1. **Mathematical Measures** - Physical (P) and risk-neutral (Q) measures, Girsanov theorem
2. **Risk Measures** - VaR, CVaR, Greeks, coherent risk measures
3. **Market Microstructure** - Market maker behavior, arbitrage activities, noise trading

See `theory/` directory for detailed derivations.

---
“Beyond pricing — exploring how reason meets uncertainty, with light in every step.” 🌅
