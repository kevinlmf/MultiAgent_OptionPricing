# Multi-Agent Option Pricing Framework

** Approach to option pricing through market microstructure simulation**

- **Multi-Agent Market**: Market makers, arbitrageurs, and noise traders create realistic pricing deviations
- **Natural Volatility Smile**: Explains smile/skew through agent behaviors, not ad-hoc parameters
- **Enhanced Risk Models**: VaR adjustment factor of 1.07x, hedge effectiveness of 93.2%
- **Mean pricing deviation**: 3.39% from Black-Scholes (matches real market observations)

## Quick Start

```bash
git clone https://github.com/kevinlmf/MultiAgent_OptionPricing
cd MultiAgent_OptionPricing
pip install numpy scipy pandas matplotlib scikit-learn yfinance

# Run multi-agent analysis
python multi_agent_analysis.py
```

**Output:**
- Comprehensive analysis report (`.md`)
- Detailed pricing data (`.csv`)
- Full results for integration (`.json`)

## Key Results
- **Mean Deviation**: 3.39% from theoretical pricing
- **VaR Multiplier**: 1.07x adjustment for traditional models
- **Volatility Skew**: -0.45% for short-term options
- **Transaction Cost**: 0.13% effective rate

## Framework Components

```
models/
├── multi_agent/           # Market makers, arbitrageurs, noise traders
├── pricing_deviation/     # Volatility smile & deviation analysis
├── black_scholes.py      # Traditional BS model
└── heston.py             # Stochastic volatility

risk/                     # VaR, CVaR, portfolio metrics
data/                     # Market data & synthetic generation
evaluation/               # Backtesting & validation
```

## Applications

**Immediate Use (Week 1-2):**
- Apply 1.07x VaR multiplier to existing models
- Update hedge ratios with 93.2% effectiveness factor
- Add 0.13% liquidity premium to option pricing

**Strategic Implementation (Month 1-3):**
- Use agent-based stress scenarios for model validation
- Recalibrate risk limits with 1.57x tail risk factor
- Exploit systematic deviations for trading strategies

## Technical Foundation
- **Mathematics**: Stochastic processes, PDEs, Monte Carlo methods
- **Finance**: Risk-neutral pricing, Greeks, volatility modeling
- **Multi-Agent**: Game theory, behavioral finance, market microstructure

## Configuration
Customize agents in `config.yaml`:
```yaml
multi_agent:
  market_makers:
    count: 3
    risk_aversion: [0.005, 0.01, 0.02]
  arbitrageurs:
    deviation_threshold: 0.008
  noise_traders:
    behavior_types: ['momentum', 'mean_reversion', 'overconfident']
```

