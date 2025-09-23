# Derivatives Risk Management Framework

A **comprehensive quantitative finance platform** for derivatives pricing, risk management, and model validation, combining **pure mathematics**, **applied mathematics**, and **financial modeling** into a unified theoretical framework.

## Overview
- **Mathematical foundation**: Probability theory, stochastic processes, PDEs, optimization
- **Pricing models**: Black-Scholes, Heston stochastic volatility, Binomial trees
- **Risk management**: VaR, CVaR, portfolio risk metrics, coherent risk measures
- **Data infrastructure**: Market data providers, preprocessing, synthetic generation
- **Model validation**: Backtesting, statistical tests, performance analysis

## Theoretical Framework

The system follows a **three-layer mathematical progression**:

```
Pure Mathematics → Applied Mathematics → Financial Models
      ↓                    ↓                    ↓
- Probability Theory   - Stochastic Processes  - Black-Scholes
- Linear Algebra       - PDEs & Monte Carlo    - Heston Model
- Real Analysis        - Numerical Methods     - VaR/CVaR
- Optimization         - Parameter Estimation  - Risk Attribution
```

## Project Structure
```
Derivatives-Risk-Management/
├── theory/                    # Mathematical foundations & theoretical framework
│   ├── Theoretical_Framework.md    # Pure → Applied → Financial progression
│   └── Detailed_Methods.md         # Implementation details & formulas
├── models/                    # Pricing models & derivatives
│   ├── base_model.py             # Abstract base class
│   ├── black_scholes.py          # Black-Scholes implementation
│   ├── heston.py                 # Stochastic volatility model
│   ├── binomial_tree.py          # American option pricing
│   ├── implied_volatility.py     # Implied volatility calculations
│   └── option_portfolio.py       # Portfolio option management
├── risk/                      # Risk management & portfolio analysis
│   ├── var_models.py             # Value-at-Risk implementations
│   ├── cvar_models.py            # Conditional VaR & coherent measures
│   ├── portfolio_risk.py         # Portfolio-level risk metrics
│   ├── risk_measures.py          # General risk measure framework
│   ├── option_risk.py            # Options-specific risk metrics
│   └── integrated_risk.py        # Integrated risk management
├── data/                      # Data infrastructure & preprocessing
│   ├── market_data.py            # Market data providers
│   ├── data_preprocessing.py     # Cleaning & transformation
│   ├── sample_generators.py      # Synthetic data generation
│   └── sample_data/              # Sample data directory
├── evaluation_modules/        # Model validation & testing
│   ├── model_validation.py       # Backtesting & validation framework
│   ├── performance_metrics.py    # Performance & risk-adjusted metrics
│   └── statistical_tests.py      # Statistical testing suite
├── notebooks/                 # Jupyter notebooks for analysis
├── enhanced_pipeline_results/ # Pipeline output
│   ├── detailed_results.json     # Detailed JSON results
│   └── enhanced_pipeline_report.md # Enhanced analysis report
├── enhanced_pipeline.py       # Main end-to-end pipeline
├── test_framework.py          # Comprehensive test suite
└── config.yaml               # Configuration parameters
```

## Quick Start
```bash
git clone <repository-url>
cd Derivatives-Risk-Management

# Install dependencies
pip install numpy scipy pandas matplotlib scikit-learn statsmodels yfinance

# Test the framework
python test_framework.py

# Run the main pipeline
python enhanced_pipeline.py

# Or run individual components:

# Price options with Black-Scholes
python -c "
from models.black_scholes import BlackScholesModel
model = BlackScholesModel(S=100, K=105, T=0.25, r=0.05, sigma=0.2)
print(f'Call Price: {model.call_price():.2f}')
print(f'Put Price: {model.put_price():.2f}')
"

# Calculate portfolio VaR
python -c "
from risk.var_models import VaRModel
import numpy as np
returns = np.random.normal(0.001, 0.02, 252)
var_model = VaRModel()
var_95 = var_model.historical_var(returns, confidence_level=0.05)
print(f'95% VaR: {var_95:.4f}')
"
```

## Key Features

### **Mathematical Rigor**
- **Pure Mathematics**: Measure theory, functional analysis, optimization theory
- **Applied Mathematics**: Stochastic calculus, numerical PDEs, Monte Carlo methods
- **Financial Mathematics**: Risk-neutral pricing, martingale theory, volatility modeling

### **Pricing Models**
- **Black-Scholes**: European options with full Greeks calculation
- **Heston Model**: Stochastic volatility with characteristic function methods
- **Binomial Trees**: American option pricing with early exercise
- **Implied Volatility**: Advanced volatility surface calculations
- **Option Portfolio**: Multi-instrument portfolio management
- **Extensible Framework**: Abstract base classes for custom models

### **Risk Management**
- **Value-at-Risk**: Historical, parametric, and Monte Carlo methods
- **Conditional VaR**: Expected shortfall and coherent risk measures
- **Portfolio Risk**: Risk attribution, diversification benefits, stress testing
- **Option Risk**: Greeks-based risk analysis and sensitivity testing
- **Integrated Risk**: Unified risk management across asset classes
- **Backtesting**: Kupiec, Christoffersen tests for model validation

### **Performance Analysis**
- **Risk-adjusted Metrics**: Sharpe, Sortino, Calmar ratios
- **Benchmark Comparison**: Alpha, beta, tracking error, information ratio
- **Statistical Testing**: Normality, independence, homoscedasticity tests
- **Model Validation**: Cross-validation, out-of-sample testing

### **Data Infrastructure**
- **Market Data**: Yahoo Finance integration, mock providers for testing
- **Preprocessing**: Data quality validation, missing value handling
- **Synthetic Generation**: Realistic market scenarios for backtesting

## Use Cases

- **Quantitative Analysts** → Derivatives pricing & risk model development
- **Risk Managers** → Portfolio risk measurement & regulatory compliance
- **Researchers** → Academic research in quantitative finance
- **Financial Institutions** → Model validation & stress testing frameworks
- **Students** → Learning quantitative finance with practical implementations

## Mathematical Dependencies

The framework requires understanding of:
- **Probability Theory**: Measure theory, random variables, distributions
- **Stochastic Processes**: Brownian motion, Itô calculus, martingales
- **Partial Differential Equations**: Black-Scholes PDE, boundary conditions
- **Numerical Methods**: Finite differences, Monte Carlo, optimization
- **Statistics**: Hypothesis testing, parameter estimation, model selection

## Additional Tools

```bash
pip install jupyter notebook  # For interactive analysis in notebooks/
```

## Contributing

This framework follows rigorous mathematical principles. When contributing:
1. Ensure mathematical correctness and theoretical soundness
2. Include comprehensive documentation with mathematical derivations
3. Implement proper unit tests for numerical accuracy
4. Follow the Pure Math → Applied Math → Financial Models progression

---

*Bridging pure mathematics and practical finance through rigorous quantitative modeling.*