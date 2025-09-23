# Detailed Methods and Implementations

## Overview

This document provides detailed explanations of specific mathematical methods used in derivatives risk management, building upon the theoretical framework. Each method is presented with its mathematical foundation, implementation details, and practical applications.

---

## 1. Black-Scholes Model

### Mathematical Foundation

**Underlying Assumptions**:
1. Stock price follows geometric Brownian motion: dS = μSdt + σSdW
2. Constant risk-free rate r and volatility σ
3. No dividends (or constant dividend yield q)
4. European exercise only
5. No transaction costs or bid-ask spreads

**Derivation Chain**:
```
Geometric Brownian Motion → Itô's Lemma → Risk-Neutral Valuation → Black-Scholes PDE
```

**The Black-Scholes Partial Differential Equation**:
```
∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
```

**Analytical Solutions**:

*European Call Option*:
```
C = S₀e^(-qT)Φ(d₁) - Ke^(-rT)Φ(d₂)
```

*European Put Option*:
```
P = Ke^(-rT)Φ(-d₂) - S₀e^(-qT)Φ(-d₁)
```

Where:
```
d₁ = [ln(S₀/K) + (r - q + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
Φ(x) = standard normal CDF
```

### Greeks (Risk Sensitivities)

**Delta (Δ)** - Price sensitivity to underlying:
```
Call: Δ = e^(-qT)Φ(d₁)
Put:  Δ = e^(-qT)[Φ(d₁) - 1]
```

**Gamma (Γ)** - Delta sensitivity to underlying:
```
Γ = e^(-qT)φ(d₁)/(S₀σ√T)
```

**Vega (ν)** - Sensitivity to volatility:
```
ν = S₀e^(-qT)φ(d₁)√T
```

**Theta (Θ)** - Time decay:
```
Call: Θ = -S₀e^(-qT)φ(d₁)σ/(2√T) - rKe^(-rT)Φ(d₂) + qS₀e^(-qT)Φ(d₁)
Put:  Θ = -S₀e^(-qT)φ(d₁)σ/(2√T) + rKe^(-rT)Φ(-d₂) - qS₀e^(-qT)Φ(-d₁)
```

**Rho (ρ)** - Interest rate sensitivity:
```
Call: ρ = KTe^(-rT)Φ(d₂)
Put:  ρ = -KTe^(-rT)Φ(-d₂)
```

### Implementation Considerations

**Numerical Stability**:
- Handle extreme moneyness cases: S/K → 0 or S/K → ∞
- Use asymptotic expansions for very small or large time to expiry
- Implement robust normal CDF/PDF calculations

**Implied Volatility Calculation**:
Newton-Raphson iteration:
```
σₙ₊₁ = σₙ - [BS_Price(σₙ) - Market_Price] / Vega(σₙ)
```

**Extensions**:
- American options via binomial trees or finite difference methods
- Dividend adjustments for discrete dividends
- Currency options with foreign interest rate

---

## 2. Heston Stochastic Volatility Model

### Mathematical Framework

**Two-Factor Model**:
```
dS(t) = rS(t)dt + √v(t)S(t)dW₁(t)
dv(t) = κ(θ - v(t))dt + ξ√v(t)dW₂(t)
```

With correlation: `dW₁(t)dW₂(t) = ρdt`

**Parameters**:
- κ: mean reversion speed of variance
- θ: long-run variance level
- ξ: volatility of volatility
- ρ: correlation between price and volatility
- v₀: initial variance

**Feller Condition**: 2κθ > ξ² (ensures variance stays positive)

### Characteristic Function Method

**Heston Characteristic Function**:
```
φ(u,T) = exp(C(u,T) + D(u,T)v₀ + iu ln(S₀))
```

**Component Functions**:
```
d = √[(ρξui - κ)² + ξ²(ui + u²)]
g = (κ - ρξui - d)/(κ - ρξui + d)

C(u,T) = rTui + (κθ/ξ²)[(κ - ρξui - d)T - 2ln((1 - ge^(-dT))/(1 - g))]
D(u,T) = ((κ - ρξui - d)/ξ²) × ((1 - e^(-dT))/(1 - ge^(-dT)))
```

### Option Pricing via Fourier Inversion

**Call Option Formula**:
```
C = S₀e^(-qT)P₁ - Ke^(-rT)P₂
```

**Probability Integrals**:
```
Pⱼ = (1/2) + (1/π) ∫₀^∞ Re[e^(-iu ln K)φⱼ(u,T)/(iu)] du
```

Where φ₁ and φ₂ are modified characteristic functions.

### Numerical Implementation

**FFT-Based Pricing**:
1. Discretize the integration domain: u ∈ [0, U] with N points
2. Apply FFT to compute option prices for multiple strikes simultaneously
3. Use damping parameter α to ensure convergence

**Monte Carlo Simulation**:
Euler discretization with full truncation:
```
Sᵢ₊₁ = Sᵢ exp((r - q - vᵢ/2)Δt + √(vᵢΔt)Z₁,ᵢ₊₁)
vᵢ₊₁ = vᵢ + κ(θ - vᵢ)Δt + ξ√(max(vᵢ,0)Δt)Z₂,ᵢ₊₁
```

With correlated random numbers:
```
Z₂ = ρZ₁ + √(1-ρ²)Z̃₂
```

### Model Calibration

**Objective Function**:
Minimize sum of squared relative errors:
```
f(Θ) = Σᵢ [(V_market^i - V_model^i(Θ))/V_market^i]²
```

**Parameter Constraints**:
- v₀ > 0: initial variance positive
- κ > 0: positive mean reversion
- θ > 0: positive long-run variance
- ξ > 0: positive vol-of-vol
- |ρ| < 1: correlation bounds
- 2κθ > ξ²: Feller condition

**Optimization Methods**:
- Global optimization: Differential Evolution, Particle Swarm
- Local refinement: L-BFGS-B with parameter bounds
- Multi-start approach to avoid local minima

---

## 3. Value at Risk (VaR) and Risk Measures

### Mathematical Definition

**Value at Risk**:
```
VaRₐ(X) = inf{x ∈ ℝ : P(X ≤ x) ≥ α}
```

**Conditional Value at Risk (Expected Shortfall)**:
```
CVaRₐ(X) = E[X | X ≤ VaRₐ(X)]
```

### Calculation Methods

**1. Historical Simulation**:
```
VaR₅% = 5th percentile of historical P&L distribution
```

Steps:
1. Collect historical asset returns: r₁, r₂, ..., rₙ
2. Apply current portfolio weights: P&Lᵢ = wᵀrᵢ
3. Sort P&L values: P&L₍₁₎ ≤ P&L₍₂₎ ≤ ... ≤ P&L₍ₙ₎
4. VaRₐ = P&L₍⌊αn⌋₎

**2. Parametric Method (Variance-Covariance)**:
Assume normal distribution: P&L ~ N(μ, σ²)
```
VaRₐ = μ + σΦ⁻¹(α)
```

For portfolios: σ² = wᵀΣw where Σ is covariance matrix

**3. Monte Carlo Simulation**:
```
1. Simulate asset price paths: S₁, S₂, ..., Sₙ
2. Calculate portfolio values: V₁, V₂, ..., Vₙ
3. Compute P&L: P&Lᵢ = Vᵢ - V₀
4. VaRₐ = αth quantile of {P&L₁, ..., P&Lₙ}
```

### Coherent Risk Measures

A risk measure ρ is coherent if it satisfies:

**1. Monotonicity**: X ≤ Y ⟹ ρ(X) ≥ ρ(Y)
**2. Translation Invariance**: ρ(X + c) = ρ(X) - c
**3. Positive Homogeneity**: ρ(λX) = λρ(X) for λ > 0
**4. Sub-additivity**: ρ(X + Y) ≤ ρ(X) + ρ(Y)

**Note**: VaR fails sub-additivity; CVaR is coherent.

### Implementation Considerations

**Backtesting**:
Kupiec test for VaR model validation:
```
H₀: Violation rate = α
Test statistic: LR = -2ln[L(p̂)/L(α)]
where p̂ = number of violations / number of observations
```

**Stress Testing**:
- Scenario analysis with extreme market conditions
- Factor shock tests
- Historical scenario replications

---

## 4. Monte Carlo Methods

### Basic Theory

**Law of Large Numbers**:
```
(1/N)Σᵢ₌₁ᴺ f(Xᵢ) → E[f(X)] as N → ∞
```

**Central Limit Theorem**:
```
√N[(1/N)Σf(Xᵢ) - E[f(X)]] → N(0, Var[f(X)])
```

**Convergence Rate**: O(1/√N) - independent of dimension

### Path Generation Methods

**1. Euler-Maruyama Scheme**:
For SDE: dX = μ(X,t)dt + σ(X,t)dW
```
Xₜ₊Δₜ = Xₜ + μ(Xₜ,t)Δt + σ(Xₜ,t)√Δt Z
```

**2. Milstein Scheme** (higher order):
```
Xₜ₊Δₜ = Xₜ + μΔt + σ√Δt Z + (1/2)σσ'Δt(Z² - 1)
```

**3. Exact Simulation** (when available):
For geometric Brownian motion:
```
S(T) = S₀ exp((r - σ²/2)T + σ√T Z)
```

### Variance Reduction Techniques

**1. Antithetic Variates**:
```
Use both Z and -Z in simulation
Estimator: [f(Z) + f(-Z)]/2
Variance reduction when f is monotonic
```

**2. Control Variates**:
```
f̃ = f(X) - β[g(X) - E[g(X)]]
Choose β to minimize Var[f̃]
Optimal: β* = Cov[f(X),g(X)]/Var[g(X)]
```

**3. Importance Sampling**:
```
E[f(X)] = ∫ f(x)p(x)dx = ∫ f(x)[p(x)/q(x)]q(x)dx
Estimator: (1/N)Σf(Yᵢ)[p(Yᵢ)/q(Yᵢ)] where Yᵢ ~ q
```

**4. Stratified Sampling**:
Divide [0,1] into k strata, sample within each:
```
Estimator: Σₖ(nₖ/N)X̄ₖ where nₖ = stratum k sample size
```

### Multi-Dimensional Simulation

**Cholesky Decomposition** for correlated normals:
```
If Σ = LLᵀ (Cholesky decomposition)
Then X = μ + LZ gives X ~ N(μ, Σ)
```

**Principal Component Analysis**:
```
Σ = QΛQᵀ (eigendecomposition)
Generate Y ~ N(0, Λ), set X = μ + QY
```

### Quasi-Monte Carlo

**Low-Discrepancy Sequences**:
- Sobol sequences
- Halton sequences
- Latin hypercube sampling

**Convergence Rate**: O((log N)ᵈ/N) where d = dimension

Better than MC for smooth integrands in moderate dimensions.

---

## 5. Numerical Methods for PDEs

### Finite Difference Methods

**Grid Setup**:
```
Space: S ∈ [0, Sₘₐₓ] with ΔS = Sₘₐₓ/M
Time: t ∈ [0, T] with Δt = T/N
Grid points: (iΔS, jΔt) for i = 0,...,M; j = 0,...,N
```

**Discrete Operators**:
```
∂V/∂S ≈ (Vᵢ₊₁ʲ - Vᵢ₋₁ʲ)/(2ΔS)  (central difference)
∂²V/∂S² ≈ (Vᵢ₊₁ʲ - 2Vᵢʲ + Vᵢ₋₁ʲ)/(ΔS)²  (second difference)
∂V/∂t ≈ (Vᵢʲ⁺¹ - Vᵢʲ)/Δt  (forward difference)
```

**Explicit Scheme** (Forward Euler):
```
Vᵢʲ⁺¹ = Vᵢʲ + Δt[½σ²i²ΔS²(Vᵢ₊₁ʲ - 2Vᵢʲ + Vᵢ₋₁ʲ)/(ΔS)² +
         riΔS(Vᵢ₊₁ʲ - Vᵢ₋₁ʲ)/(2ΔS) - rVᵢʲ]
```

Stability condition: Δt ≤ (ΔS)²/(σ²Sₜₒₚ²)

**Implicit Scheme** (Backward Euler):
```
Vᵢʲ = Vᵢʲ⁺¹ + Δt[LVᵢʲ⁺¹]
```
Requires solving tridiagonal system, but unconditionally stable.

**Crank-Nicolson Scheme**:
```
Vᵢʲ⁺¹ - Vᵢʲ = (Δt/2)[LVᵢʲ + LVᵢʲ⁺¹]
```
Second-order accurate in time, unconditionally stable.

### Boundary Conditions

**Far-field Boundaries**:
- S → 0: V ≈ discounted intrinsic value
- S → ∞: V ≈ S - Ke^(-r(T-t)) (call); V ≈ 0 (put)

**Free Boundary Problems** (American options):
Use penalty methods or linear complementarity formulation.

---

## 6. Model Calibration and Parameter Estimation

### Maximum Likelihood Estimation

**Log-Likelihood for Geometric Brownian Motion**:
Given observations S₀, S₁, ..., Sₙ at times 0, Δt, 2Δt, ..., nΔt

```
ℓ(μ, σ) = -½n ln(2πσ²Δt) - (1/(2σ²Δt))Σᵢ₌₁ⁿ[ln(Sᵢ/Sᵢ₋₁) - (μ - σ²/2)Δt]²
```

**MLE Solutions**:
```
μ̂ = (1/(nΔt))ln(Sₙ/S₀) + σ̂²/2
σ̂² = (1/(nΔt))Σᵢ₌₁ⁿ[ln(Sᵢ/Sᵢ₋₁) - μ̂Δt + σ̂²Δt/2]²
```

### Method of Moments

Match sample moments to theoretical moments:
```
Sample mean = Theoretical mean
Sample variance = Theoretical variance
...
```

For GBM with observations rᵢ = ln(Sᵢ/Sᵢ₋₁):
```
μ̂ = r̄/Δt + σ̂²/2
σ̂² = s²ᵣ/Δt
```

### Implied Parameter Calibration

**Objective Function**:
```
min Σᵢ wᵢ[V_model(Kᵢ, Tᵢ; θ) - V_market(Kᵢ, Tᵢ)]²
```

Common weights: wᵢ = 1/V_market(Kᵢ, Tᵢ) (relative error)

**Optimization Challenges**:
- Non-convex objective function
- Multiple local minima
- Parameter constraints
- Ill-conditioning near parameter boundaries

**Regularization Techniques**:
```
Objective = Fit Error + λ × Penalty(θ)
```
Where Penalty(θ) prevents extreme parameter values.

### Implied Volatility Surface

**Smile/Skew Phenomena**:
- At-the-money: benchmark volatility
- Out-of-the-money puts: higher implied vol (crash fear)
- Out-of-the-money calls: varying patterns

**Parametric Models**:
1. **SVI (Stochastic Volatility Inspired)**:
   ```
   σ²ᵢᵥ(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]
   ```

2. **SABR Model**:
   ```
   σᵢᵥ(K,T) ≈ (α/f̃^(1-β))[1 + ((1-β)²α²)/(24f̃^(2-2β))T + ...]
   ```

**Arbitrage-Free Conditions**:
- Calendar arbitrage: ∂σᵢᵥ/∂T ≥ constraints
- Butterfly arbitrage: ∂²C/∂K² ≥ 0

---

## Implementation Best Practices

### Code Structure
1. **Modular Design**: Separate mathematical components
2. **Parameter Validation**: Check bounds and conditions
3. **Numerical Stability**: Handle edge cases gracefully
4. **Performance Optimization**: Vectorization, caching
5. **Testing**: Unit tests for each mathematical component

### Common Pitfalls
1. **Numerical Overflow**: Use log-space calculations when possible
2. **Underflow in PDFs**: Implement robust normal PDF/CDF
3. **Parameter Bounds**: Enforce constraints during optimization
4. **Convergence Criteria**: Set appropriate tolerances
5. **Market Data Quality**: Clean and validate inputs

### Validation Methods
1. **Analytical Benchmarks**: Compare to known solutions
2. **Convergence Testing**: Verify O(h²) or O(1/√N) rates
3. **Cross-Validation**: Out-of-sample testing
4. **Stress Testing**: Extreme parameter regimes
5. **Market Comparison**: Validate against observed prices

---

This detailed methods document provides the mathematical rigor and implementation guidance necessary for robust derivatives risk management systems. Each method builds upon the theoretical framework while providing practical insights for real-world applications.