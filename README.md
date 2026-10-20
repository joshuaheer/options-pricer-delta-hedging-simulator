# Options Pricing & Delta-Hedging Lab

A Python library that prices European and American options three ways, solves for implied volatility from market quotes, and then asks the question a trader actually cares about: **if I sell an option and hedge it, how much money do I make or lose, and why?**

The answer comes from simulation and is checked against theory at every step.

![Hedging error vs rebalancing frequency](figures/04_hedge_frequency.png)

## Headline findings

| Question | Result |
|---|---|
| Do the three pricers agree? | Binomial tree error falls like 1/N (fitted slope −1.00); Monte Carlo error like 1/√N (slope −0.49). A control variate cuts Monte Carlo variance **29×**. |
| How much does discrete hedging cost in risk? | Std of hedging P&L falls like 1/√N (slope −0.49) and matches the Derman-Kamal formula to within 3% for N ≥ 42. Hedging daily instead of weekly cuts risk from $0.96 to $0.44 on a $3.99 option. |
| What drives the P&L path by path? | The theta-gamma identity ½ΓS²(σ²_implied − σ²_realised)dt explains the simulated P&L with correlation **0.98**. |
| What if realised vol ≠ implied vol? | The seller earns BS(σ_implied) − BS(σ_realised) on average, whichever vol is used to hedge. Hedging at the realised vol makes it near-certain (std $0.11 at 10% realised); hedging at implied vol leaves it path-dependent (std $0.51). |
| With transaction costs, how often should you hedge? | Total risk is U-shaped in N. Best frequency drops from 504 to 126 rebalances (per 3 months) as costs rise from 2 to 10 bp. Pricing at the Leland-adjusted vol removes 70–85% of the cost drag; what remains is the cost of opening and closing the hedge, which Leland's formula doesn't include. |

Full numbers for every experiment are in [RESULTS.md](RESULTS.md).

## What's inside

```
optlab/
  black_scholes.py   closed-form prices and Greeks (delta, gamma, vega, theta, rho), vectorised
  binomial.py        Cox-Ross-Rubinstein tree, European and American, Greeks read off the tree
  monte_carlo.py     Monte Carlo with antithetic variates and a control variate, with standard errors
  greeks.py          finite-difference Greeks for any pricer (used to check the closed forms)
  implied_vol.py     safeguarded Newton-Raphson solver with no-arbitrage checks
  market_data.py     option-chain download (Yahoo Finance), cleaning, put-call-parity forward, surface
  paths.py           exact geometric Brownian motion simulation
  hedging.py         discrete delta-hedging simulator with rates, dividends and transaction costs
experiments/         six scripts, one per question; each saves a figure in figures/
tests/               26 tests checking the code against known results and theory
```

## Running it

```bash
pip install -r requirements.txt
python -m experiments.run_all             # all six experiments, ~1 minute; writes RESULTS.md
python -m experiments.run_all --live SPY  # same, but build the vol surface from today's SPY options
pytest                                    # run the tests
```

Each experiment also runs on its own, e.g. `python -m experiments.vol_mismatch`.

> **About the vol-surface figure.** The committed figure uses a synthetic option chain generated from a known smile, so the solver's accuracy can be measured: it recovers the true vols from noisy quotes to within 0.09 vol points. Run with `--live SPY` to replace it with real market data; the downloaded chain is saved to `data/` so the result is reproducible.

## The experiments

### 1. Pricer convergence
![](figures/01_convergence.png)

The raw binomial price zig-zags because the strike sits in a different place relative to the terminal nodes for odd and even N. Averaging the N- and N+1-step prices removes the zig-zag. Monte Carlo variance reduction moves the error line down without changing its slope: you still need 4× the paths to halve the error, but from a much lower starting point.

### 2. Greeks
![](figures/02_greeks.png)

Analytic Greeks agree with finite differences to 2×10⁻⁶ (relative). As expiry approaches, delta becomes a step and gamma spikes at the strike. That makes short-dated at-the-money options the hardest to hedge, and it feeds directly into experiment 4.

One gotcha found along the way: bumping the stock price and rebuilding a binomial tree gives a badly wrong gamma (0.15 vs 0.02), because the bump shifts every node relative to the strike. `crr_greeks` reads delta and gamma off nodes the tree already has instead.

### 3. Implied volatility surface
![](figures/03_iv_surface.png)

The pipeline does what a desk would do with raw quotes:
1. Drop options with no bid, crossed quotes or spreads wider than 25% of mid.
2. Infer the forward price and discount factor for each expiry by regressing C − P on K (put-call parity: C − P = DF·(F − K)). This removes the need to guess the interest rate or dividend yield.
3. Use out-of-the-money options only, since in-the-money quotes are mostly intrinsic value plus noise.
4. Solve for implied vol with a safeguarded Newton method: Newton steps when they stay inside a bracket, bisection when they don't. It never fails, and it returns NaN instead of a misleading number when a price breaks no-arbitrage bounds.

### 4. Hedging frequency
Sell a 3-month at-the-money call at 20% vol, with the stock genuinely moving at 20% vol, and delta-hedge N times. The option is fairly priced, so all P&L is hedging error. Every frequency runs on the same 20,000 paths.

Over each step, a delta-hedged short option collects theta and pays for gamma:

  dP&L ≈ ½ Γ S² (σ²_implied · dt − (ΔS/S)²)

With continuous hedging the bracket averages to zero. With N steps, each squared return is a random draw around σ²dt, which leaves an error whose std scales like 1/√N. Derman & Kamal's approximation, √(π/4) · vega · σ / √N, matches the simulation to within 3% once N ≥ 42; with very few rebalances the approximation overstates the error a little (7% at N = 6).

### 5. Realised vs implied volatility
![](figures/05_vol_mismatch.png)

This experiment holds the key idea of options trading: **you are paid for the volatility you sell and you pay for the volatility that turns up.** The average P&L sits exactly on BS(20%) − BS(σ_realised) whichever delta is used. The choice of hedge changes the *risk*:
- Delta at the **realised** vol locks in the price difference almost exactly, but you can't know realised vol in advance.
- Delta at the **implied** vol (what you can observe) gives a gamma-weighted P&L. You make the most when the stock stays near the strike, where gamma is high, so the same realised vol can produce very different outcomes.

### 6. Transaction costs
![](figures/06_transaction_costs.png)

Each rebalance trades about Γ·S·|ΔS| shares, and |ΔS| ~ σS√dt, so expected rebalancing cost grows like √N while hedging error falls like 1/√N. The sum is U-shaped, and the optimum moves to less frequent hedging as costs rise.

Leland's fix is to price and hedge at an adjusted vol, σ²_L = σ²(1 + √(8/π) · k / (σ√dt)), where k is the one-way cost. At 5 bp and 252 rebalances, σ_L = 21.2%, and the mean loss shrinks from −$0.30 to −$0.05. The remaining −$0.05 is almost exactly the cost of buying the initial hedge and selling it at expiry (≈ 1 × cost × S), which Leland's formula leaves out by design.

## Things the tests check

The tests compare against results that are known independently of this code:
- Black-Scholes textbook values (Hull) and put-call parity.
- Binomial and Monte Carlo prices converge to Black-Scholes; an American call with no dividends equals the European.
- Analytic Greeks match finite differences; tree Greeks match analytic.
- Implied vol round-trips across moneyness, maturity and vol levels; arbitrage-violating prices are rejected.
- Put-call-parity regression recovers the forward and discount factor that generated the quotes.
- Fair option → zero mean hedging P&L; 4× rebalancing → half the error; the vol-gap P&L; Leland's adjustment.
- **Drift has only a second-order effect.** A real-world drift μ adds μ²dt² to each squared return, which predicts a small loss of ½μ²·dt·vega/σ. The test checks the simulated bias against that prediction (≈ −$0.018 at μ = ±30% with daily hedging).

## Limitations and next steps

- The stock follows geometric Brownian motion with constant vol, so there are no jumps and no stochastic volatility. Adding Merton jumps or a Heston model would show hedging failures that more rebalancing can't fix.
- Delta hedging only. Adding a second option to hedge gamma or vega is the natural next step.
- Live data from Yahoo Finance is delayed and has wide spreads on illiquid strikes; the cleaning step handles most of this, but a fitted parametric surface (e.g. SVI) would smooth the rest.
- A backtest on historical prices (sell at implied, hedge with realised moves) would test the variance-risk-premium idea from experiment 5 on real data.

## References

- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*.
- Cox, J., Ross, S. & Rubinstein, M. (1979). Option Pricing: A Simplified Approach. *Journal of Financial Economics*.
- Leland, H. (1985). Option Pricing and Replication with Transactions Costs. *Journal of Finance*.
- Derman, E. & Kamal, M. (1999). When You Cannot Hedge Continuously: The Corrections of Black-Scholes. *Risk*.
- Ahmad, R. & Wilmott, P. (2005). Which Free Lunch Would You Like Today, Sir? Delta Hedging, Volatility Arbitrage and Optimal Portfolios. *Wilmott Magazine*.
- Hull, J. *Options, Futures, and Other Derivatives*.
