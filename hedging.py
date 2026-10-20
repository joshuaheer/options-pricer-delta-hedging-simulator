"""
Discrete delta-hedging simulator.

Set-up
------
A trader sells one European option at time 0 for the Black-Scholes price at the
*implied* vol sigma_i, then hedges by holding delta shares, rebalancing N times
before expiry. The stock actually moves with the *realised* vol sigma_r, which
the trader does not know in advance.

Cash account: starts at premium - delta_0 * S_0, earns the risk-free rate, pays
for every rebalance, collects dividends on the shares held, and pays
proportional transaction costs (cost * |shares traded| * price). At expiry the
shares are sold and the option payoff is paid out:

    P&L (short) = cash_T + delta_{N-1} S_T - payoff(S_T) - costs

Why the P&L is what it is
-------------------------
Over one small step, a delta-hedged short option earns theta and loses gamma:

    dP&L ~ 0.5 * Gamma * S^2 * (sigma_i^2 dt - (dS/S)^2)

The first term is the option decay the seller collects (priced at sigma_i); the
second is what the realised move costs. Summed over the life of the option:

    P&L ~ sum_k  0.5 * Gamma_k * S_k^2 * (sigma_i^2 - realised variance over step k) * dt

So the seller profits when realised variance is below implied variance, and the
profit is weighted by gamma -- it depends on *where* the stock went, not just on
how volatile it was. `gamma_pnl` in the result computes this sum path by path so
it can be checked against the simulated P&L.

With continuous hedging and sigma_r = sigma_i the P&L would be exactly zero. With
N discrete rebalances, the squared return over each step is a random variable
rather than sigma^2 dt, which leaves a hedging error whose standard deviation
falls like 1/sqrt(N) (Derman & Kamal, 1999).

Two results that surprise people, and that the experiments demonstrate:
* The stock's real-world drift mu has no first-order effect: the hedge removes
  it. With discrete rebalancing, mu adds mu^2 dt^2 to each squared return, a
  tiny bias of about -0.5 * mu^2 * dt * vega / sigma that vanishes as dt -> 0.
* Hedging with delta at the realised vol locks in a near-certain P&L equal to the
  price difference; hedging at the implied vol gives a P&L with the same average
  but far more path dependence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import black_scholes as bs
from .paths import gbm_paths


@dataclass
class HedgeResult:
    pnl: np.ndarray            # net P&L per path at expiry, after transaction costs
    gross_pnl: np.ndarray      # P&L before transaction costs
    costs: np.ndarray          # transaction costs paid, carried forward to expiry
    gamma_pnl: np.ndarray      # theta-gamma approximation of gross_pnl (see module docstring)
    premium: float
    n_rebalance: int
    history: dict = field(default_factory=dict)  # per-step arrays, only if keep_history=True

    def summary(self) -> dict:
        return {
            "mean_pnl": float(self.pnl.mean()),
            "std_pnl": float(self.pnl.std(ddof=1)),
            "mean_cost": float(self.costs.mean()),
            "p05": float(np.percentile(self.pnl, 5)),
            "p95": float(np.percentile(self.pnl, 95)),
        }


def simulate_delta_hedge(
    S0=100.0, K=100.0, T=0.25, r=0.0, sigma_implied=0.20, sigma_realised=0.20,
    n_rebalance=63, n_paths=10_000, kind="call", position="short",
    hedge_vol=None, mu=None, q=0.0, cost=0.0, seed=None, paths=None, keep_history=False,
) -> HedgeResult:
    """Simulate selling (or buying) one option and delta-hedging it n_rebalance times.

    sigma_implied  vol the option is traded at (sets the premium)
    sigma_realised vol the stock actually moves with
    hedge_vol      vol used to compute the hedge delta (default: sigma_implied)
    mu             real-world drift of the stock (default: r)
    cost           proportional transaction cost per side, e.g. 0.0005 = 5 bp
    paths          optional pre-simulated stock paths, shape (n_paths, M + 1),
                   where M is a multiple of n_rebalance (lets several hedge
                   frequencies be compared on identical paths)
    """
    if position not in ("short", "long"):
        raise ValueError("position must be 'short' or 'long'")
    hedge_vol = sigma_implied if hedge_vol is None else hedge_vol
    mu = r if mu is None else mu

    if paths is None:
        S = gbm_paths(S0, mu, sigma_realised, T, n_rebalance, n_paths, seed=seed)
    else:
        n_steps = paths.shape[1] - 1
        if n_steps % n_rebalance:
            raise ValueError("number of path steps must be a multiple of n_rebalance")
        S = paths[:, :: n_steps // n_rebalance]
    n_paths = S.shape[0]

    dt = T / n_rebalance
    times = np.arange(n_rebalance + 1) * dt
    growth = np.exp(r * dt)

    premium = bs.price(S0, K, T, r, sigma_implied, q, kind)
    shares = np.full(n_paths, bs.delta(S0, K, T, r, hedge_vol, q, kind))
    cash = premium - shares * S0
    trade_cost = cost * np.abs(shares) * S0
    cash -= trade_cost
    costs = trade_cost * np.exp(r * T)
    gamma_pnl = np.zeros(n_paths)

    if keep_history:
        hist = {"S": S, "times": times, "shares": [shares.copy()], "cash": [cash.copy()]}

    for k in range(1, n_rebalance + 1):
        S_prev, S_now = S[:, k - 1], S[:, k]
        tau_prev = T - times[k - 1]

        # Theta-gamma decomposition for this step (Gamma at the implied vol).
        step_return = S_now / S_prev - 1.0
        g = bs.gamma(S_prev, K, tau_prev, r, sigma_implied, q)
        gamma_pnl += np.exp(r * (T - times[k])) * 0.5 * g * S_prev**2 * (sigma_implied**2 * dt - step_return**2)

        # Interest on cash plus dividends on the shares held over the step.
        cash = cash * growth + shares * S_prev * q * dt

        if k < n_rebalance:
            new_shares = bs.delta(S_now, K, T - times[k], r, hedge_vol, q, kind)
            traded = new_shares - shares
            trade_cost = cost * np.abs(traded) * S_now
            cash -= traded * S_now + trade_cost
            costs += trade_cost * np.exp(r * (T - times[k]))
            shares = new_shares
        else:  # expiry: unwind the hedge
            trade_cost = cost * np.abs(shares) * S_now
            cash += shares * S_now - trade_cost
            costs += trade_cost
            shares = np.zeros(n_paths)

        if keep_history:
            hist["shares"].append(shares.copy())
            hist["cash"].append(cash.copy())

    S_T = S[:, -1]
    payoff = np.maximum(S_T - K, 0.0) if kind == "call" else np.maximum(K - S_T, 0.0)
    net_short = cash - payoff
    gross_short = net_short + costs

    if position == "short":
        net, gross, gpnl = net_short, gross_short, gamma_pnl
    else:  # a long hedged option is the mirror image, but still pays the costs
        net, gross, gpnl = -gross_short - costs, -gross_short, -gamma_pnl

    result = HedgeResult(net, gross, costs, gpnl, float(premium), n_rebalance)
    if keep_history:
        result.history = {"S": hist["S"], "times": hist["times"],
                          "shares": np.array(hist["shares"]).T, "cash": np.array(hist["cash"]).T}
    return result


# ---------------------------------------------------------------- theory to compare against

def derman_kamal_std(S0, K, T, r, sigma, n_rebalance, q=0.0) -> float:
    """Approximate std of the hedging error for an at-the-money option hedged n times.

    Derman & Kamal (1999): std ~ sqrt(pi / 4) * vega * sigma / sqrt(n).
    """
    return float(np.sqrt(np.pi / 4) * bs.vega(S0, K, T, r, sigma, q) * sigma / np.sqrt(n_rebalance))


def leland_vol(sigma, cost, dt, short_gamma=True) -> float:
    """Leland (1985) volatility adjusted for proportional transaction costs.

    sigma_L^2 = sigma^2 * (1 +/- sqrt(8 / pi) * cost / (sigma * sqrt(dt)))

    with cost the one-way proportional cost. A seller of options (short gamma)
    buys high and sells low when rebalancing, so they must price and hedge at a
    *higher* vol to cover expected costs; a buyer uses a lower one.
    """
    adj = np.sqrt(8 / np.pi) * cost / (sigma * np.sqrt(dt))
    factor = 1 + adj if short_gamma else 1 - adj
    if factor <= 0:
        raise ValueError("Leland adjustment makes variance negative; rebalance less often")
    return float(sigma * np.sqrt(factor))
