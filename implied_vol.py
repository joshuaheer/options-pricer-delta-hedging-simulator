"""
Implied volatility: the sigma that makes the Black-Scholes price equal a market price.

There is no closed form, so we solve f(sigma) = BS(sigma) - market_price = 0.

Newton-Raphson uses f'(sigma) = vega:

    sigma_{n+1} = sigma_n - (BS(sigma_n) - price) / vega(sigma_n)

It converges quadratically near the root but can fail for deep in- or
out-of-the-money options, where vega is tiny and a Newton step overshoots. Because
the BS price is strictly increasing in sigma, the root is always bracketed, so we
keep a [low, high] bracket and fall back to bisection whenever Newton would step
outside it. This "safeguarded Newton" is as fast as Newton when Newton works and
never fails when it doesn't.

A price outside the no-arbitrage bounds has no implied vol; we return NaN rather
than a misleading number. For a call:  max(S e^{-qT} - K e^{-rT}, 0) < C < S e^{-qT}.
"""

from __future__ import annotations

import numpy as np

from . import black_scholes as bs

SIGMA_LOW, SIGMA_HIGH = 1e-4, 5.0


def no_arbitrage_bounds(S, K, T, r, q=0.0, kind="call") -> tuple[float, float]:
    fwd_s, fwd_k = S * np.exp(-q * T), K * np.exp(-r * T)
    if kind == "call":
        return max(fwd_s - fwd_k, 0.0), fwd_s
    return max(fwd_k - fwd_s, 0.0), fwd_k


def implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-10, max_iter=100) -> float:
    """Implied volatility for one option, or NaN if the price is outside arbitrage bounds."""
    lower, upper = no_arbitrage_bounds(S, K, T, r, q, kind)
    if not (lower < price < upper) or T <= 0:
        return float("nan")
    if price - lower < 1e-8 * S:
        # Deep in the money with essentially zero time value: every vol gives
        # the same price to machine precision, so implied vol is not defined.
        return float("nan")

    low, high = SIGMA_LOW, SIGMA_HIGH
    if bs.price(S, K, T, r, high, q, kind) < price:
        return float("nan")  # would need vol above 500%

    # Starting guess: Brenner-Subrahmanyam ATM approximation, clipped to the bracket.
    sigma = np.clip(np.sqrt(2 * np.pi / T) * price / S, 0.05, 1.0)

    for _ in range(max_iter):
        diff = bs.price(S, K, T, r, sigma, q, kind) - price
        if abs(diff) < tol:
            return float(sigma)
        # Keep the bracket: price increases with sigma.
        if diff > 0:
            high = sigma
        else:
            low = sigma
        v = bs.vega(S, K, T, r, sigma, q)
        step_ok = v > 1e-12
        if step_ok:
            candidate = sigma - diff / v
            step_ok = low < candidate < high
        sigma = candidate if step_ok else 0.5 * (low + high)
        if high - low < tol:
            return float(sigma)
    return float(sigma)


def implied_vol_array(prices, S, K, T, r, q=0.0, kind="call", **kwargs) -> np.ndarray:
    """Vectorised wrapper: every argument may be a scalar or an array of the same length."""
    prices, S, K, T, r, q = np.broadcast_arrays(*(np.asarray(x, float) for x in (prices, S, K, T, r, q)))
    kinds = np.broadcast_to(np.asarray(kind), prices.shape)
    return np.array([
        implied_vol(p, s, k, t, rr, qq, str(kd), **kwargs)
        for p, s, k, t, rr, qq, kd in zip(prices.ravel(), S.ravel(), K.ravel(), T.ravel(), r.ravel(), q.ravel(), kinds.ravel())
    ]).reshape(prices.shape)
