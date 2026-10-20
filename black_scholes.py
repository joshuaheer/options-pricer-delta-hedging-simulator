"""
Black-Scholes-Merton pricing and analytic Greeks for European options.

Model
-----
Under the risk-neutral measure the stock follows geometric Brownian motion

    dS = (r - q) S dt + sigma S dW

and a European option is worth the discounted expected payoff. Solving the
expectation gives the closed form

    C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

    d1 = [ln(S/K) + (r - q + sigma^2 / 2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

Conventions used across the whole package
-----------------------------------------
* T is time to expiry in years.
* r and q are continuously compounded rates (risk-free rate, dividend yield).
* sigma is annualised volatility as a decimal (0.20 = 20%).
* vega is per 1.00 change in sigma (divide by 100 for "per vol point").
* theta is per year (divide by 365 for per calendar day).

All functions accept NumPy arrays and broadcast, so a whole option chain or a
whole set of simulated paths can be priced in one call.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_VALID_KINDS = ("call", "put")


def _check_kind(kind: str) -> str:
    kind = kind.lower()
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return kind


def d1_d2(S, K, T, r, sigma, q=0.0):
    """Return (d1, d2). T and sigma must be strictly positive."""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black-Scholes price. At or after expiry (T <= 0) returns intrinsic value."""
    kind = _check_kind(kind)
    S, K, T, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, sigma)))
    live = T > 0
    T_safe = np.where(live, T, 1.0)  # dummy value so no division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = d1_d2(S, K, T_safe, r, sigma, q)
    disc_S = S * np.exp(-q * T_safe)
    disc_K = K * np.exp(-r * T_safe)
    if kind == "call":
        value = disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
        intrinsic = np.maximum(S - K, 0.0)
    else:
        value = disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
        intrinsic = np.maximum(K - S, 0.0)
    out = np.where(live, value, intrinsic)
    return out.item() if out.ndim == 0 else out


def delta(S, K, T, r, sigma, q=0.0, kind="call"):
    """dV/dS: shares of stock needed to hedge one option."""
    kind = _check_kind(kind)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    out = np.exp(-q * np.asarray(T)) * (norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0)
    return out.item() if np.ndim(out) == 0 else out


def gamma(S, K, T, r, sigma, q=0.0, kind="call"):
    """d2V/dS2 (same for calls and puts)."""
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    out = np.exp(-q * np.asarray(T)) * norm.pdf(d1) / (np.asarray(S) * np.asarray(sigma) * np.sqrt(T))
    return out.item() if np.ndim(out) == 0 else out


def vega(S, K, T, r, sigma, q=0.0, kind="call"):
    """dV/dsigma per 1.00 of vol (same for calls and puts)."""
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    out = np.asarray(S) * np.exp(-q * np.asarray(T)) * norm.pdf(d1) * np.sqrt(T)
    return out.item() if np.ndim(out) == 0 else out


def theta(S, K, T, r, sigma, q=0.0, kind="call"):
    """dV/dt per year (calendar time passing, so usually negative)."""
    kind = _check_kind(kind)
    S, T, sigma = np.asarray(S, float), np.asarray(T, float), np.asarray(sigma, float)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    decay = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if kind == "call":
        out = decay + q * S * np.exp(-q * T) * norm.cdf(d1) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        out = decay - q * S * np.exp(-q * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return out.item() if np.ndim(out) == 0 else out


def rho(S, K, T, r, sigma, q=0.0, kind="call"):
    """dV/dr per 1.00 change in the rate."""
    kind = _check_kind(kind)
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    T = np.asarray(T, float)
    if kind == "call":
        out = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        out = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return out.item() if np.ndim(out) == 0 else out


def greeks(S, K, T, r, sigma, q=0.0, kind="call") -> dict:
    """All analytic Greeks in one dictionary."""
    args = (S, K, T, r, sigma, q, kind)
    return {
        "price": price(*args),
        "delta": delta(*args),
        "gamma": gamma(*args),
        "vega": vega(*args),
        "theta": theta(*args),
        "rho": rho(*args),
    }
