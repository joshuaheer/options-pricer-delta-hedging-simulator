"""
Finite-difference Greeks for any pricer.

Bump one input, reprice, and take central differences:

    delta ~ [V(S + h) - V(S - h)] / 2h
    gamma ~ [V(S + h) - 2 V(S) + V(S - h)] / h^2

Central differences have O(h^2) truncation error, but making h too small lets
floating-point rounding dominate (gamma divides by h^2). The bump sizes below
balance the two for prices of order 1-100.

Comparing these against the analytic Black-Scholes Greeks is a check on both:
if the closed forms have a sign or factor error, the numbers will disagree.
The same function works for the binomial pricer, with one caveat: bumping S
moves every tree node relative to the strike, so finite-difference gamma on a
tree is noisy. For trees, use binomial.crr_greeks, which reads delta and gamma
off nodes the tree already has. For Monte Carlo, reuse the same random seed for
the bumped and unbumped prices (common random numbers) or the noise swamps the
difference.
"""

from __future__ import annotations

from typing import Callable


def fd_greeks(pricer: Callable, S, K, T, r, sigma, q=0.0, kind="call", **pricer_kwargs) -> dict:
    """pricer must have the signature pricer(S, K, T, r, sigma, q, kind, **kwargs)."""

    def V(S_=S, T_=T, r_=r, sigma_=sigma):
        return pricer(S_, K, T_, r_, sigma_, q, kind, **pricer_kwargs)

    h_s = 1e-3 * S
    h_vol = 1e-4
    h_t = min(1e-4, 0.5 * T)
    h_r = 1e-4

    base = V()
    up, down = V(S_=S + h_s), V(S_=S - h_s)
    return {
        "price": base,
        "delta": (up - down) / (2 * h_s),
        "gamma": (up - 2 * base + down) / h_s**2,
        "vega": (V(sigma_=sigma + h_vol) - V(sigma_=sigma - h_vol)) / (2 * h_vol),
        # theta is dV/dt = -dV/dT (time passing shortens expiry)
        "theta": -(V(T_=T + h_t) - V(T_=T - h_t)) / (2 * h_t),
        "rho": (V(r_=r + h_r) - V(r_=r - h_r)) / (2 * h_r),
    }
