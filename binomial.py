"""
Cox-Ross-Rubinstein (CRR) binomial tree.

Time to expiry is split into N steps of length dt = T / N. Each step the stock
moves up by u = e^{sigma sqrt(dt)} or down by d = 1 / u. The risk-neutral
probability of an up-move is

    p = (e^{(r - q) dt} - d) / (u - d)

chosen so the tree reproduces the risk-free forward. The option is valued by
backward induction from the payoff at expiry:

    V_i = e^{-r dt} [p V_up + (1 - p) V_down]

For American options each node also takes max(continuation, exercise value).

As N grows the tree converges to Black-Scholes at rate ~1/N, but the error
oscillates between odd and even N because the strike sits at a different place
relative to the terminal nodes. Averaging the N and N+1 prices removes most of
that oscillation (see experiments/convergence.py).
"""

from __future__ import annotations

import numpy as np


def crr_price(S, K, T, r, sigma, q=0.0, kind="call", steps=500, american=False) -> float:
    return _crr(S, K, T, r, sigma, q, kind, steps, american)["price"]


def crr_greeks(S, K, T, r, sigma, q=0.0, kind="call", steps=500, american=False) -> dict:
    """Price, delta and gamma read straight off the first two levels of the tree.

    Bumping S and re-building the tree (finite differences) gives noisy gamma,
    because the bump shifts every node relative to the strike. Reading delta and
    gamma from nodes that already exist avoids that and costs nothing extra.
    """
    return _crr(S, K, T, r, sigma, q, kind, steps, american)


def _crr(S, K, T, r, sigma, q, kind, steps, american) -> dict:
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk-neutral probability {p:.4f} outside (0, 1); use more steps")

    sign = 1.0 if kind == "call" else -1.0

    # Terminal stock prices, index j = number of up-moves (0..N).
    j = np.arange(steps + 1)
    stock = S * u**j * d ** (steps - j)
    values = np.maximum(sign * (stock - K), 0.0)

    level = {}  # option values at steps 1 and 2, kept for delta and gamma
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            stock = S * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))
            values = np.maximum(values, sign * (stock - K))
        if i in (1, 2):
            level[i] = values.copy()

    out = {"price": float(values[0])}
    if steps >= 2:
        v_d, v_u = level[1]
        out["delta"] = float((v_u - v_d) / (S * u - S * d))
    if steps >= 3:
        v_dd, v_ud, v_uu = level[2]
        s_uu, s_ud, s_dd = S * u * u, S, S * d * d
        delta_up = (v_uu - v_ud) / (s_uu - s_ud)
        delta_down = (v_ud - v_dd) / (s_ud - s_dd)
        out["gamma"] = float((delta_up - delta_down) / (0.5 * (s_uu - s_dd)))
    return out
