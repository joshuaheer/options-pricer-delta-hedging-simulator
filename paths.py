"""
Simulation of geometric Brownian motion paths.

The exact solution of dS = mu S dt + sigma S dW over a step of length dt is

    S_{t+dt} = S_t exp[(mu - sigma^2 / 2) dt + sigma sqrt(dt) Z],   Z ~ N(0, 1)

so there is no discretisation error in the stock price itself; the only error in
the hedging experiments comes from rebalancing at discrete times, which is the
effect we want to isolate.
"""

from __future__ import annotations

import numpy as np


def gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=None, antithetic=False, rng=None):
    """Return an array of shape (n_paths, n_steps + 1); column 0 is S0.

    antithetic=True pairs every draw Z with -Z, which cancels odd moments of the
    noise and usually cuts Monte Carlo variance for smooth payoffs.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, n_steps))
        z = np.concatenate([z, -z])[:n_paths]
    else:
        z = rng.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    return S0 * np.exp(log_paths)
