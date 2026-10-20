"""
Monte Carlo pricing of European options with two variance-reduction methods.

A European option only depends on S_T, so we sample the terminal price directly:

    S_T = S exp[(r - q - sigma^2 / 2) T + sigma sqrt(T) Z]

price = e^{-rT} * mean(payoff), and the standard error of that mean shrinks like
1 / sqrt(n_paths). Halving the error needs four times as many paths, which is
why variance reduction matters.

* Antithetic variates: use Z and -Z in pairs. The pair average is the sample.
* Control variate: the discounted stock price X = e^{-rT} S_T has a known mean
  (S e^{-qT}). Replace Y with Y - b (X - E[X]) using the optimal
  b = Cov(Y, X) / Var(X). Because a call payoff is highly correlated with S_T,
  this removes a large share of the variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        return self.price - z * self.std_error, self.price + z * self.std_error


def mc_price(
    S, K, T, r, sigma, q=0.0, kind="call", n_paths=100_000,
    antithetic=False, control_variate=False, seed=None,
) -> MCResult:
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    rng = np.random.default_rng(seed)

    n_draws = (n_paths + 1) // 2 if antithetic else n_paths
    z = rng.standard_normal(n_draws)
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    sign = 1.0 if kind == "call" else -1.0

    def discounted(z_):
        s_t = S * np.exp(drift + vol * z_)
        return disc * np.maximum(sign * (s_t - K), 0.0), disc * s_t

    y, x = discounted(z)
    if antithetic:
        y_anti, x_anti = discounted(-z)
        y, x = 0.5 * (y + y_anti), 0.5 * (x + x_anti)  # one sample per pair

    if control_variate:
        x_mean = S * np.exp(-q * T)  # known exactly under the risk-neutral measure
        cov = np.cov(y, x, ddof=1)
        b = cov[0, 1] / cov[1, 1]
        y = y - b * (x - x_mean)

    return MCResult(price=float(y.mean()), std_error=float(y.std(ddof=1) / np.sqrt(len(y))), n_paths=n_paths)
