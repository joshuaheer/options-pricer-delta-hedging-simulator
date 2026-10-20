"""
Experiment 1: do the binomial tree and Monte Carlo converge to Black-Scholes, and how fast?

Expected:
* Binomial error ~ 1/N, oscillating between odd and even N; averaging N and N+1
  smooths it out.
* Monte Carlo standard error ~ 1/sqrt(N); antithetic variates and a control
  variate shift the line down without changing its slope.
"""

import numpy as np
import matplotlib.pyplot as plt

from optlab import binomial, black_scholes as bs, monte_carlo as mc
from experiments import _style as st

S, K, T, r, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0


def run(quick=False):
    st.apply()
    exact = bs.price(S, K, T, r, SIGMA, Q, "call")

    steps = np.unique(np.round(np.logspace(1, 3 if quick else 3.3, 40)).astype(int))
    tree = np.array([binomial.crr_price(S, K, T, r, SIGMA, Q, "call", n) for n in steps])
    tree_next = np.array([binomial.crr_price(S, K, T, r, SIGMA, Q, "call", n + 1) for n in steps])
    err_raw = np.abs(tree - exact)
    err_avg = np.abs(0.5 * (tree + tree_next) - exact)

    n_paths = np.array([1_000, 3_000, 10_000, 30_000, 100_000, 300_000] + ([] if quick else [1_000_000]))
    methods = {
        "Plain": dict(antithetic=False, control_variate=False),
        "Antithetic": dict(antithetic=True, control_variate=False),
        "Antithetic + control variate": dict(antithetic=True, control_variate=True),
    }
    se = {name: np.array([mc.mc_price(S, K, T, r, SIGMA, Q, "call", n, seed=7, **kw).std_error for n in n_paths])
          for name, kw in methods.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.loglog(steps, err_raw, color=st.RAMP[0], lw=1.2, label="Raw CRR price")
    ax1.loglog(steps, err_avg, color=st.BLUE, marker="o", ms=4, lw=2, label="Average of N and N+1 steps")
    ax1.loglog(steps, 0.8 * err_avg[0] * steps[0] / steps, color=st.INK_2, ls="--", lw=1, label="Slope -1 reference")
    ax1.set(title="Binomial tree error vs Black-Scholes", xlabel="Tree steps N", ylabel="Absolute pricing error")
    ax1.legend()

    for (name, values), colour, marker in zip(se.items(), (st.BLUE, st.ORANGE, st.AQUA), ("o", "s", "^")):
        ax2.loglog(n_paths, values, color=colour, marker=marker, label=name)
    ax2.loglog(n_paths, se["Plain"][0] * np.sqrt(n_paths[0] / n_paths), color=st.INK_2, ls="--", lw=1,
               label="Slope -1/2 reference")
    ax2.set(title="Monte Carlo standard error", xlabel="Number of paths", ylabel="Standard error of price")
    ax2.legend()
    path = st.save(fig, "01_convergence.png")

    i = list(n_paths).index(100_000)
    vr_anti = (se["Plain"][i] / se["Antithetic"][i]) ** 2
    vr_cv = (se["Plain"][i] / se["Antithetic + control variate"][i]) ** 2
    slope_raw = np.polyfit(np.log(steps), np.log(err_raw), 1)[0]
    slope_mc = np.polyfit(np.log(n_paths), np.log(se["Plain"]), 1)[0]
    return {
        "figure": path.name,
        "bs_price": exact,
        "tree_error_1000": abs(binomial.crr_price(S, K, T, r, SIGMA, Q, "call", 1000) - exact),
        "tree_slope": slope_raw,
        "mc_slope": slope_mc,
        "variance_reduction_antithetic": vr_anti,
        "variance_reduction_antithetic_cv": vr_cv,
    }


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:36s} {v}")
