"""
Experiment 2: check the analytic Greeks against finite differences and the binomial tree,
and show how delta and gamma change as expiry approaches.

Expected: analytic and finite-difference Greeks agree to ~6 significant figures.
Near expiry, delta becomes a step function and gamma spikes at the strike --
which is exactly why short-dated at-the-money options are the hardest to hedge.
"""

import numpy as np
import matplotlib.pyplot as plt

from optlab import binomial, black_scholes as bs, greeks
from experiments import _style as st

S, K, T, r, SIGMA, Q = 100.0, 100.0, 0.5, 0.04, 0.25, 0.01
NAMES = ["price", "delta", "gamma", "vega", "theta", "rho"]


def run(quick=False):
    st.apply()
    rows = []
    worst_rel = 0.0
    for kind in ("call", "put"):
        analytic = bs.greeks(S, K, T, r, SIGMA, Q, kind)
        fd = greeks.fd_greeks(bs.price, S, K, T, r, SIGMA, Q, kind)
        tree = binomial.crr_greeks(S, K, T, r, SIGMA, Q, kind, steps=1000 if quick else 2000)
        for name in NAMES:
            rel = abs(fd[name] - analytic[name]) / max(abs(analytic[name]), 1e-12)
            worst_rel = max(worst_rel, rel)
            rows.append((kind, name, analytic[name], fd[name], tree.get(name, np.nan)))

    spots = np.linspace(70, 130, 400)
    expiries = [(1.0, "1 year"), (0.25, "3 months"), (0.02, "1 week")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for (tau, label), colour in zip(expiries, (st.BLUE, st.ORANGE, st.AQUA)):
        ax1.plot(spots, bs.delta(spots, K, tau, r, SIGMA, Q, "call"), color=colour, label=label)
        ax2.plot(spots, bs.gamma(spots, K, tau, r, SIGMA, Q), color=colour, label=label)
    ax1.set(title="Call delta as expiry approaches", xlabel="Stock price (strike = 100)", ylabel="Delta")
    ax2.set(title="Gamma concentrates at the strike", xlabel="Stock price (strike = 100)", ylabel="Gamma")
    ax1.legend(title="Time to expiry", title_fontsize=9)
    ax2.legend(title="Time to expiry", title_fontsize=9)
    path = st.save(fig, "02_greeks.png")
    return {"figure": path.name, "table": rows, "worst_relative_fd_error": worst_rel}


def table_markdown(rows):
    lines = ["| Option | Greek | Analytic | Finite difference | Binomial tree |", "|---|---|---:|---:|---:|"]
    for kind, name, a, f, t in rows:
        tree = "" if np.isnan(t) else f"{t:.6f}"
        lines.append(f"| {kind} | {name} | {a:.6f} | {f:.6f} | {tree} |")
    return "\n".join(lines)


if __name__ == "__main__":
    out = run()
    print(table_markdown(out["table"]))
    print("worst relative FD error:", out["worst_relative_fd_error"])
