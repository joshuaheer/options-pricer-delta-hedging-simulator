"""
Experiment 4: how much hedging error is left when you can only rebalance N times?

Set-up: sell a 3-month at-the-money call at 20% implied vol; the stock really does
move with 20% vol, so the option was fairly priced and any P&L is pure hedging
error. All frequencies are run on the same 20,000 stock paths.

Expected:
* Mean P&L ~ 0 at every frequency (discrete hedging adds risk, not bias).
* Std of P&L ~ 1/sqrt(N): four times as many rebalances halves the error, and it
  should match Derman & Kamal's formula  sqrt(pi/4) * vega * sigma / sqrt(N).
* Path by path, P&L ~ sum of 0.5 * Gamma * S^2 * (sigma^2 dt - realised return^2):
  the theta-gamma decomposition should explain almost all of the variation.
"""

import numpy as np
import matplotlib.pyplot as plt

from optlab import hedging, paths
from experiments import _style as st

S0, K, T, R, SIGMA = 100.0, 100.0, 0.25, 0.0, 0.20
FREQUENCIES = [6, 12, 21, 42, 63, 126, 252, 504, 1008]   # all divide 1008
LABELS = {12: "weekly", 63: "daily", 252: "4x daily"}


def run(quick=False):
    st.apply()
    n_paths = 5_000 if quick else 20_000
    S = paths.gbm_paths(S0, R, SIGMA, T, 1008, n_paths, seed=42)

    results = {n: hedging.simulate_delta_hedge(S0, K, T, R, SIGMA, SIGMA, n, paths=S) for n in FREQUENCIES}
    std = np.array([results[n].pnl.std(ddof=1) for n in FREQUENCIES])
    mean = np.array([results[n].pnl.mean() for n in FREQUENCIES])
    theory = np.array([hedging.derman_kamal_std(S0, K, T, R, SIGMA, n) for n in FREQUENCIES])
    slope = np.polyfit(np.log(FREQUENCIES), np.log(std), 1)[0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.4))

    ax1.loglog(FREQUENCIES, theory, color=st.INK_2, ls="--", lw=1.2, label="Derman-Kamal formula")
    ax1.loglog(FREQUENCIES, std, color=st.BLUE, marker="o", ls="none", ms=7, label="Simulated")
    for n, name in LABELS.items():
        i = FREQUENCIES.index(n)
        ax1.annotate(name, (n, std[i]), textcoords="offset points", xytext=(6, 6), fontsize=8, color=st.INK_2)
    ax1.set(title="Error falls like 1/sqrt(N)", xlabel="Rebalances over 3 months",
            ylabel="Std of P&L per option ($)")
    ax1.legend()

    bins = np.linspace(-3, 3, 81)
    ax2.hist(results[12].pnl, bins=bins, color=st.ORANGE, alpha=0.75, label="Weekly (N = 12)")
    ax2.hist(results[63].pnl, bins=bins, color=st.BLUE, alpha=0.75, label="Daily (N = 63)")
    ax2.set(title="P&L of the hedged short call",
            xlabel=f"P&L per option (\\$), premium \\${results[63].premium:.2f}", ylabel="Paths")
    ax2.legend()

    r63 = results[63]
    corr = np.corrcoef(r63.gross_pnl, r63.gamma_pnl)[0, 1]
    lim = np.percentile(np.abs(r63.gross_pnl), 99.5)
    ax3.plot([-lim, lim], [-lim, lim], color=st.INK_2, lw=1, ls="--")
    ax3.scatter(r63.gamma_pnl[:3000], r63.gross_pnl[:3000], s=6, color=st.BLUE, alpha=0.35, linewidths=0)
    ax3.set(title="Theta-gamma explains the P&L", xlabel="Theta-gamma prediction ($)",
            ylabel="Simulated P&L ($)", xlim=(-lim, lim), ylim=(-lim, lim))
    ax3.text(0.04, 0.94, f"correlation = {corr:.3f}", transform=ax3.transAxes, fontsize=9, color=st.INK)
    path = st.save(fig, "04_hedge_frequency.png")

    table = [(n, float(m), float(s), float(t)) for n, m, s, t in zip(FREQUENCIES, mean, std, theory)]
    return {"figure": path.name, "premium": r63.premium, "slope": slope, "gamma_corr_daily": corr,
            "table": table, "max_ratio_error": float(np.max(np.abs(std / theory - 1)))}


if __name__ == "__main__":
    out = run()
    print("N, mean, std, Derman-Kamal")
    for row in out["table"]:
        print(row)
    print({k: v for k, v in out.items() if k != "table"})
