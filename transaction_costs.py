"""
Experiment 6: with transaction costs, how often should you rebalance?

Set-up: sell a 3-month ATM call at 20% vol (fairly priced), hedge N times, and pay
a proportional cost per side on every share traded: 0, 2, 5 or 10 basis points.

The trade-off:
* hedging error falls like 1/sqrt(N) (experiment 4), but
* expected cost rises like sqrt(N): each rebalance trades ~ Gamma * S * dS, and
  |dS| ~ sigma S sqrt(dt), so N rebalances cost ~ N * sqrt(T / N) = sqrt(N T).

Total risk, measured as root-mean-square P&L (bias and noise together), is
U-shaped in N, and the best frequency drops as costs rise.

Leland (1985) fix: charge and hedge at an adjusted vol
    sigma_L^2 = sigma^2 * (1 + sqrt(8 / pi) * cost / (sigma * sqrt(dt)))
which should cover the expected *rebalancing* cost. It does not cover the cost of
setting up and unwinding the initial hedge, so a small loss remains.
"""

import numpy as np
import matplotlib.pyplot as plt

from optlab import hedging, paths
from experiments import _style as st

S0, K, T, R, SIGMA = 100.0, 100.0, 0.25, 0.0, 0.20
FREQUENCIES = [6, 12, 21, 42, 63, 126, 252, 504, 1008]
COSTS_BP = [0, 2, 5, 10]


def run(quick=False):
    st.apply()
    n_paths = 5_000 if quick else 20_000
    S = paths.gbm_paths(S0, R, SIGMA, T, 1008, n_paths, seed=7)

    grid = {}
    for bp in COSTS_BP:
        for n in FREQUENCIES:
            res = hedging.simulate_delta_hedge(S0, K, T, R, SIGMA, SIGMA, n, paths=S, cost=bp / 1e4)
            grid[bp, n] = {"rmse": float(np.sqrt(np.mean(res.pnl**2))), "mean": float(res.pnl.mean()),
                           "std": float(res.pnl.std(ddof=1)), "cost": float(res.costs.mean()),
                           "hedge_err": float(res.gross_pnl.std(ddof=1))}
    best = {bp: min(FREQUENCIES, key=lambda n: grid[bp, n]["rmse"]) for bp in COSTS_BP}

    leland = []
    for bp in COSTS_BP[1:]:
        for n in (63, 252):
            vol_l = hedging.leland_vol(SIGMA, bp / 1e4, T / n)
            plain = grid[bp, n]["mean"]
            adj = hedging.simulate_delta_hedge(S0, K, T, R, vol_l, SIGMA, n, paths=S, cost=bp / 1e4)
            leland.append({"cost_bp": bp, "n": n, "leland_vol": vol_l, "mean_plain": plain,
                           "mean_leland": float(adj.pnl.mean()), "mean_cost": grid[bp, n]["cost"]})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for bp, colour in zip(COSTS_BP, st.ramp(len(COSTS_BP))):
        y = [grid[bp, n]["rmse"] for n in FREQUENCIES]
        ax1.semilogx(FREQUENCIES, y, color=colour, marker="o", ms=5, label=f"{bp} bp")
        ax1.scatter([best[bp]], [grid[bp, best[bp]]["rmse"]], s=90, facecolor="none", edgecolor=st.INK, lw=1.4, zorder=5)
    ax1.set(title="Total risk is U-shaped once trading costs money", xlabel="Rebalances over 3 months",
            ylabel="Root-mean-square P&L ($)")
    ax1.set_ylim(bottom=0)
    ax1.legend(title="Cost per side (circle = best N)", title_fontsize=9)

    bp = 5
    ax2.loglog(FREQUENCIES, [grid[bp, n]["hedge_err"] for n in FREQUENCIES], color=st.BLUE, marker="o",
               label="Hedging error (std, before costs)")
    ax2.loglog(FREQUENCIES, [grid[bp, n]["cost"] for n in FREQUENCIES], color=st.ORANGE, marker="s",
               label="Expected transaction cost")
    ax2.set(title=f"The trade-off at {bp} bp per side", xlabel="Rebalances over 3 months", ylabel="$ per option")
    ax2.legend()
    path = st.save(fig, "06_transaction_costs.png")
    return {"figure": path.name, "best_n": best, "grid": grid, "leland": leland}


if __name__ == "__main__":
    out = run()
    print("best N by cost:", out["best_n"])
    for row in out["leland"]:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
