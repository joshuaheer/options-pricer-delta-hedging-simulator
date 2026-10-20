"""
Experiment 5: what happens when realised vol differs from the implied vol you sold at?

Set-up: sell a 3-month ATM call at 20% implied vol and hedge it 4 times a day.
The stock's true vol runs from 10% to 30%. Two hedging choices:

* hedge with delta at the IMPLIED vol (what you can observe), or
* hedge with delta at the REALISED vol (if you somehow knew it).

Expected (Ahmad & Wilmott, 2005):
* Both give the same average P&L, close to BS(20%) - BS(sigma_realised):
  you earn the vol you sold minus the vol that turned up.
* Hedging at realised vol makes that P&L almost certain.
* Hedging at implied vol makes it path-dependent: the gain is gamma-weighted,
  so it depends on whether the stock spent its time near the strike.
"""

import numpy as np
import matplotlib.pyplot as plt

from optlab import black_scholes as bs, hedging
from experiments import _style as st

S0, K, T, R, SIGMA_I, N = 100.0, 100.0, 0.25, 0.0, 0.20, 252
REALISED = np.round(np.arange(0.10, 0.301, 0.02), 2)


def run(quick=False):
    st.apply()
    n_paths = 4_000 if quick else 20_000
    rows = []
    for i, sig_r in enumerate(REALISED):
        common = dict(S0=S0, K=K, T=T, r=R, sigma_implied=SIGMA_I, sigma_realised=sig_r,
                      n_rebalance=N, n_paths=n_paths, seed=100 + i)
        at_implied = hedging.simulate_delta_hedge(**common)
        at_realised = hedging.simulate_delta_hedge(hedge_vol=sig_r, **common)
        theory = bs.price(S0, K, T, R, SIGMA_I) - bs.price(S0, K, T, R, sig_r)
        rows.append({
            "sigma_r": float(sig_r), "theory": float(theory),
            **{f"{tag}_{stat}": float(fn(res.pnl)) for tag, res in (("imp", at_implied), ("real", at_realised))
               for stat, fn in (("mean", np.mean), ("std", np.std),
                                ("p05", lambda x: np.percentile(x, 5)), ("p95", lambda x: np.percentile(x, 95)))},
        })

    col = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    x = 100 * col["sigma_r"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.fill_between(x, col["imp_p05"], col["imp_p95"], color=st.ORANGE, alpha=0.18, lw=0)
    ax1.fill_between(x, col["real_p05"], col["real_p95"], color=st.BLUE, alpha=0.25, lw=0)
    ax1.plot(x, col["imp_mean"], color=st.ORANGE, marker="s", label="Hedge at implied vol (mean, 5-95% band)")
    ax1.plot(x, col["real_mean"], color=st.BLUE, marker="o", label="Hedge at realised vol (mean, 5-95% band)")
    ax1.plot(x, col["theory"], color=st.INK_2, ls="--", lw=1.2, label="BS(20%) - BS(realised)")
    ax1.axhline(0, color=st.INK_2, lw=0.8)
    ax1.axvline(100 * SIGMA_I, color=st.INK_2, lw=0.8, ls=":")
    ax1.set(title="Short call P&L vs the vol that actually happened", xlabel="Realised vol (%), option sold at 20%",
            ylabel="P&L per option ($)")
    ax1.legend(loc="upper right")

    ax2.plot(x, col["imp_std"], color=st.ORANGE, marker="s", label="Hedge at implied vol")
    ax2.plot(x, col["real_std"], color=st.BLUE, marker="o", label="Hedge at realised vol")
    ax2.set(title="Same average, very different risk", xlabel="Realised vol (%)", ylabel="Std of P&L ($)")
    ax2.set_ylim(bottom=0)
    ax2.legend()
    path = st.save(fig, "05_vol_mismatch.png")
    return {"figure": path.name, "rows": rows}


if __name__ == "__main__":
    for r in run()["rows"]:
        print({k: round(v, 3) for k, v in r.items()})
