"""
Experiment 3: build an implied-volatility surface from an option chain.

    python -m experiments.iv_surface --live SPY    # download today's chain (needs internet + yfinance)
    python -m experiments.iv_surface               # use the saved chain in data/

If Black-Scholes were literally true, every option on the same stock would have
the same implied vol and the "smile" would be a flat line. Real equity-index
smiles slope down to the left (downside puts are expensive: crash insurance), and
the skew is steepest for short expiries.

Without --live, the script uses data/sample_chain.csv if present, otherwise it
generates a synthetic chain from a known smile. With synthetic data the true
smile is drawn too, so you can see the solver recover it from noisy quotes.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optlab import market_data as md
from experiments import _style as st


def load_chain(live_ticker=None):
    st.DATA_DIR.mkdir(exist_ok=True)
    if live_ticker:
        raw = md.fetch_option_chain(live_ticker)
        raw.to_csv(st.DATA_DIR / f"{live_ticker.lower()}_chain.csv", index=False)
        return raw, live_ticker, False
    saved = sorted(p for p in st.DATA_DIR.glob("*_chain.csv") if p.name != "sample_chain.csv")
    if saved:
        return pd.read_csv(saved[-1]), saved[-1].name.split("_")[0].upper(), False
    sample = st.DATA_DIR / "sample_chain.csv"
    if not sample.exists():
        md.synthetic_chain().to_csv(sample, index=False)
    return pd.read_csv(sample), "Synthetic", True


def run(quick=False, live_ticker=None):
    st.apply()
    raw, label, synthetic = load_chain(live_ticker)
    surface = md.build_surface(raw)

    expiries = surface.groupby("expiry")["T"].first().sort_values()
    colours = st.ramp(len(expiries))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.7, 1, 1]})
    atm = []
    for (expiry, T), colour in zip(expiries.items(), colours):
        g = surface[surface["expiry"] == expiry].sort_values("log_moneyness")
        days = round(T * 365)
        ax1.plot(g["log_moneyness"], 100 * g["iv"], color=colour, marker="o", ms=3.5, lw=1.6, label=f"{days}d")
        if synthetic:
            k = np.linspace(g["log_moneyness"].min(), g["log_moneyness"].max(), 100)
            ax1.plot(k, 100 * md.synthetic_smile(k, T), color=st.INK_2, lw=0.8, ls=":")
        atm.append((T, np.interp(0.0, g["log_moneyness"], g["iv"]), _skew(g)))
    ax1.axvline(0, color=st.INK_2, lw=0.8)
    ax1.set(title=f"Implied vol smile by expiry ({label})", xlabel="Log-moneyness ln(K / F)", ylabel="Implied vol (%)")
    ax1.legend(title="Expiry" + (" (dotted = true smile)" if synthetic else ""), title_fontsize=9, ncol=2)

    atm = np.array(atm)
    months = atm[:, 0] * 12
    ax2.plot(months, 100 * atm[:, 1], color=st.BLUE, marker="o")
    ax2.set(title="At-the-money vol", xlabel="Months to expiry", ylabel="Implied vol (%)")
    ax3.plot(months, 100 * atm[:, 2], color=st.BLUE, marker="o")
    ax3.set(title="Skew flattens with expiry", xlabel="Months to expiry",
            ylabel="Vol pts gained per 10% lower strike")
    ax3.set_ylim(bottom=0)
    for ax in (ax2, ax3):
        ax.set_xticks([0, 3, 6, 9, 12])
    path = st.save(fig, "03_iv_surface.png")

    out = {"figure": path.name, "source": label, "n_quotes": len(raw), "n_used": len(surface),
           "atm_vol_by_months": {float(round(t * 12, 1)): float(round(100 * v, 2)) for t, v, _ in atm},
           "skew_by_months": {float(round(t * 12, 1)): float(round(100 * s, 2)) for t, _, s in atm}}
    if synthetic:
        err = np.abs(surface["iv"] - md.synthetic_smile(surface["log_moneyness"], surface["T"]))
        out["max_abs_vol_error_pts"] = 100 * err.max()
        out["median_abs_vol_error_pts"] = 100 * err.median()
    return out


def _skew(g, width=0.10):
    """Minus the slope of implied vol against log-moneyness near the money, scaled to a 10% move.

    Positive means downside strikes are priced at higher vol (the usual equity skew).
    """
    near = g[np.abs(g["log_moneyness"]) <= width]
    if len(near) < 3:
        return np.nan
    slope = np.polyfit(near["log_moneyness"], near["iv"], 1)[0]
    return -slope * 0.10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", metavar="TICKER", help="download a live chain, e.g. SPY")
    args = parser.parse_args()
    for k, v in run(live_ticker=args.live).items():
        print(f"{k:26s} {v}")
