"""
Option-chain data: download, clean, infer the forward, and build an implied-vol surface.

Live data comes from Yahoo Finance via the `yfinance` package (free, delayed,
no API key). Quotes are noisy, so the pipeline is:

1. Clean: drop options with no bid, crossed quotes, or very wide spreads, and
   use the mid price (bid + ask) / 2.
2. Infer the forward and discount factor from put-call parity instead of
   guessing a rate and dividend yield. For each expiry,

       C(K) - P(K) = DF * (F - K)

   so regressing (C - P) on K gives slope = -DF and intercept = DF * F. This
   pulls the market's own view of rates and dividends out of the prices.
3. Use out-of-the-money options only (puts below the forward, calls above).
   In-the-money quotes are mostly intrinsic value plus noise, so their implied
   vols are unreliable. This is standard practice on trading desks.
4. Solve for implied vol with the Black-76 (forward) form of Black-Scholes:
   pass S = F and q = r, so S e^{-qT} = F * DF and K e^{-rT} = K * DF.

`synthetic_chain` produces quotes in exactly the same format from a known
volatility smile, so the pipeline can run offline and the solver can be tested
on whether it recovers the vols that generated the prices.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import black_scholes as bs
from .implied_vol import implied_vol

RAW_COLUMNS = ["expiry", "T", "strike", "kind", "bid", "ask", "volume", "open_interest", "spot"]


# ---------------------------------------------------------------- live download

def fetch_option_chain(ticker="SPY", n_expiries=8, min_days=7, max_days=400) -> pd.DataFrame:
    """Download calls and puts for the next `n_expiries` expiries between min_days and max_days."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("pip install yfinance to download live option chains") from exc

    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="5d")["Close"].dropna().iloc[-1])
    now = datetime.now(timezone.utc)

    frames = []
    for expiry in tk.options:
        # US equity options stop trading at 16:00 New York time, about 21:00 UTC.
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=21, tzinfo=timezone.utc)
        days = (expiry_dt - now).total_seconds() / 86_400
        if not min_days <= days <= max_days:
            continue
        chain = tk.option_chain(expiry)
        for kind, table in (("call", chain.calls), ("put", chain.puts)):
            frames.append(pd.DataFrame({
                "expiry": expiry,
                "T": days / 365.0,
                "strike": table["strike"].astype(float),
                "kind": kind,
                "bid": table["bid"].astype(float),
                "ask": table["ask"].astype(float),
                "volume": table["volume"].fillna(0).astype(float),
                "open_interest": table["openInterest"].fillna(0).astype(float),
                "spot": spot,
            }))
        if len({f["expiry"].iloc[0] for f in frames}) >= n_expiries:
            break
    if not frames:
        raise RuntimeError(f"no option expiries found for {ticker} between {min_days} and {max_days} days")
    return pd.concat(frames, ignore_index=True)[RAW_COLUMNS]


# ---------------------------------------------------------------- cleaning and the forward

def clean_chain(raw: pd.DataFrame, max_rel_spread=0.25, min_price=0.05) -> pd.DataFrame:
    df = raw.copy()
    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"])]
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df = df[(df["mid"] >= min_price) & ((df["ask"] - df["bid"]) / df["mid"] <= max_rel_spread)]
    return df.reset_index(drop=True)


def implied_forward(chain_one_expiry: pd.DataFrame, band=0.10) -> tuple[float, float]:
    """Return (forward, discount_factor) from put-call parity on strikes near spot."""
    spot = chain_one_expiry["spot"].iloc[0]
    calls = chain_one_expiry[chain_one_expiry["kind"] == "call"].set_index("strike")["mid"]
    puts = chain_one_expiry[chain_one_expiry["kind"] == "put"].set_index("strike")["mid"]
    both = pd.concat([calls.rename("C"), puts.rename("P")], axis=1, join="inner")
    both = both[np.abs(both.index / spot - 1) <= band]
    if len(both) < 3:
        raise ValueError("need at least 3 strikes with both a call and a put near spot")
    slope, intercept = np.polyfit(both.index.values, (both["C"] - both["P"]).values, 1)
    discount = -slope
    if not 0.5 < discount <= 1.05:
        raise ValueError(f"implausible discount factor {discount:.4f}; quotes too noisy")
    return intercept / discount, discount


def build_surface(raw: pd.DataFrame, max_rel_spread=0.25, max_abs_log_moneyness=0.4) -> pd.DataFrame:
    """Clean a raw chain and add forward, implied rate, log-moneyness and implied vol (OTM only)."""
    df = clean_chain(raw, max_rel_spread=max_rel_spread)
    pieces = []
    for expiry, group in df.groupby("expiry"):
        try:
            fwd, disc = implied_forward(group)
        except ValueError:
            continue  # skip an expiry whose quotes cannot pin down the forward
        T = group["T"].iloc[0]
        g = group.copy()
        g["forward"] = fwd
        g["discount"] = disc
        g["r_implied"] = -np.log(disc) / T
        g["log_moneyness"] = np.log(g["strike"] / fwd)
        otm = ((g["kind"] == "put") & (g["strike"] < fwd)) | ((g["kind"] == "call") & (g["strike"] >= fwd))
        g = g[otm & (np.abs(g["log_moneyness"]) <= max_abs_log_moneyness)]
        r_imp = g["r_implied"].iloc[0] if len(g) else 0.0
        g["iv"] = [implied_vol(m, fwd, k, T, r_imp, r_imp, kd) for m, k, kd in zip(g["mid"], g["strike"], g["kind"])]
        pieces.append(g)
    if not pieces:
        raise RuntimeError(
            "no expiry had enough clean quotes to build a surface. Outside US market hours "
            "(14:30-21:00 UK time) Yahoo often reports zero bids; try again while the market is open.")
    return pd.concat(pieces, ignore_index=True).dropna(subset=["iv"]).sort_values(["T", "strike"]).reset_index(drop=True)


# ---------------------------------------------------------------- offline synthetic chain

def synthetic_smile(log_moneyness, T):
    """A stylised equity-index smile: downside skew that is steeper for short expiries."""
    k = np.asarray(log_moneyness, float)
    atm = 0.17 - 0.03 * np.exp(-T / 0.25)          # upward-sloping ATM term structure
    skew = -0.10 / np.sqrt(np.maximum(T, 0.04))     # skew flattens like 1/sqrt(T)
    curvature = 0.35
    return np.maximum(atm + skew * k + curvature * k**2, 0.05)


def synthetic_chain(spot=100.0, r=0.04, q=0.015, expiry_days=(14, 30, 60, 91, 182, 365),
                    strike_range=(0.70, 1.30), strike_step=0.01, half_spread=0.01, seed=0) -> pd.DataFrame:
    """Quotes generated from `synthetic_smile`, with a bid-ask spread and a little noise in the mid."""
    rng = np.random.default_rng(seed)
    rows = []
    strikes = spot * np.arange(strike_range[0], strike_range[1] + 1e-9, strike_step)
    for days in expiry_days:
        T = days / 365.0
        fwd = spot * np.exp((r - q) * T)
        vols = synthetic_smile(np.log(strikes / fwd), T)
        for kind in ("call", "put"):
            mids = bs.price(spot, strikes, T, r, vols, q, kind)
            mids = mids * (1 + rng.normal(0, 0.002, len(mids)))  # 0.2% quote noise
            spread = np.maximum(half_spread, 0.01 * mids)
            for K, m, s in zip(strikes, mids, spread):
                rows.append({"expiry": f"T+{days}d", "T": T, "strike": round(K, 2), "kind": kind,
                             "bid": max(m - s, 0.0), "ask": m + s, "volume": 100, "open_interest": 1000, "spot": spot})
    return pd.DataFrame(rows)[RAW_COLUMNS]
