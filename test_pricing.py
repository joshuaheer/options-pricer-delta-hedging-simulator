"""Tests for the pricers, Greeks and implied-vol solver.  Run with:  pytest"""

import itertools

import numpy as np
import pytest

from optlab import binomial, black_scholes as bs, greeks, implied_vol as iv, market_data as md, monte_carlo as mc

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0)


def test_black_scholes_textbook_values():
    # Hull, Options, Futures and Other Derivatives: S=K=100, T=1, r=5%, vol=20%.
    assert bs.price(**BASE, kind="call") == pytest.approx(10.450584, abs=1e-6)
    assert bs.price(**BASE, kind="put") == pytest.approx(5.573526, abs=1e-6)


@pytest.mark.parametrize("K,T,q", [(80, 0.5, 0.0), (100, 1.0, 0.02), (130, 2.0, 0.03)])
def test_put_call_parity(K, T, q):
    S, r, sigma = 100.0, 0.04, 0.3
    lhs = bs.price(S, K, T, r, sigma, q, "call") - bs.price(S, K, T, r, sigma, q, "put")
    assert lhs == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_price_at_expiry_is_intrinsic():
    assert bs.price(110, 100, 0.0, 0.05, 0.2, kind="call") == pytest.approx(10.0)
    assert bs.price(90, 100, 0.0, 0.05, 0.2, kind="put") == pytest.approx(10.0)


def test_binomial_converges_to_black_scholes():
    exact = bs.price(**BASE, kind="call")
    assert binomial.crr_price(**BASE, kind="call", steps=2000) == pytest.approx(exact, abs=2e-3)


def test_american_put_worth_at_least_european_and_american_call_equals_european_without_dividends():
    euro_put = binomial.crr_price(**BASE, kind="put", steps=800)
    amer_put = binomial.crr_price(**BASE, kind="put", steps=800, american=True)
    assert amer_put > euro_put + 0.1  # early exercise premium is material here
    euro_call = binomial.crr_price(**BASE, kind="call", steps=800)
    amer_call = binomial.crr_price(**BASE, kind="call", steps=800, american=True)
    assert amer_call == pytest.approx(euro_call, abs=1e-10)


def test_tree_delta_and_gamma_match_black_scholes():
    tree = binomial.crr_greeks(**BASE, kind="put", steps=2000)
    assert tree["delta"] == pytest.approx(bs.delta(**BASE, kind="put"), abs=1e-3)
    assert tree["gamma"] == pytest.approx(bs.gamma(**BASE), abs=1e-4)


@pytest.mark.parametrize("kw", [{}, {"antithetic": True}, {"antithetic": True, "control_variate": True}])
def test_monte_carlo_within_four_standard_errors(kw):
    exact = bs.price(**BASE, kind="call")
    res = mc.mc_price(**BASE, kind="call", n_paths=200_000, seed=11, **kw)
    assert abs(res.price - exact) < 4 * res.std_error


def test_control_variate_reduces_standard_error():
    plain = mc.mc_price(**BASE, kind="call", n_paths=50_000, seed=3)
    cv = mc.mc_price(**BASE, kind="call", n_paths=50_000, seed=3, control_variate=True)
    assert cv.std_error < 0.5 * plain.std_error


@pytest.mark.parametrize("kind", ["call", "put"])
def test_analytic_greeks_match_finite_differences(kind):
    args = (100.0, 95.0, 0.75, 0.03, 0.25, 0.01, kind)
    analytic = bs.greeks(*args)
    fd = greeks.fd_greeks(bs.price, *args)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert fd[name] == pytest.approx(analytic[name], rel=1e-4), name


def test_implied_vol_round_trip():
    for K, T, sigma, kind in itertools.product([50, 80, 100, 120, 200], [0.02, 0.25, 2.0],
                                               [0.05, 0.2, 0.6, 1.5], ["call", "put"]):
        price = bs.price(100, K, T, 0.03, sigma, 0.01, kind)
        lower, _ = iv.no_arbitrage_bounds(100, K, T, 0.03, 0.01, kind)
        if price - lower < 1e-6:
            continue  # no time value: implied vol is not identifiable
        assert iv.implied_vol(price, 100, K, T, 0.03, 0.01, kind) == pytest.approx(sigma, abs=1e-5)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(iv.implied_vol(0.0, 100, 100, 1, 0.0))        # below intrinsic/zero
    assert np.isnan(iv.implied_vol(101.0, 100, 100, 1, 0.0))      # call worth more than the stock


def test_put_call_parity_regression_recovers_forward_and_discount():
    spot, r, q, days = 100.0, 0.04, 0.015, 182
    raw = md.synthetic_chain(spot=spot, r=r, q=q, expiry_days=(days,))
    fwd, disc = md.implied_forward(md.clean_chain(raw))
    T = days / 365
    assert fwd == pytest.approx(spot * np.exp((r - q) * T), rel=2e-3)
    assert disc == pytest.approx(np.exp(-r * T), abs=2e-3)


def test_surface_recovers_known_smile():
    surface = md.build_surface(md.synthetic_chain())
    true = md.synthetic_smile(surface["log_moneyness"], surface["T"])
    assert np.max(np.abs(surface["iv"] - true)) < 0.002  # within 0.2 vol points despite quote noise
