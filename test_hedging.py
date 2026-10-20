"""Tests for the delta-hedging simulator.  Run with:  pytest"""

import numpy as np
import pytest

from optlab import black_scholes as bs, hedging, paths

COMMON = dict(S0=100.0, K=100.0, T=0.25, r=0.0, sigma_implied=0.2, sigma_realised=0.2, n_paths=20_000)


def test_fairly_priced_option_has_zero_mean_pnl():
    res = hedging.simulate_delta_hedge(n_rebalance=63, seed=1, **COMMON)
    se = res.pnl.std() / np.sqrt(len(res.pnl))
    assert abs(res.pnl.mean()) < 4 * se


def test_hedging_error_matches_derman_kamal_and_halves_with_4x_rebalancing():
    S = paths.gbm_paths(100, 0.0, 0.2, 0.25, 252, 20_000, seed=2)
    std63 = hedging.simulate_delta_hedge(n_rebalance=63, paths=S, **COMMON).pnl.std()
    std252 = hedging.simulate_delta_hedge(n_rebalance=252, paths=S, **COMMON).pnl.std()
    assert std252 / std63 == pytest.approx(0.5, abs=0.04)
    assert std252 == pytest.approx(hedging.derman_kamal_std(100, 100, 0.25, 0.0, 0.2, 252), rel=0.05)


def test_drift_only_has_a_second_order_effect():
    # A drift mu adds mu^2 dt^2 to each squared return, so the seller loses about
    # 0.5 * mu^2 * dt * sum(Gamma S^2 dt) ~ 0.5 * mu^2 * dt * vega / sigma.
    # It is tiny and vanishes as dt -> 0; there is no first-order (linear in mu) effect.
    n, vega = 63, bs.vega(100, 100, 0.25, 0.0, 0.2)
    dt = 0.25 / n
    for mu in (-0.3, 0.3):
        res = hedging.simulate_delta_hedge(n_rebalance=n, mu=mu, seed=3, **{**COMMON, "n_paths": 50_000})
        predicted = -0.5 * mu**2 * dt * vega / 0.2
        assert res.pnl.mean() == pytest.approx(predicted, abs=0.008)
        assert abs(res.pnl.mean()) < 0.01 * res.premium


def test_seller_earns_the_vol_gap_and_realised_vol_hedge_is_less_risky():
    args = {**COMMON, "sigma_realised": 0.15}
    at_implied = hedging.simulate_delta_hedge(n_rebalance=252, seed=4, **args)
    at_realised = hedging.simulate_delta_hedge(n_rebalance=252, hedge_vol=0.15, seed=4, **args)
    gap = bs.price(100, 100, 0.25, 0, 0.2) - bs.price(100, 100, 0.25, 0, 0.15)
    assert at_implied.pnl.mean() == pytest.approx(gap, abs=0.02)
    assert at_realised.pnl.mean() == pytest.approx(gap, abs=0.02)
    assert at_realised.pnl.std() < 0.6 * at_implied.pnl.std()


def test_theta_gamma_decomposition_explains_pnl():
    res = hedging.simulate_delta_hedge(n_rebalance=63, seed=5, **{**COMMON, "sigma_realised": 0.25})
    assert np.corrcoef(res.gross_pnl, res.gamma_pnl)[0, 1] > 0.97


def test_long_position_mirrors_short_before_costs():
    short = hedging.simulate_delta_hedge(n_rebalance=21, seed=6, cost=0.0005, **COMMON)
    long = hedging.simulate_delta_hedge(n_rebalance=21, seed=6, cost=0.0005, position="long", **COMMON)
    np.testing.assert_allclose(long.gross_pnl, -short.gross_pnl)
    np.testing.assert_allclose(long.costs, short.costs)


def test_put_with_rates_and_dividends_is_hedged_correctly():
    res = hedging.simulate_delta_hedge(n_rebalance=252, seed=7, kind="put",
                                       **{**COMMON, "r": 0.05}, q=0.02)
    se = res.pnl.std() / np.sqrt(len(res.pnl))
    assert abs(res.pnl.mean()) < 4 * se


def test_leland_vol_covers_rebalancing_costs():
    cost, n = 0.0005, 252
    S = paths.gbm_paths(100, 0.0, 0.2, 0.25, n, 20_000, seed=8)
    vol_l = hedging.leland_vol(0.2, cost, 0.25 / n)
    plain = hedging.simulate_delta_hedge(n_rebalance=n, paths=S, cost=cost, **COMMON)
    leland = hedging.simulate_delta_hedge(n_rebalance=n, paths=S, cost=cost, **{**COMMON, "sigma_implied": vol_l})
    # What is left is roughly the cost of opening and closing the initial hedge (~ 1 x cost x S).
    assert abs(leland.pnl.mean()) < 0.3 * abs(plain.pnl.mean())
    assert leland.pnl.mean() == pytest.approx(-cost * 100, abs=0.03)
