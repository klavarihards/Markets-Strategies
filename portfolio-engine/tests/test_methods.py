"""The five methods: cross-checks against PyPortfolioOpt / Riskfolio-Lib, and profile compliance."""

import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import ROOT

from portfolio_engine.allocation import (
    RiskProfile,
    build_method_allocators,
    hrp_weights,
    risk_contributions,
    risk_parity_weights,
)
from portfolio_engine.allocation.optimizer import solve
from portfolio_engine.config import load_config
from portfolio_engine.data import load_market_data, make_synthetic_frames
from portfolio_engine.estimators import black_litterman, historical_mean, implied_returns, ledoit_wolf

pypfopt = pytest.importorskip("pypfopt")


@pytest.fixture(scope="module")
def cfg2():
    return load_config(ROOT / "configs" / "phase2_methods.yaml")


@pytest.fixture(scope="module")
def data2(cfg2):
    return load_market_data(cfg2, frames=make_synthetic_frames(end="2014-12-31", seed=9))


@pytest.fixture(scope="module")
def window(data2):
    return data2.returns.iloc[-756:]


@pytest.fixture(scope="module")
def cov(window):
    return ledoit_wolf(window)[0]


def _pypfopt_frontier(mu, cov, profile, asset_class):
    """PyPortfolioOpt set up with the same hard limits as the profile (no volatility cap)."""
    from pypfopt import EfficientFrontier
    ef = EfficientFrontier(mu, cov, weight_bounds=(0, profile.max_weight))
    group_of = {s: g for g, classes in profile.groups.items() for s, c in asset_class.items() if c in classes}
    ef.add_sector_constraints(group_of, {g: b[0] for g, b in profile.bounds.items()},
                              {g: b[1] for g, b in profile.bounds.items()})
    return ef


@pytest.mark.parametrize("pname", ["conservative", "balanced", "growth"])
def test_mean_variance_matches_pypfopt(cfg2, data2, window, cov, pname):
    profile = RiskProfile.from_config(cfg2, pname)
    mu = historical_mean(window)
    w, _ = solve(cov.to_numpy(), list(cov.index), data2.asset_class, profile, "utility",
                 mu=mu.to_numpy(), risk_aversion=2.5, vol_cap=False)
    ef = _pypfopt_frontier(mu, cov, profile, data2.asset_class)
    ref = pd.Series(ef.max_quadratic_utility(risk_aversion=2.5)).reindex(cov.index)
    # Agree to ~1e-6. Cash (a TWD deposit) has almost no variance, which leaves both
    # solvers a slightly ill-conditioned problem, so allow solver-tolerance differences.
    np.testing.assert_allclose(w, ref.to_numpy(), atol=1e-5)


def test_black_litterman_weights_match_pypfopt(cfg2, data2, cov):
    bl = cfg2["methods"]["black_litterman"]
    profile = RiskProfile.from_config(cfg2, "balanced")
    pi = implied_returns(cov, pd.Series(bl["market_weights"]), 2.5)
    P = np.array([[1, -1, 0, 0, 0, 0, 0.0]])
    Q = np.array([0.02])
    mu, post_cov = black_litterman(cov, pi, P, Q, tau=bl["tau"])
    w, _ = solve(post_cov.to_numpy(), list(cov.index), data2.asset_class, profile, "utility",
                 mu=mu.to_numpy(), risk_aversion=2.5, vol_cap=False)

    from pypfopt.black_litterman import BlackLittermanModel
    blm = BlackLittermanModel(cov, pi=pi, P=P, Q=Q, tau=bl["tau"])
    ef = _pypfopt_frontier(blm.bl_returns(), blm.bl_cov(), profile, data2.asset_class)
    ref = pd.Series(ef.max_quadratic_utility(risk_aversion=2.5)).reindex(cov.index)
    np.testing.assert_allclose(w, ref.to_numpy(), atol=1e-5)   # agree to ~1e-6: solver tolerance


def test_risk_parity_matches_riskfolio(window, cov):
    rp = pytest.importorskip("riskfolio")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        port = rp.Portfolio(returns=window)
        port.assets_stats(method_mu="hist", method_cov="hist")
        port.cov = cov
        ref = port.rp_optimization(model="Classic", rm="MV", rf=0, b=None, hist=True)["weights"]
    mine = risk_parity_weights(cov.to_numpy())
    np.testing.assert_allclose(mine, ref.reindex(cov.index).to_numpy(), atol=1e-4)
    # Our solver hits the defining property essentially exactly.
    np.testing.assert_allclose(risk_contributions(mine, cov.to_numpy()), 1 / len(cov), atol=1e-12)


def test_risk_budgets_are_respected(cov):
    b = np.array([4, 1, 1, 1, 1, 1, 1.0])
    w = risk_parity_weights(cov.to_numpy(), b)
    np.testing.assert_allclose(risk_contributions(w, cov.to_numpy()), b / b.sum(), atol=1e-12)


def test_hrp_matches_pypfopt(cov, monkeypatch):
    import scipy.cluster.hierarchy as sch
    from pypfopt import HRPOpt
    # PyPortfolioOpt 1.6 validates the linkage name against a private scipy list
    # that newer scipy versions removed; put it back so the reference code can run.
    if not hasattr(sch, "_LINKAGE_METHODS"):
        monkeypatch.setattr(sch, "_LINKAGE_METHODS", {"single": 0, "complete": 1, "average": 2,
                                                      "weighted": 6}, raising=False)
    ref = pd.Series(HRPOpt(cov_matrix=cov).optimize()).reindex(cov.index)
    np.testing.assert_allclose(hrp_weights(cov).to_numpy(), ref.to_numpy(), atol=1e-10)


def test_every_method_respects_every_profile(cfg2, data2):
    allocs = build_method_allocators(cfg2)
    assert len(allocs) == 15
    dates = data2.returns.index[[800, 1500, -1]]
    for d in dates:
        hist = data2.returns.loc[d - pd.DateOffset(years=3):d]
        for name, alloc in allocs.items():
            w = alloc(hist)
            assert alloc.profile.violations(w, data2.asset_class) == [], (name, d)
            info = alloc.log[-1]
            if not info["vol_cap_relaxed"]:
                assert info["ex_ante_vol"] <= alloc.profile.vol_band[1] + 1e-6, name


def test_vol_cap_is_relaxed_rather_than_breaking_hard_limits(cfg2, data2, cov):
    profile = RiskProfile.from_config(cfg2, "growth")
    tight = RiskProfile(**{**profile.__dict__, "vol_band": (0.0, 0.01)})   # impossible with 65% equity
    w, info = solve(cov.to_numpy(), list(cov.index), data2.asset_class, tight, "nearest",
                    target=np.full(7, 1 / 7))
    assert info["vol_cap_relaxed"]
    assert tight.violations(pd.Series(w, cov.index), data2.asset_class) == []


def test_nearest_leaves_compliant_weights_alone(cfg2, data2, cov):
    profile = RiskProfile.from_config(cfg2, "balanced")
    target = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.15])   # already inside 'balanced'
    w, info = solve(cov.to_numpy(), list(cov.index), data2.asset_class, profile, "nearest",
                    target=target, vol_cap=False)
    np.testing.assert_allclose(w, target, atol=1e-6)
    assert info["moved_by_profile"] < 1e-6


def test_methods_are_deterministic(cfg2, window):
    a = build_method_allocators(cfg2)
    b = build_method_allocators(cfg2)
    for k in a:
        pd.testing.assert_series_equal(a[k](window), b[k](window))
