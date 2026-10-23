"""Estimators, cross-checked against PyPortfolioOpt and scikit-learn."""

import numpy as np
import pandas as pd
import pytest

from portfolio_engine.data import make_synthetic_frames
from portfolio_engine.estimators import (black_litterman, correlation_from_cov, historical_mean,
                                         implied_returns, ledoit_wolf, sample_cov, views_matrices)

pypfopt = pytest.importorskip("pypfopt")


@pytest.fixture(scope="module")
def rets():
    px = make_synthetic_frames(end="2010-12-31", seed=2)["prices"]
    return px.pct_change().dropna().iloc[-756:]


def test_ledoit_wolf_constant_variance_matches_sklearn(rets):
    from sklearn.covariance import ledoit_wolf as sk_lw
    mine, delta = ledoit_wolf(rets, "constant_variance")
    ref, ref_delta = sk_lw(rets.to_numpy())
    assert delta == pytest.approx(ref_delta, rel=1e-10)
    np.testing.assert_allclose(mine.to_numpy(), ref * 252, rtol=1e-10)


def test_ledoit_wolf_constant_correlation_matches_pypfopt(rets):
    from pypfopt.risk_models import CovarianceShrinkage
    mine, delta = ledoit_wolf(rets, "constant_correlation")
    cs = CovarianceShrinkage(rets, returns_data=True, frequency=252)
    ref = cs.ledoit_wolf("constant_correlation")
    # PyPortfolioOpt builds its sample matrix with 1/(T-1); the paper (and this
    # code) uses 1/T. Re-running the formula with 1/(T-1) reproduces PyPortfolioOpt
    # to machine precision, so the only gap is that convention: about 1/T = 0.13%.
    np.testing.assert_allclose(mine.to_numpy(), ref.to_numpy(), rtol=3e-3)
    assert delta == pytest.approx(cs.delta, rel=1e-2)


def test_shrinkage_moves_towards_target(rets):
    S = sample_cov(rets)
    shrunk, delta = ledoit_wolf(rets.iloc[:60])           # few observations: more shrinkage
    _, delta_long = ledoit_wolf(rets)
    assert 0 < delta_long < delta <= 1
    np.testing.assert_allclose(np.diag(ledoit_wolf(rets)[0]), np.diag(S) * (len(rets) - 1) / len(rets))
    corr = correlation_from_cov(shrunk).to_numpy()
    off = corr[~np.eye(len(corr), dtype=bool)]
    assert off.std() < correlation_from_cov(sample_cov(rets.iloc[:60])).to_numpy()[~np.eye(len(corr), dtype=bool)].std()


def test_historical_mean_is_annualised(rets):
    assert historical_mean(rets).equals(rets.mean() * 252)


def test_black_litterman_matches_pypfopt(rets):
    from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns
    cov = ledoit_wolf(rets)[0]
    mkt = pd.Series(np.arange(1, len(cov) + 1, dtype=float), cov.index)
    pi = implied_returns(cov, mkt, 2.5)
    ref_pi = market_implied_prior_returns(mkt, 2.5, cov)
    np.testing.assert_allclose(pi.to_numpy(), ref_pi.to_numpy(), rtol=1e-12)

    views = [{"weights": {"EWT": 1, "SPY": -1}, "q": 0.02}, {"weights": {"GLD": 1}, "q": 0.06}]
    P, Q = views_matrices(views, cov.index)
    post, post_cov = black_litterman(cov, pi, P, Q, tau=0.05)
    bl = BlackLittermanModel(cov, pi=ref_pi, P=P, Q=Q, tau=0.05)
    np.testing.assert_allclose(post.to_numpy(), bl.bl_returns().to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(post_cov.to_numpy(), bl.bl_cov().to_numpy(), rtol=1e-10)


def test_black_litterman_without_views_returns_equilibrium(rets):
    cov = ledoit_wolf(rets)[0]
    pi = implied_returns(cov, pd.Series(1.0, cov.index), 2.5)
    post, post_cov = black_litterman(cov, pi, tau=0.05)
    pd.testing.assert_series_equal(post, pi, check_names=False)
    np.testing.assert_allclose(post_cov.to_numpy(), 1.05 * cov.to_numpy())


def test_confident_view_moves_returns_further(rets):
    cov = ledoit_wolf(rets)[0]
    pi = implied_returns(cov, pd.Series(1.0, cov.index), 2.5)
    P, Q = views_matrices([{"weights": {"GLD": 1}, "q": pi["GLD"] + 0.05}], cov.index)
    vague = black_litterman(cov, pi, P, Q, omega=np.array([[1.0]]))[0]
    sure = black_litterman(cov, pi, P, Q, omega=np.array([[1e-6]]))[0]
    assert pi["GLD"] < vague["GLD"] < sure["GLD"] == pytest.approx(pi["GLD"] + 0.05, abs=1e-4)
