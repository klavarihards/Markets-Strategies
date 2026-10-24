"""Dated Black-Litterman views: confidence -> uncertainty, walk-forward use, scoring."""

import numpy as np
import pandas as pd
import pytest
from conftest import ROOT

from portfolio_engine.allocation import RiskProfile
from portfolio_engine.allocation.methods import BlackLitterman
from portfolio_engine.config import load_config
from portfolio_engine.data import load_market_data, make_synthetic_frames
from portfolio_engine.estimators import implied_returns, ledoit_wolf
from portfolio_engine.estimators.views import active_views, idzorek_omega, load_views, score_views

VIEWS = """
views:
  - id: tw_vs_us
    entered: "2012-06-29"
    source: "test"
    statement: "Taiwan equity beats US equity by 5% a year over 12 months"
    weights: {taiwan_equity: 1, us_equity: -1}
    q: 0.05
    confidence: 0.9
    horizon_months: 12
"""


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    cfg = load_config(ROOT / "configs" / "phase2_methods.yaml")
    data = load_market_data(cfg, frames=make_synthetic_frames(end="2014-12-31", seed=5))
    path = tmp_path_factory.mktemp("v") / "views.yaml"
    path.write_text(VIEWS)
    return cfg, data, path


def test_idzorek_omega_matches_pypfopt(setup):
    bl_mod = pytest.importorskip("pypfopt.black_litterman")
    cfg, data, _ = setup
    cov = ledoit_wolf(data.returns.iloc[-756:])[0]
    pi = implied_returns(cov, pd.Series(cfg["methods"]["black_litterman"]["market_weights"]), 2.5)
    P = np.zeros((2, 7))
    P[0, 0], P[0, 1], P[1, 5] = 1, -1, 1
    Q, conf = np.array([0.03, 0.02]), np.array([0.6, 0.25])
    ref = bl_mod.BlackLittermanModel.idzorek_method(conf, cov.to_numpy(), pi.to_numpy().reshape(-1, 1),
                                                      Q.reshape(-1, 1), P, 0.05)
    np.testing.assert_allclose(idzorek_omega(P, cov.to_numpy(), conf, 0.05), ref, rtol=1e-9, atol=1e-11)


def test_views_are_used_only_while_active(setup):
    cfg, data, path = setup
    views = load_views(path)
    assert len(active_views(views, pd.Timestamp("2012-06-28"), data.returns.columns)[3]) == 0   # before
    assert active_views(views, pd.Timestamp("2012-12-31"), data.returns.columns)[3] == ["tw_vs_us"]
    assert len(active_views(views, pd.Timestamp("2013-07-01"), data.returns.columns)[3]) == 0   # expired
    bl = cfg["methods"]["black_litterman"]
    prof = RiskProfile.from_config(cfg, "balanced")
    with_v = BlackLitterman(prof, data.asset_class, market_weights=bl["market_weights"], views_file=str(path))
    without = BlackLitterman(prof, data.asset_class, market_weights=bl["market_weights"])
    for d, should_differ in ((pd.Timestamp("2012-03-30"), False), (pd.Timestamp("2012-09-28"), True)):
        hist = data.returns.loc[d - pd.DateOffset(years=3):d]
        a, b = with_v(hist), without(hist)
        assert (not np.allclose(a, b)) == should_differ
        if should_differ:                      # a confident "Taiwan beats US" view moves money that way
            assert a["taiwan_equity"] - a["us_equity"] > b["taiwan_equity"] - b["us_equity"]


def test_scoring(setup):
    _, data, path = setup
    idx = pd.bdate_range("2012-06-01", "2013-12-31")
    r = pd.DataFrame(0.0, index=idx, columns=data.returns.columns)
    r["taiwan_equity"] = 0.0004                 # Taiwan beats US by about 10% a year
    s = score_views(load_views(path), r, pd.Series(0.0, idx))
    row = s.loc["tw_vs_us"]
    assert row["status"] == "closed" and row["direction_right"] and row["realised"] == pytest.approx(0.105, abs=0.01)
    assert row["error"] == pytest.approx(row["realised"] - 0.05)


def test_repository_views_file_loads():
    assert isinstance(load_views(ROOT / "configs" / "views.yaml"), list)
