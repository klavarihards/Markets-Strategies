"""Phase 4: overlays, regime detection and the pre-registered decision rules."""

import numpy as np
import pandas as pd
import pytest
from conftest import ROOT

from portfolio_engine.allocation import RiskProfile, build_method_allocators
from portfolio_engine.allocation.overlays import MomentumTilt, VolTarget, move_from_equity, rule_regime
from portfolio_engine.backtest import CostModel, run_backtest
from portfolio_engine.backtest.studies import apply_keep_rule, daily_regime, select_methods
from portfolio_engine.config import load_config
from portfolio_engine.data import load_market_data, make_synthetic_frames


@pytest.fixture(scope="module")
def cfg4():
    return load_config(ROOT / "configs" / "phase4_overlays.yaml")


@pytest.fixture(scope="module")
def data4(cfg4):
    return load_market_data(cfg4, frames=make_synthetic_frames(end="2013-12-31", seed=6))


def test_move_from_equity(data4):
    ac = data4.asset_class
    w = pd.Series({"taiwan_equity": 0.2, "us_equity": 0.2, "intl_equity": 0.2, "long_treasury": 0.1,
                   "intermediate_bond": 0.1, "gold": 0.1, "cash": 0.1})
    out = move_from_equity(w, 0.15, ac, to="cash")
    assert out.sum() == pytest.approx(1) and out["cash"] == pytest.approx(0.25)
    assert out[["taiwan_equity", "us_equity", "intl_equity"]].sum() == pytest.approx(0.45)
    out = move_from_equity(w, 0.3, ac)                       # to bonds and cash, in proportion
    assert out["long_treasury"] == pytest.approx(0.2) and out["gold"] == pytest.approx(0.1)


def test_vol_target_only_acts_above_the_band(data4):
    ac = data4.asset_class
    w = pd.Series(1 / 7, index=data4.returns.columns)
    hist = data4.returns.iloc[:1200]
    calm = VolTarget(ac, cap=5.0)(hist, w, hist.index[-1])           # cap far above any forecast
    pd.testing.assert_series_equal(calm, w)
    tight = VolTarget(ac, cap=0.01)
    out = tight(hist, w, hist.index[-1])
    assert out[["taiwan_equity", "us_equity", "intl_equity"]].sum() < w[["taiwan_equity", "us_equity", "intl_equity"]].sum()
    assert out.sum() == pytest.approx(1) and (out >= 0).all() and tight.log[-1]["equity_scale"] < 1


def test_momentum_tilt_stays_near_core_and_inside_profile(cfg4, data4):
    prof = RiskProfile.from_config(cfg4, "balanced")
    core_alloc = build_method_allocators(cfg4)["balanced:risk_parity"]
    d = data4.returns.index[1500]
    core = core_alloc(data4.returns.loc[d - pd.DateOffset(years=3):d])
    tilt = MomentumTilt(prof, data4.asset_class, cfg4["methods"]["black_litterman"]["market_weights"])
    w = tilt(data4.returns.loc[:d], core, d)
    assert (w - core).abs().max() <= 0.05 + 1e-6
    assert prof.violations(w, data4.asset_class) == []


def test_rule_regime_flags_a_crash(data4):
    r = data4.returns.copy()
    crash = r.index[-60:]
    eq = ["taiwan_equity", "us_equity", "intl_equity"]
    rng = np.random.default_rng(0)
    r.loc[crash, eq] = rng.normal(-0.01, 0.04, size=(60, 3))          # falling and volatile
    flags = rule_regime(r, None, data4.asset_class, pd.DatetimeIndex([r.index[-300], r.index[-1]]))
    assert bool(flags.iloc[-1]) is True


def test_overlays_never_see_the_future(cfg4, data4):
    seen = []

    def spy(history, core, d):
        seen.append((history.index[-1], d))
        return core

    costs = CostModel.from_config(cfg4, list(data4.returns.columns))
    alloc = build_method_allocators(cfg4)["balanced:equal_weight"]
    res = run_backtest(data4.returns, alloc, costs, 3, overlay=spy, overlay_frequency="monthly")
    assert all(last == d for last, d in seen)
    # An overlay that changes nothing never trades between rebalances.
    assert set(res.meta["trades"]["kind"]) == {"rebalance"}


def test_regime_flag_applies_from_next_day():
    regime = pd.Series([False, True], index=pd.to_datetime(["2020-01-31", "2020-02-28"]))
    idx = pd.bdate_range("2020-02-25", "2020-03-04")
    flag = daily_regime(regime, idx)
    assert not flag.loc["2020-02-28"] and flag.loc["2020-03-02"]


def test_decision_rules():
    cfg = {"decision_rules": {"keep_overlay": {"min_risk_improvement": 0.10, "max_sharpe_loss": 0.05},
                              "select_method": {"sharpe_tolerance": 0.05}},
           "profiles": {"balanced": {}}}
    idx = pd.MultiIndex.from_product([["balanced"], ["m1", "m2", "m3"], ["vol_target", "tilt"]],
                                     names=["profile", "method", "variant"])
    eff = pd.DataFrame({"d_sharpe": [-0.01, 0.02, -0.02, 0.03, 0.0, -0.08],
                        "maxdd_improvement": [0.2, 0.0, 0.15, 0.01, 0.12, 0.0],
                        "cvar_improvement": [0.05, 0.0, 0.05, 0.0, 0.05, 0.0],
                        "d_ann_return": 0.0, "d_turnover": 0.0}, index=idx)
    kept = apply_keep_rule(eff, cfg)
    assert kept.loc[("balanced", "vol_target"), "keep"] and not kept.loc[("balanced", "tilt"), "keep"]

    rows = [("balanced", "equal_weight|core", 0.50, -0.25, -0.04), ("balanced", "hrp|core", 0.53, -0.18, -0.035),
            ("balanced", "risk_parity|vol_target", 0.52, -0.15, -0.03), ("balanced", "mean_variance|core", 0.40, -0.10, -0.02)]
    summ = pd.DataFrame([{"window_years": 3, "profile": p, "method": m, "sharpe": s, "max_drawdown": d,
                          "cvar95_monthly": c} for p, m, s, d, c in rows]).set_index(["window_years", "profile", "method"])
    comp = summ[[]].assign(max_loss_breached=[True, False, False, False])
    choice = select_methods(summ, comp, kept, cfg)
    # mean_variance is outside the Sharpe tolerance; equal weight broke the limit; lowest CVaR wins.
    assert choice.loc["balanced", "method"] == "risk_parity" and choice.loc["balanced", "variant"] == "vol_target"
