"""Phase 3 risk tools: CVaR optimiser (vs Riskfolio-Lib), decomposition, stress tests, goal simulation."""

import warnings

import numpy as np
import pandas as pd
import pytest

from portfolio_engine.allocation import RiskProfile, build_method_allocators
from portfolio_engine.allocation.optimizer import solve
from portfolio_engine.config import load_config
from portfolio_engine.data import load_market_data, make_synthetic_frames
from portfolio_engine.risk import (Client, bootstrap_paths, correlation_stress_cov, cvar_contributions, decompose,
                                   goal_report, replay, shock_returns, simulate_goal, student_t_paths,
                                   volatility_contributions, wealth_paths)

from conftest import ROOT


@pytest.fixture(scope="module")
def cfg3():
    return load_config(ROOT / "configs" / "phase3_risk.yaml")


@pytest.fixture(scope="module")
def rets():
    return make_synthetic_frames(end="2010-12-31", seed=2)["prices"].pct_change().dropna().iloc[-756:]


def _cvar(w, R, alpha=0.95):
    """Rockafellar-Uryasev CVaR of losses, computed exactly by sorting."""
    loss = np.sort(-(R @ w))[::-1]
    k = (1 - alpha) * len(loss)
    whole = int(np.floor(k))
    return (loss[:whole].sum() + (k - whole) * (loss[whole] if whole < len(loss) else 0)) / k


# --- CVaR optimiser -------------------------------------------------------------------------

def test_min_cvar_matches_riskfolio(rets):
    rf = pytest.importorskip("riskfolio")
    cov = rets.cov().to_numpy() * 252
    w, _ = solve(cov, list(rets.columns), {c: "equity" for c in rets.columns}, None, "min_cvar",
                 scenarios=rets.to_numpy(), alpha=0.95)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        port = rf.Portfolio(returns=rets)
        port.assets_stats(method_mu="hist", method_cov="hist")
        ref = port.optimization(model="Classic", rm="CVaR", obj="MinRisk", rf=0, l=0, hist=True)["weights"]
    np.testing.assert_allclose(w, ref.reindex(rets.columns).to_numpy(), atol=1e-6)


def test_min_cvar_beats_random_portfolios(rets):
    R = rets.to_numpy()
    w, _ = solve(rets.cov().to_numpy() * 252, list(rets.columns), {c: "equity" for c in rets.columns}, None,
                 "min_cvar", scenarios=R, alpha=0.95)
    rng = np.random.default_rng(0)
    others = rng.dirichlet(np.ones(R.shape[1]), size=500)
    assert _cvar(w, R) <= min(_cvar(x, R) for x in others) + 1e-12


def test_min_cvar_respects_profiles(cfg3):
    data = load_market_data(cfg3, frames=make_synthetic_frames(end="2012-12-31", seed=3))
    hist = data.returns.iloc[-756:]
    allocs = {k: a for k, a in build_method_allocators(cfg3).items() if k.endswith("min_cvar")}
    assert len(allocs) == 3
    for name, a in allocs.items():
        w = a(hist)
        assert a.profile.violations(w, data.asset_class) == [], name


# --- decomposition --------------------------------------------------------------------------

def test_contributions_add_up(rets):
    w = pd.Series(np.full(7, 1 / 7), rets.columns)
    cov = rets.cov() * 252
    assert volatility_contributions(w, cov).sum() == pytest.approx(1.0)
    contrib, total = cvar_contributions(w, rets)
    assert contrib.sum() == pytest.approx(total)
    port = rets @ w
    assert total == pytest.approx(port[port <= port.quantile(0.05)].mean())
    by_class = decompose(w, cov, rets, asset_class={c: ("bond" if c in ("TLT", "IEF") else "other") for c in rets})
    assert by_class[["weight", "share_of_volatility", "share_of_cvar"]].sum().round(10).eq(1).all()


def test_equity_dominates_risk_of_a_balanced_mix():
    rng = np.random.default_rng(1)
    r = pd.DataFrame({"eq": rng.normal(0, 0.012, 2000), "bond": rng.normal(0, 0.003, 2000)})
    w = pd.Series({"eq": 0.4, "bond": 0.6})
    share = volatility_contributions(w, r.cov())
    assert share["eq"] > 0.85                      # 40% of the money, almost all of the risk


# --- stress tests ---------------------------------------------------------------------------

def test_replay_hand_calculation():
    idx = pd.bdate_range("2020-01-01", periods=6)
    r = pd.DataFrame({"a": [0.0, -0.5, 0.0, 1.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    out = replay(pd.Series({"a": 0.5, "b": 0.5}), r, "2020-01-02", "2020-01-03")
    # a halves: portfolio 0.5*0.5 + 0.5 = 0.75
    assert out["total_return"] == pytest.approx(-0.25)
    assert out["max_drawdown"] == pytest.approx(-0.25)
    # recovered when 'a' doubles back on 2020-01-06 (0.5*1.0 + 0.5 = 1.0)
    assert out["months_to_recover"] == pytest.approx(round(3 / 30.44, 1))


def test_shock_returns():
    sleeves = ["taiwan_equity", "us_equity", "long_treasury", "cash"]
    fx_exposed = ["us_equity", "long_treasury"]
    r = shock_returns({"shocks": {"taiwan_equity": -0.30}, "fx": 0.10}, sleeves, fx_exposed, {"long_treasury": 17})
    assert r.to_dict() == pytest.approx({"taiwan_equity": -0.30, "us_equity": 0.10, "long_treasury": 0.10, "cash": 0.0})
    r = shock_returns({"rates": 0.02}, sleeves, fx_exposed, {"long_treasury": 17})
    assert r["long_treasury"] == pytest.approx(-0.34) and r["us_equity"] == 0


def test_correlation_stress_keeps_volatilities(rets):
    cov = rets.cov() * 252
    risky = list(rets.columns[:4])
    st = correlation_stress_cov(cov, 0.8, risky)
    np.testing.assert_allclose(np.diag(st), np.diag(cov))
    sd = np.sqrt(np.diag(st))
    corr = st.to_numpy() / np.outer(sd, sd)
    assert corr[0, 1] == pytest.approx(0.8) and corr[0, 5] == pytest.approx(cov.iloc[0, 5] / np.sqrt(cov.iloc[0, 0] * cov.iloc[5, 5]))


# --- goals ----------------------------------------------------------------------------------

def _client(**kw):
    base = dict(name="t", description="", horizon_years=10, initial=100000, monthly=1000, goal=300000, max_loss=-0.2)
    return Client(**{**base, **kw})


def test_wealth_matches_annuity_formula():
    r = 0.004
    paths = np.full((3, 120), r)
    w = wealth_paths(paths, 100000, 1000)
    # Contributions at the start of each month: future value of an annuity due.
    expected = 100000 * (1 + r) ** 120 + 1000 * (1 + r) * ((1 + r) ** 120 - 1) / r
    assert w[0, -1] == pytest.approx(expected)


def test_goal_report_on_certain_paths():
    good = np.full((10, 240), 0.01)
    bad = np.full((10, 240), -0.01)
    assert goal_report(good, _client())["p_reach_goal"] == 1.0
    rep = goal_report(bad, _client())
    assert rep["p_reach_goal"] == 0.0 and rep["bad_case_years_to_goal"] == np.inf
    # Losses are measured on returns, not wealth: contributions do not hide them.
    assert rep["p_loss_beyond_limit"] == 1.0


def test_simulation_is_seeded_and_sensible():
    rng = np.random.default_rng(0)
    hist = pd.Series(rng.normal(0.005, 0.03, 200))
    mc = {"n_paths": 2000, "block_months": 12, "student_t_dof": 5, "extra_years": 5,
          "sensitivity": {"extra_monthly": 2000, "extra_years": 2}}
    a = simulate_goal(hist, _client(), mc, seed=1)
    b = simulate_goal(hist, _client(), mc, seed=1)
    assert a["bootstrap"] == b["bootstrap"] and a["student_t"] == b["student_t"]
    for kind in ("bootstrap", "student_t"):
        rep = a[kind]
        assert rep["p_reach_goal_more_monthly"] >= rep["p_reach_goal"]     # same draws, more money
        assert rep["p_reach_goal_later"] >= rep["p_reach_goal"]            # same draws, more time
        assert rep["p5_wealth"] <= rep["median_wealth"] <= rep["p95_wealth"]


def test_student_t_paths_match_target_moments():
    x = student_t_paths(0.006, 0.04, 20000, 60, dof=5, seed=3)
    assert x.mean() == pytest.approx(0.006, abs=2e-4)                      # antithetic: mean nearly exact
    assert x.std() == pytest.approx(0.04, rel=0.03)


def test_bootstrap_uses_only_history():
    hist = np.arange(24, dtype=float) / 1000
    p = bootstrap_paths(hist, 50, 36, block=12, seed=0)
    assert set(np.unique(p)) <= set(hist)
    # each 12-month block is a run of consecutive months (circular)
    first_block = p[0, :12]
    assert np.all(np.diff((first_block * 1000).astype(int)) % 24 == 1)


# --- the full Phase 3 analysis on synthetic data ----------------------------------------------

def test_risk_analysis_end_to_end(cfg3):
    from portfolio_engine.backtest import run_experiment
    from portfolio_engine.config import override
    from portfolio_engine.risk import run_risk_analysis
    cfg = override(cfg3, **{"backtest.window_years": [3], "monte_carlo.n_paths": 400,
                            "monte_carlo.extra_years": 2})
    data = load_market_data(cfg, frames=make_synthetic_frames(end="2015-12-31", seed=4))
    res = run_experiment(cfg, data)
    a = run_risk_analysis(cfg, data, res)
    n_ports = 3 + 3 * 6
    assert len(a["weights"]) == n_ports
    assert len(a["historical_stress"]) == n_ports * 5 and len(a["hypothetical_stress"]) == n_ports * 4
    assert a["risk_summary"]["cvar95_daily"].lt(0).all()
    # Each client is simulated with their own profile's six methods plus the three benchmarks, two ways.
    assert len(a["goals"]) == 3 * (6 + 3) * 2
    assert a["goals"]["p_reach_goal"].between(0, 1).all()
    for d in a["decomposition"].values():
        assert d["share_of_volatility"].sum() == pytest.approx(1) and d["share_of_cvar"].sum() == pytest.approx(1)
