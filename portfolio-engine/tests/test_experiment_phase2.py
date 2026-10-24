"""Phase 2 gate on synthetic data: all five methods run walk-forward and the comparison tables exist."""

import pandas as pd
import pytest
from conftest import ROOT

from portfolio_engine.backtest import (
    compliance_table,
    exposure_table,
    run_experiment,
    sharpe_comparison_table,
    summary_table,
)
from portfolio_engine.config import load_config, override
from portfolio_engine.data import load_market_data, make_synthetic_frames


@pytest.fixture(scope="module")
def phase2():
    cfg = override(load_config(ROOT / "configs" / "phase2_methods.yaml"),
                   **{"backtest.window_years": [3], "bootstrap.n": 200})
    data = load_market_data(cfg, frames=make_synthetic_frames(end="2012-12-31", seed=4))
    return cfg, data, run_experiment(cfg, data)


def test_all_methods_and_profiles_run(phase2):
    cfg, data, res = phase2
    summ = summary_table(res, data, cfg)
    assert len(summ) == 3 + 3 * 5
    assert set(summ.index.get_level_values("profile")) == {"benchmark", "conservative", "balanced", "growth"}
    assert set(summ.xs("balanced", level="profile").index.get_level_values("method")) == {
        "equal_weight", "mean_variance", "black_litterman", "risk_parity", "hrp"}
    starts = summ["start"].unique()
    assert len(starts) == 1                      # everyone is compared over the same days


def test_comparison_tables(phase2):
    cfg, data, res = phase2
    comp = compliance_table(res, data, cfg)
    assert len(comp) == 15 and comp["share_vol_cap_relaxed"].between(0, 1).all()
    exp = exposure_table(res, data)
    assert exp[["equity", "bond", "cash", "gold"]].sum(axis=1).round(10).eq(1).all()
    # Every profile portfolio stays inside its equity range on average.
    for prof, (lo, hi) in [("conservative", (0.15, 0.35)), ("balanced", (0.40, 0.60)), ("growth", (0.65, 0.85))]:
        eq = exp.xs(prof, level="profile")["equity"]
        assert ((eq >= lo - 1e-6) & (eq <= hi + 1e-6)).all()
    sc = sharpe_comparison_table(res, data, cfg)
    assert len(sc) == 30 and (sc["ci_low"] <= sc["ci_high"]).all()


def test_diagnostics_are_per_run(phase2):
    _, _, res = phase2
    log = res[(3, "growth:hrp")].meta["log"]
    assert list(log.index) == list(res[(3, "growth:hrp")].weights.index)


def test_phase2_reproduces(phase2):
    cfg, data, res = phase2
    again = run_experiment(cfg, data)
    for k in res:
        pd.testing.assert_series_equal(res[k].returns, again[k].returns, check_exact=True)
