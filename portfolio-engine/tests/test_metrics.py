import numpy as np
import pandas as pd
import pytest

from portfolio_engine.backtest import metrics as m


def _s(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_cagr_doubling_in_one_year():
    r = _s([2 ** (1 / 252) - 1] * 252)
    assert m.cagr(r) == pytest.approx(1.0)


def test_cagr_is_geometric():
    r = _s([0.5, -0.5] * 126)        # +50% then -50% loses money every two days
    assert m.cagr(r) < 0


def test_max_drawdown_and_recovery():
    r = _s([0.10, -0.50, 0.20, 0.50, 0.40])
    # wealth 1.10, 0.55, 0.66, 0.99, 1.386: worst fall is 1.10 -> 0.55 = -50%
    assert m.max_drawdown(r) == pytest.approx(-0.5)
    st = m.drawdown_stats(r)
    assert st["recovery_days"] == 3
    assert st["longest_underwater_days"] == 3


def test_drawdown_from_a_losing_start():
    r = _s([-0.10, 0.05])
    assert m.max_drawdown(r) == pytest.approx(-0.10)


def test_sharpe_uses_excess_returns():
    rng = np.random.default_rng(0)
    r = _s(rng.normal(0.0004, 0.01, 1000))
    rf = _s([0.0001] * 1000)
    ex = r - rf
    assert m.sharpe(r, rf) == pytest.approx(ex.mean() / ex.std() * np.sqrt(252))
    assert m.sharpe(r, rf) < m.sharpe(r, 0.0)


def test_cvar_is_the_mean_of_the_tail():
    r = _s(np.linspace(-0.10, 0.09, 20))   # 20 evenly spaced returns; worst 5% = the single worst
    assert m.cvar(r, 0.95) == pytest.approx(-0.10)
    assert m.cvar(r, 0.90) == pytest.approx(np.mean([-0.10, -0.09]))


def test_tracking_error_zero_against_itself():
    r = _s(np.random.default_rng(1).normal(0, 0.01, 300))
    assert m.tracking_error(r, r) == 0


def test_sharpe_difference_ci():
    rng = np.random.default_rng(0)
    base = rng.normal(0.0003, 0.01, 2500)
    a = _s(base + rng.normal(0, 0.002, 2500))
    b = _s(base + rng.normal(0, 0.002, 2500))
    same = m.sharpe_difference_ci(a, b, n_boot=500, seed=1)
    assert same["ci_low"] < 0 < same["ci_high"]              # two near-identical strategies: no winner
    better = m.sharpe_difference_ci(a + 0.0005, b, n_boot=500, seed=1)
    assert better["ci_low"] > 0                              # a clear edge is detected
    assert m.sharpe_difference_ci(a, b, n_boot=500, seed=1) == same   # seeded: reproducible


def test_fast_bootstrap_equals_direct_resampling():
    rng = np.random.default_rng(3)
    a, b = _s(rng.normal(3e-4, 0.01, 1000)), _s(rng.normal(1e-4, 0.008, 1000))
    fast = m.sharpe_difference_ci(a, b, n_boot=300, block=21, seed=5)
    # Direct construction: glue random 21-day blocks into 1000-day histories.
    ex, t = np.c_[a, b], 1000
    starts = np.random.default_rng(5).integers(0, t, size=(300, int(np.ceil(t / 21))))
    idx = ((starts[:, :, None] + np.arange(21)) % t).reshape(300, -1)[:, :t]
    smp = ex[idx]
    sr = smp.mean(1) / smp.std(1, ddof=1) * np.sqrt(252)
    lo, hi = np.quantile(sr[:, 0] - sr[:, 1], [0.05, 0.95])
    assert fast["ci_low"] == pytest.approx(lo, abs=1e-12)
    assert fast["ci_high"] == pytest.approx(hi, abs=1e-12)
