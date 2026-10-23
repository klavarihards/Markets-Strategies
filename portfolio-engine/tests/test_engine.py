import numpy as np
import pandas as pd
import pytest

from portfolio_engine.allocation import FixedWeights
from portfolio_engine.backtest import CostModel, first_rebalance, rebalance_dates, run_backtest


class Spy:
    """Allocator that records what it was shown, and alternates its answer."""

    def __init__(self):
        self.calls = []

    def __call__(self, history):
        self.calls.append(history.index.max())
        w = pd.Series(0.0, index=history.columns)
        # A 'signal' computed from the history it was given, so leaks would show up.
        best = history.iloc[-60:].mean().idxmax()
        w[best] = 0.5
        w[history.columns[-1]] += 0.5
        return w


def test_rebalance_dates_are_last_trading_day_of_quarter():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    d = rebalance_dates(idx)
    assert list(d) == list(pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"]))
    # Data stopping mid-quarter: that quarter has no rebalance yet.
    d2 = rebalance_dates(idx[idx <= "2020-11-15"])
    assert d2[-1] == pd.Timestamp("2020-09-30")


def test_allocator_never_sees_the_future(data, cfg):
    spy = Spy()
    costs = CostModel.from_config(cfg, list(data.returns.columns))
    res = run_backtest(data.returns, spy, costs, window_years=3)
    assert list(res.weights.index) == spy.calls          # called exactly on rebalance dates...
    assert all(seen <= d for seen, d in zip(spy.calls, res.weights.index))  # ...with nothing after


def test_changing_the_future_does_not_change_the_past(data, cfg):
    costs = CostModel.from_config(cfg, list(data.returns.columns))
    cut = data.returns.index[len(data.returns) * 2 // 3]
    shocked = data.returns.copy()
    shocked.loc[shocked.index > cut] *= -3.0
    a = run_backtest(data.returns, Spy(), costs, window_years=3)
    b = run_backtest(shocked, Spy(), costs, window_years=3)
    pd.testing.assert_series_equal(a.returns.loc[:cut], b.returns.loc[:cut])
    pd.testing.assert_frame_equal(a.weights.loc[:cut], b.weights.loc[:cut])
    assert not a.returns.loc[cut:].equals(b.returns.loc[cut:])


def test_hold_out_run_extends_the_development_run(cfg, frames, tmp_path):
    from portfolio_engine.data import load_market_data
    dev = load_market_data(cfg, frames=frames, log_dir=tmp_path)
    full = load_market_data(cfg, frames=frames, unlock_holdout=True, log_dir=tmp_path)
    costs = CostModel.from_config(cfg, list(dev.returns.columns))
    a = run_backtest(dev.returns, Spy(), costs, window_years=3)
    b = run_backtest(full.returns, Spy(), costs, window_years=3)
    pd.testing.assert_series_equal(a.returns, b.returns.loc[a.returns.index])


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame({"a": rng.normal(0.0005, 0.01, n), "b": rng.normal(0.0002, 0.005, n),
                         "cash": 0.0001}, index=idx)


def test_buy_and_hold_drift_matches_hand_calculation():
    r = _toy()
    res = run_backtest(r, FixedWeights({"a": 0.5, "b": 0.5}), CostModel.zero(list(r.columns)),
                       window_years=0.25)
    reb = list(res.weights.index)
    d0, d1 = reb[0], reb[1]
    # Over one quarter, a 50/50 portfolio left alone grows like the average of the two growths.
    seg = r.loc[d0:d1].iloc[1:]
    expected = 0.5 * (1 + seg["a"]).prod() + 0.5 * (1 + seg["b"]).prod()
    got = (1 + res.returns.loc[seg.index]).prod()
    assert got == pytest.approx(expected, rel=1e-12)


def test_single_asset_without_costs_tracks_the_asset():
    r = _toy()
    res = run_backtest(r, FixedWeights({"a": 1.0}), CostModel.zero(list(r.columns)), window_years=0.25)
    pd.testing.assert_series_equal(res.returns.iloc[1:], r["a"].loc[res.returns.index[1:]],
                                   check_names=False)
    assert res.returns.iloc[0] == 0


def test_costs_are_charged_correctly():
    r = _toy()
    cost_cfg = {"a": {"fund_fee": 0.01, "half_spread": 0.002, "sell_tax": 0.001}, "b": {}, "cash": {}}
    costs = CostModel.build(cost_cfg, list(r.columns))
    res = run_backtest(r, FixedWeights({"a": 1.0}), costs, window_years=0.25)
    # Day 0: buy 100% of 'a' from cash, paying the half-spread.
    assert res.returns.iloc[0] == pytest.approx(-0.002)
    # Later days: asset return minus 1%/252 fee (no trading needed when weights don't drift).
    d = res.returns.index[5]
    assert res.returns.loc[d] == pytest.approx(r.loc[d, "a"] - 0.01 / 252)
    assert res.turnover.iloc[1:].sum() == pytest.approx(0)


def test_not_enough_history_raises():
    r = _toy(100)
    with pytest.raises(ValueError):
        first_rebalance(r.index, 3)
