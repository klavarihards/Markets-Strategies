import json

import numpy as np
import pandas as pd
import pytest

from portfolio_engine.data import (SnapshotIntegrityError, holdout_start, load_market_data,
                                   make_synthetic_frames, read_snapshot, snapshot_id,
                                   to_base_currency, write_snapshot)


def test_synthetic_data_is_reproducible():
    a = make_synthetic_frames(end="2008-12-31", seed=3)
    b = make_synthetic_frames(end="2008-12-31", seed=3)
    c = make_synthetic_frames(end="2008-12-31", seed=4)
    pd.testing.assert_frame_equal(a["prices"], b["prices"])
    assert not a["prices"].equals(c["prices"])


def test_snapshot_round_trip_and_tamper_detection(tmp_path, frames):
    out = write_snapshot(frames, tmp_path / "snap", {"source": "synthetic"})
    loaded, manifest = read_snapshot(out)
    pd.testing.assert_frame_equal(loaded["prices"], frames["prices"], check_freq=False)
    assert manifest["files"]["prices"]["rows"] == len(frames["prices"])
    sid = snapshot_id(manifest)
    assert sid == snapshot_id(json.loads((out / "manifest.json").read_text()))

    # Change one value and save over the file: the loader must refuse it.
    bad = loaded["prices"].copy()
    bad.iloc[10, 0] *= 1.01
    bad.to_parquet(out / "prices.parquet")
    with pytest.raises(SnapshotIntegrityError):
        read_snapshot(out)


def test_holdout_is_locked_by_default(cfg, frames, tmp_path):
    dev = load_market_data(cfg, frames=frames, log_dir=tmp_path)
    full = load_market_data(cfg, frames=frames, unlock_holdout=True, log_dir=tmp_path)
    assert dev.returns.index.max() < dev.holdout_start
    assert full.returns.index.max() >= full.holdout_start
    assert dev.holdout_start == full.holdout_start
    # Roughly two years locked away.
    locked_days = (full.returns.index >= full.holdout_start).sum()
    assert 480 < locked_days < 530
    # Development data is an exact prefix of the full data.
    pd.testing.assert_frame_equal(dev.returns, full.returns.loc[dev.returns.index])
    # Opening the hold-out is logged; the locked load is not.
    log = (tmp_path / "holdout_log.jsonl").read_text().strip().splitlines()
    assert len(log) == 1


def test_holdout_start_date():
    assert holdout_start(pd.Timestamp("2026-09-30"), 2) == pd.Timestamp("2024-10-01")


def test_currency_conversion():
    r = pd.DataFrame({"a": [0.10, -0.05]})
    fx = pd.Series([-0.10, 0.02])
    out = to_base_currency(r, fx)
    assert out["a"].iloc[0] == pytest.approx(1.10 * 0.90 - 1)
    assert out["a"].iloc[1] == pytest.approx(0.95 * 1.02 - 1)


def test_sleeve_construction(cfg, frames, data):
    px = frames["prices"]
    tick = px.pct_change()
    d = data.returns_usd.index[50]
    # Blended sleeve = average of its two proxies' returns.
    assert data.returns_usd.loc[d, "intl_equity"] == pytest.approx(
        0.5 * tick.loc[d, "EFA"] + 0.5 * tick.loc[d, "EEM"])
    assert data.returns_usd.loc[d, "us_equity"] == pytest.approx(tick.loc[d, "SPY"])
    # Cash earns yesterday's rate, not today's.
    rate = frames["fred"]["cash_rate"].reindex(px.index).ffill()
    prev = px.index[px.index.get_loc(d) - 1]
    assert data.rf.loc[d] == pytest.approx(rate.loc[prev] / 100 / 252)
    assert (data.returns_usd["cash"] == data.rf).all()


def test_returns_are_clean(data):
    assert not data.returns.isna().any().any()
    assert list(data.returns.columns) == list(data.asset_class)
    assert data.returns.index.is_monotonic_increasing
    assert np.isfinite(data.returns.values).all()


def test_sleeve_can_be_marked_as_local_currency(cfg, frames):
    from portfolio_engine.config import override
    local = override(cfg, **{"sleeves.cash": {**cfg["sleeves"]["cash"], "currency": "TWD"}})
    a = load_market_data(cfg, frames=frames)
    b = load_market_data(local, frames=frames)
    pd.testing.assert_series_equal(b.returns["cash"], b.returns_usd["cash"])   # no FX on cash
    assert not a.returns["cash"].equals(a.returns_usd["cash"])
    pd.testing.assert_series_equal(a.returns["gold"], b.returns["gold"])      # others unchanged
