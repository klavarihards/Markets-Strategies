"""Phase 1 gate: baseline numbers reproduce exactly from a clean run."""

import json

import pandas as pd

from portfolio_engine.backtest import run_experiment, save_run, summary_table
from portfolio_engine.config import config_hash, override
from portfolio_engine.data import load_market_data, make_synthetic_frames, write_snapshot
from portfolio_engine.data.snapshot import file_sha256


def _run(cfg, snap, out):
    data = load_market_data(cfg, snapshot_dir=snap, log_dir=out)
    res = run_experiment(cfg, data)
    summ = summary_table(res, data, cfg)
    return save_run(res, summ, cfg, data, out, registry=out / "registry.csv"), summ


def test_clean_runs_reproduce_exactly(cfg, tmp_path):
    snap = write_snapshot(make_synthetic_frames(end="2015-12-31", seed=11), tmp_path / "snap")
    d1, s1 = _run(cfg, snap, tmp_path / "run1")
    d2, s2 = _run(cfg, snap, tmp_path / "run2")
    pd.testing.assert_frame_equal(s1, s2, check_exact=True)
    assert file_sha256(d1 / "daily_returns.parquet") == file_sha256(d2 / "daily_returns.parquet")
    rec = json.loads((d1 / "run.json").read_text())
    assert rec["config_hash"] == config_hash(cfg)
    assert rec["includes_holdout"] is False
    assert set(s1.index.get_level_values("method")) == {"equal_weight", "sixty_forty", "taiwan_equity_only"}


def test_config_hash_tracks_settings(cfg):
    assert config_hash(cfg) == config_hash(dict(cfg))
    assert config_hash(cfg) != config_hash(override(cfg, **{"backtest.window_years": [5]}))
    assert cfg["backtest"]["window_years"] == [3, 5]   # override never mutates the original


def test_costs_lower_returns(cfg, data):
    from portfolio_engine.backtest import CostModel
    res = run_experiment(cfg, data)
    free = run_experiment(cfg, data, costs=CostModel.zero(list(data.returns.columns)))
    key = (3, "sixty_forty")
    assert (1 + res[key].returns).prod() < (1 + free[key].returns).prod()


def test_evaluate_holdout_reports_only_holdout_dates(cfg, tmp_path):
    from portfolio_engine.backtest import evaluate_holdout
    snap = write_snapshot(make_synthetic_frames(end="2015-12-31", seed=11), tmp_path / "snap")
    summ, res, run_dir = evaluate_holdout(cfg, snapshot_dir=snap, out_dir=tmp_path / "final",
                                         registry=tmp_path / "registry.csv")
    rec = json.loads((run_dir / "run.json").read_text())
    assert rec["includes_holdout"] is True
    assert (summ["start"] >= pd.Timestamp(rec["holdout_start"]).date()).all()
    assert (tmp_path / "final" / "holdout_log.jsonl").exists()


def test_runs_are_registered(cfg, tmp_path):
    import pandas as pd
    from portfolio_engine.backtest.registry import development_trials
    snap = write_snapshot(make_synthetic_frames(end="2015-12-31", seed=11), tmp_path / "snap")
    d1, _ = _run(cfg, snap, tmp_path / "run1")
    reg = pd.read_csv(tmp_path / "run1" / "registry.csv")
    assert len(reg) == 6 and (reg["tag"] == "dev").all()          # 3 benchmarks x 2 windows
    _run(cfg, snap, tmp_path / "run1")                            # same run again: refreshed, not duplicated
    assert len(pd.read_csv(tmp_path / "run1" / "registry.csv")) == 6
    assert len(development_trials(tmp_path / "run1" / "registry.csv")) == 6
