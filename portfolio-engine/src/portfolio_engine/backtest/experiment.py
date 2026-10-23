"""Run a configured experiment and record exactly what produced it.

``run_experiment`` backtests every allocator under every estimation window
in the config. Within one window, all methods start on the same date (the
first quarter-end with a full window of history), so their rows cover
exactly the same days. A longer window starts later; to compare 3-year and
5-year windows fairly, summarise both over their common period with
``summary_table(..., period=(start, end))``.

``save_run`` writes the daily returns, the summary table and a ``run.json``
holding the config hash, the data-snapshot id, the seed and package
versions. Re-running the same config on the same snapshot must reproduce
the same numbers; the tests check that it does.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__
from ..allocation import Allocator, benchmark_allocators
from ..config import config_hash
from ..data.loader import MarketData
from . import metrics
from .costs import CostModel
from .engine import BacktestResult, first_rebalance, run_backtest


def _windows(cfg: dict[str, Any]) -> list[float]:
    w = cfg["backtest"]["window_years"]
    return list(w) if isinstance(w, (list, tuple)) else [w]


def run_experiment(cfg: dict[str, Any], data: MarketData,
                   allocators: dict[str, Allocator] | None = None,
                   costs: CostModel | None = None) -> dict[tuple[float, str], BacktestResult]:
    """Backtest each allocator for each estimation window. Keys are ``(window_years, name)``."""
    bt = cfg["backtest"]
    allocators = allocators or benchmark_allocators(cfg)
    costs = costs or CostModel.from_config(cfg, list(data.returns.columns))
    out = {}
    for w in _windows(cfg):
        start = first_rebalance(data.returns.index, w, bt.get("rebalance", "quarterly"))
        for name, alloc in allocators.items():
            out[(w, name)] = run_backtest(
                data.returns, alloc, costs, window_years=w,
                frequency=bt.get("rebalance", "quarterly"),
                start_in_cash=bt.get("start_in_cash", True), start=start, name=name)
    return out


def summary_table(results: dict[tuple[float, str], BacktestResult], data: MarketData,
                  cfg: dict[str, Any], period: tuple[Any, Any] | None = None) -> pd.DataFrame:
    """One row of metrics per (window, method), optionally restricted to a date range.

    ``period`` is how the hold-out is reported: the walk-forward run covers
    all dates, and only the hold-out dates are summarised.
    """
    ppy = cfg["backtest"].get("periods_per_year", 252)
    bench_name = cfg.get("report", {}).get("tracking_error_vs")
    rows = []
    for (w, name), res in results.items():
        r = res.returns
        turns = res.turnover.iloc[1:]
        if period is not None:
            lo, hi = (pd.Timestamp(p) if p is not None else None for p in period)
            r = r.loc[lo:hi]
            turns = turns.loc[lo:hi]
        bench = results.get((w, bench_name)).returns if (w, bench_name) in results else None
        row = metrics.summarize(r, data.rf, None if name == bench_name else bench,
                                annual_turnover=float(turns.sum() / (len(r) / ppy)), ppy=ppy)
        rows.append({"window_years": w, "method": name, **row,
                     "cost_drag_per_year": metrics.cagr(res.gross_returns.loc[r.index], ppy) - row["ann_return"]})
    return pd.DataFrame(rows).set_index(["window_years", "method"])


def save_run(results: dict[tuple[float, str], BacktestResult], summary: pd.DataFrame,
             cfg: dict[str, Any], data: MarketData, out_dir: str | Path = "results") -> Path:
    """Write returns, weights, summary and a provenance record to ``out_dir/<config>_<hash>/``."""
    h = config_hash(cfg)
    tag = "holdout" if data.includes_holdout else "dev"
    run_dir = Path(out_dir) / f"{cfg.get('name', 'run')}_{h}_{data.snapshot_id}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rets = pd.DataFrame({f"{n}|{w}y": r.returns for (w, n), r in results.items()})
    rets.to_parquet(run_dir / "daily_returns.parquet")
    weights = pd.concat({f"{n}|{w}y": r.weights for (w, n), r in results.items()}, names=["run", "date"])
    weights.to_parquet(run_dir / "weights.parquet")
    summary.to_csv(run_dir / "summary.csv")
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_name": cfg.get("name"), "config_hash": h, "config_file": cfg.get("_source"),
        "snapshot_id": data.snapshot_id, "seed": cfg.get("seed"),
        "includes_holdout": data.includes_holdout, "holdout_start": str(data.holdout_start.date()),
        "data_first": str(data.returns.index[0].date()), "data_last": str(data.returns.index[-1].date()),
        "package_version": __version__, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, default=str))
    return run_dir
