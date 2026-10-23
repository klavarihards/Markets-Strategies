"""Run a configured experiment and record exactly what produced it.

``run_experiment`` backtests every allocator under every estimation window
in the config. Within one window, all methods start on the same date (the
first quarter-end with a full window of history), so their rows cover
exactly the same days. A longer window starts later; to compare 3-year and
5-year windows fairly, summarise both over their common period with
``summary_table(..., period=(start, end))``.

Names of profile methods have the form ``"profile:method"`` (for example
``"balanced:hrp"``); benchmarks have no profile. The tables split the two,
so results can be read profile by profile.

``save_run`` writes the daily returns, the summary table and a ``run.json``
holding the config hash, the data-snapshot id, the seed and package
versions. Re-running the same config on the same snapshot must reproduce
the same numbers; the tests check that it does.
"""

from __future__ import annotations

import copy
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__
from ..allocation import Allocator, RiskProfile, benchmark_allocators, build_method_allocators
from ..config import config_hash
from ..data.loader import MarketData
from . import metrics
from .costs import CostModel
from .engine import BacktestResult, first_rebalance, run_backtest


def _windows(cfg: dict[str, Any]) -> list[float]:
    w = cfg["backtest"]["window_years"]
    return list(w) if isinstance(w, (list, tuple)) else [w]


def default_allocators(cfg: dict[str, Any]) -> dict[str, Allocator]:
    """The benchmarks, plus every profile x method pair if the config defines profiles."""
    out: dict[str, Allocator] = dict(benchmark_allocators(cfg))
    if "profiles" in cfg and "methods" in cfg:
        out.update(build_method_allocators(cfg))
    return out


def split_name(name: str) -> tuple[str, str]:
    """``"balanced:hrp"`` -> ``("balanced", "hrp")``; a benchmark -> ``("benchmark", name)``."""
    return tuple(name.split(":", 1)) if ":" in name else ("benchmark", name)


def run_experiment(cfg: dict[str, Any], data: MarketData,
                   allocators: dict[str, Allocator] | None = None,
                   costs: CostModel | None = None) -> dict[tuple[float, str], BacktestResult]:
    """Backtest each allocator for each estimation window. Keys are ``(window_years, name)``.

    Each run gets a fresh copy of its allocator, so diagnostics logged by a
    method (e.g. dates where the volatility cap was relaxed) belong to that
    run only; they are kept in ``result.meta["log"]``.
    """
    bt = cfg["backtest"]
    allocators = allocators or default_allocators(cfg)
    costs = costs or CostModel.from_config(cfg, list(data.returns.columns))
    out = {}
    for w in _windows(cfg):
        start = first_rebalance(data.returns.index, w, bt.get("rebalance", "quarterly"))
        for name, alloc in allocators.items():
            fresh = copy.deepcopy(alloc)
            res = run_backtest(
                data.returns, fresh, costs, window_years=w,
                frequency=bt.get("rebalance", "quarterly"),
                start_in_cash=bt.get("start_in_cash", True), start=start, name=name)
            if hasattr(fresh, "log"):
                res.meta["log"] = pd.DataFrame(fresh.log).set_index("date")
            out[(w, name)] = res
    return out


def _period(r: pd.Series, period: tuple[Any, Any] | None) -> pd.Series:
    if period is None:
        return r
    lo, hi = (pd.Timestamp(p) if p is not None else None for p in period)
    return r.loc[lo:hi]


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
        r = _period(res.returns, period)
        turns = res.turnover.iloc[1:].loc[r.index[0]:r.index[-1]]
        bench = results.get((w, bench_name)).returns if (w, bench_name) in results else None
        row = metrics.summarize(r, data.rf, None if name == bench_name else bench,
                                annual_turnover=float(turns.sum() / (len(r) / ppy)), ppy=ppy)
        profile, method = split_name(name)
        rows.append({"window_years": w, "profile": profile, "method": method, **row,
                     "cost_drag_per_year": metrics.cagr(res.gross_returns.loc[r.index], ppy) - row["ann_return"]})
    return pd.DataFrame(rows).set_index(["window_years", "profile", "method"])


def compliance_table(results: dict[tuple[float, str], BacktestResult], data: MarketData,
                     cfg: dict[str, Any], period: tuple[Any, Any] | None = None) -> pd.DataFrame:
    """Did each profile portfolio behave like the client was promised?

    For each profile method: realised volatility against the profile's band,
    worst drawdown against the client's maximum acceptable loss, how often
    the volatility cap had to be dropped, and (for the methods that are
    moved onto the profile) how far the profile moved them on average.
    """
    rows = []
    for (w, name), res in results.items():
        profile_name, method = split_name(name)
        if profile_name == "benchmark":
            continue
        prof = RiskProfile.from_config(cfg, profile_name)
        r = _period(res.returns, period)
        log = res.meta.get("log", pd.DataFrame())
        log = log.loc[r.index[0]:r.index[-1]] if len(log) else log
        vol, mdd = metrics.volatility(r), metrics.max_drawdown(r)
        rows.append({
            "window_years": w, "profile": profile_name, "method": method,
            "realised_vol": vol, "vol_band_low": prof.vol_band[0], "vol_band_high": prof.vol_band[1],
            "vol_in_band": prof.vol_band[0] <= vol <= prof.vol_band[1],
            "max_drawdown": mdd, "max_loss_limit": prof.max_loss, "max_loss_breached": mdd < prof.max_loss,
            "avg_ex_ante_vol": float(log["ex_ante_vol"].mean()) if len(log) else np.nan,
            "share_vol_cap_relaxed": float(log["vol_cap_relaxed"].mean()) if len(log) else np.nan,
            "avg_moved_by_profile": float(log["moved_by_profile"].mean())
            if "moved_by_profile" in log else np.nan,
        })
    return pd.DataFrame(rows).set_index(["window_years", "profile", "method"])


def exposure_table(results: dict[tuple[float, str], BacktestResult], data: MarketData) -> pd.DataFrame:
    """Average target weight per asset class over all rebalances (what each method really holds)."""
    rows = []
    for (w, name), res in results.items():
        profile, method = split_name(name)
        by_class = res.weights.T.groupby(pd.Series(data.asset_class)).sum().T.mean()
        rows.append({"window_years": w, "profile": profile, "method": method, **by_class.to_dict(),
                     "max_single_sleeve": float(res.weights.max(axis=1).mean())})
    return pd.DataFrame(rows).set_index(["window_years", "profile", "method"])


def sharpe_comparison_table(results: dict[tuple[float, str], BacktestResult], data: MarketData,
                            cfg: dict[str, Any], versus: tuple[str, ...] = ("equal_weight", "sixty_forty"),
                            period: tuple[Any, Any] | None = None) -> pd.DataFrame:
    """Sharpe difference of every profile method against each benchmark, with a bootstrap interval.

    ``significant`` is True only when the whole interval is on one side of
    zero. Most rows are expected to be False: that is the honest answer when
    the data cannot separate two methods.
    """
    bcfg = cfg.get("bootstrap", {})
    rows = []
    for (w, name), res in results.items():
        profile, method = split_name(name)
        if profile == "benchmark":
            continue
        r = _period(res.returns, period)
        for b in versus:
            if (w, b) not in results:
                continue
            rb = results[(w, b)].returns.reindex(r.index)
            ci = metrics.sharpe_difference_ci(r, rb, data.rf, n_boot=bcfg.get("n", 2000),
                                              block=bcfg.get("block_days", 21), ci=bcfg.get("ci", 0.90),
                                              seed=cfg.get("seed", 0))
            rows.append({"window_years": w, "profile": profile, "method": method, "versus": b, **ci,
                         "significant": ci["ci_low"] > 0 or ci["ci_high"] < 0})
    return pd.DataFrame(rows).set_index(["window_years", "profile", "method", "versus"])


def save_run(results: dict[tuple[float, str], BacktestResult], summary: pd.DataFrame,
             cfg: dict[str, Any], data: MarketData, out_dir: str | Path = "results",
             tables: dict[str, pd.DataFrame] | None = None, register: bool = True,
             registry: str | Path = "experiments/registry.csv") -> Path:
    """Write returns, weights, tables and a provenance record to ``out_dir/<config>_<hash>_.../``.

    Every saved run is also appended to the trial registry (see ``registry.py``).
    """
    h = config_hash(cfg)
    tag = "holdout" if data.includes_holdout else "dev"
    run_dir = Path(out_dir) / f"{cfg.get('name', 'run')}_{h}_{data.snapshot_id}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rets = pd.DataFrame({f"{n}|{w}y": r.returns for (w, n), r in results.items()})
    rets.to_parquet(run_dir / "daily_returns.parquet")
    weights = pd.concat({f"{n}|{w}y": r.weights for (w, n), r in results.items()}, names=["run", "date"])
    weights.to_parquet(run_dir / "weights.parquet")
    summary.to_csv(run_dir / "summary.csv")
    for tname, table in (tables or {}).items():
        table.to_csv(run_dir / f"{tname}.csv")
    logs = {f"{n}|{w}y": r.meta["log"] for (w, n), r in results.items() if "log" in r.meta}
    if logs:
        pd.concat(logs, names=["run", "date"]).to_parquet(run_dir / "method_diagnostics.parquet")
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
    if register:
        from .registry import register_run
        register_run(run_dir, registry)
    return run_dir


def evaluate_holdout(cfg: dict[str, Any], allocators: dict[str, Allocator] | None = None,
                     snapshot_dir: str | Path | None = None, out_dir: str | Path = "results",
                     registry: str | Path = "experiments/registry.csv"):
    """The final, one-time evaluation on the locked hold-out period.

    Loads the full snapshot (this is the only place in the package that
    unlocks it, and the unlock is logged), runs the same walk-forward
    backtest over all dates, and summarises only the hold-out dates. Because
    the backtest only ever looks backwards, the portfolio on the first
    hold-out day is exactly the one the development run would have held.

    Call this once, after every method and setting has been frozen.
    """
    from ..data.loader import load_market_data

    data = load_market_data(cfg, snapshot_dir=snapshot_dir, unlock_holdout=True, log_dir=out_dir)
    results = run_experiment(cfg, data, allocators)
    period = (data.holdout_start, None)
    summary = summary_table(results, data, cfg, period=period)
    tables = {}
    if "profiles" in cfg:
        tables = {"compliance": compliance_table(results, data, cfg, period=period),
                  "sharpe_comparison": sharpe_comparison_table(results, data, cfg, period=period)}
    run_dir = save_run(results, summary, cfg, data, out_dir, tables, registry=registry)
    return summary, results, run_dir
