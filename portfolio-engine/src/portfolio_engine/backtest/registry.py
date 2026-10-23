"""Trial registry: every configuration ever run, kept in git.

Overfitting rarely comes from one dishonest step. It comes from trying
many reasonable-looking variants and reporting the best. The defence is to
*count*: every saved run (each method, profile, window and setting) is
appended to ``experiments/registry.csv`` with its Sharpe ratio (on returns
in excess of cash), skewness
and kurtosis. The deflated Sharpe ratio then uses that count to ask whether
the best result is better than what the same number of random trials would
have produced. Runs that touched the hold-out are recorded with tag
``holdout`` and never count as development trials.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REGISTRY = Path("experiments/registry.csv")
KEY = ["config_hash", "snapshot_id", "tag", "run"]


def register_run(run_dir: str | Path, registry: str | Path = REGISTRY) -> pd.DataFrame:
    """Append (or refresh) one row per backtest of a saved run directory."""
    run_dir = Path(run_dir)
    rec = json.loads((run_dir / "run.json").read_text())
    rets = pd.read_parquet(run_dir / "daily_returns.parquet")
    # Sharpe ratios are on returns in excess of cash, as everywhere else in the engine.
    rf = pd.read_parquet(run_dir / "rf.parquet")["rf"] if (run_dir / "rf.parquet").exists() else 0.0
    rows = []
    for col in rets.columns:
        r = (rets[col] - (rf.reindex(rets.index) if isinstance(rf, pd.Series) else rf)).dropna()
        rows.append({"recorded_utc": rec["created_utc"], "config_name": rec["config_name"],
                     "config_hash": rec["config_hash"], "snapshot_id": rec["snapshot_id"],
                     "tag": "holdout" if rec["includes_holdout"] else "dev", "run": col,
                     "first": str(r.index[0].date()), "last": str(r.index[-1].date()), "n_days": len(r),
                     "sharpe_daily": float(r.mean() / r.std(ddof=1)),
                     "sharpe_annual": float(r.mean() / r.std(ddof=1) * np.sqrt(252)),
                     "skew": float(r.skew()), "kurtosis": float(r.kurt() + 3)})
    new = pd.DataFrame(rows)
    registry = Path(registry)
    if registry.exists():
        old = pd.read_csv(registry)
        old = old[~old.set_index(KEY).index.isin(new.set_index(KEY).index)]
        new = pd.concat([old, new], ignore_index=True)
    registry.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(registry, index=False)
    return new


def development_trials(registry: str | Path = REGISTRY, snapshot_id: str | None = None) -> pd.DataFrame:
    """All development (non-hold-out) trials, optionally for one data snapshot."""
    reg = pd.read_csv(registry)
    reg = reg[reg["tag"] == "dev"]
    return reg[reg["snapshot_id"] == snapshot_id] if snapshot_id else reg
