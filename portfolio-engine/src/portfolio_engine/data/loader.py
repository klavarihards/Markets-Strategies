"""Turn a raw snapshot into daily sleeve returns, with the hold-out locked.

Three jobs:

1. **Sleeve returns.** Each of the seven building blocks gets one daily
   return series from its proxy ticker(s). Cash earns the T-bill rate.
2. **Currency.** The client spends TWD, so a USD asset's return to them is
   its USD return plus the change in the USD/TWD rate:
   ``(1 + r_usd) * (1 + r_fx) - 1``. A year where SPY gains 10% but the
   dollar falls 10% against TWD is a year of roughly zero for the client.
3. **The hold-out lock.** The last ``holdout_years`` of data are cut off
   before anything else sees them. Research is done on what remains. Only
   ``unlock_holdout=True`` returns the full history, and every unlock is
   written to a log file, so it is visible how often the hold-out was opened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .snapshot import read_snapshot, snapshot_id

# Gaps longer than this many trading days are treated as missing data, not holidays.
_MAX_FILL = 5


@dataclass
class MarketData:
    """Everything a backtest needs, already aligned on one trading calendar.

    ``returns`` are in the base currency (TWD by default); ``returns_usd``
    are the same sleeves before currency conversion. ``rf`` is the daily
    risk-free return (the T-bill yield, before currency conversion), used to
    compute excess returns in the Sharpe ratio.
    """

    returns: pd.DataFrame
    returns_usd: pd.DataFrame
    rf: pd.Series
    fx: pd.Series
    holdout_start: pd.Timestamp
    includes_holdout: bool
    snapshot_id: str
    asset_class: dict[str, str]


def sleeve_returns_usd(prices: pd.DataFrame, fred: pd.DataFrame,
                       sleeves: dict[str, Any], periods_per_year: int = 252) -> tuple[pd.DataFrame, pd.Series]:
    """Daily USD return of every sleeve, plus the daily risk-free return.

    - Ticker returns are simple daily returns of adjusted closes.
    - A blended sleeve (e.g. half EFA, half EEM) is the weighted average of
      its tickers' daily returns, i.e. a blend rebalanced back to its proxy
      weights every day. That is a small simplification.
    - Cash earns yesterday's T-bill rate spread evenly over trading days.
      Yesterday's, because today's rate is only known at today's close.
    """
    prices = prices.sort_index().ffill(limit=_MAX_FILL)
    first_full = prices.dropna().index[0]
    prices = prices.loc[first_full:]
    tick_ret = prices.pct_change().iloc[1:]

    rate = fred["cash_rate"].reindex(prices.index.union(fred.index)).ffill(limit=_MAX_FILL)
    rate = rate.reindex(prices.index)
    rf = (rate.shift(1) / 100.0 / periods_per_year).reindex(tick_ret.index)
    rf.name = "rf"

    out = {}
    for name, spec in sleeves.items():
        proxy = spec["proxy"]
        if proxy == "cash_rate":
            out[name] = rf
        else:
            out[name] = sum(w * tick_ret[t] for t, w in proxy.items())
    return pd.DataFrame(out), rf


def fx_returns(fred: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Daily % change of TWD per USD on the trading calendar (positive = USD strengthened)."""
    level = fred["usdtwd"].reindex(fred.index.union(index)).ffill(limit=_MAX_FILL).reindex(index)
    r = level.pct_change()
    r.name = "usdtwd"
    return r


def to_base_currency(returns_usd: pd.DataFrame, fx: pd.Series) -> pd.DataFrame:
    """Convert USD returns into TWD returns: ``(1 + r) * (1 + r_fx) - 1``."""
    return (1 + returns_usd).mul(1 + fx, axis=0) - 1


def holdout_start(last_date: pd.Timestamp, years: int) -> pd.Timestamp:
    """First day of the locked period: ``years`` before the last date in the snapshot."""
    return (pd.Timestamp(last_date) - pd.DateOffset(years=years) + pd.Timedelta(days=1)).normalize()


def _log_unlock(log_dir: Path, info: dict[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "holdout_log.jsonl", "a") as f:
        f.write(json.dumps(info) + "\n")


def load_market_data(cfg: dict[str, Any], snapshot_dir: str | Path | None = None,
                     frames: dict[str, pd.DataFrame] | None = None,
                     unlock_holdout: bool = False,
                     log_dir: str | Path = "results") -> MarketData:
    """Build :class:`MarketData` from a snapshot folder (or from ``frames`` directly, for tests).

    By default the hold-out period is removed. Pass ``unlock_holdout=True``
    only in the final evaluation cell; the unlock is logged with a timestamp.
    """
    if frames is None:
        frames, manifest = read_snapshot(snapshot_dir or cfg["data"]["snapshot_dir"])
        sid = snapshot_id(manifest)
    else:
        sid = "in-memory"
    ppy = cfg["backtest"].get("periods_per_year", 252)
    ret_usd, rf = sleeve_returns_usd(frames["prices"], frames["fred"], cfg["sleeves"], ppy)
    fx = fx_returns(frames["fred"], ret_usd.index)

    base = cfg.get("base_currency", "TWD").upper()
    ret = to_base_currency(ret_usd, fx) if base == "TWD" else ret_usd.copy()

    valid = ret.notna().all(axis=1) & fx.notna() & rf.notna()
    first = valid.idxmax()
    keep = ret.index >= first
    ret, ret_usd, rf, fx = ret[keep], ret_usd[keep], rf[keep], fx[keep]
    if ret.isna().any().any():
        bad = ret.columns[ret.isna().any()].tolist()
        raise ValueError(f"Gaps longer than {_MAX_FILL} days in the data for sleeves {bad}")

    h_start = holdout_start(ret.index[-1], cfg["data"].get("holdout_years", 2))
    if unlock_holdout:
        _log_unlock(Path(log_dir), {
            "opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "snapshot_id": sid, "holdout_start": str(h_start.date()),
            "config": cfg.get("name"),
        })
    else:
        dev = ret.index < h_start
        ret, ret_usd, rf, fx = ret[dev], ret_usd[dev], rf[dev], fx[dev]

    asset_class = {k: v["asset_class"] for k, v in cfg["sleeves"].items()}
    return MarketData(ret, ret_usd, rf, fx, h_start, unlock_holdout, sid, asset_class)
