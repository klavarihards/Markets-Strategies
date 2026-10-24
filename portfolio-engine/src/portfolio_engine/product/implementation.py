"""Does the fund a client would actually buy behave like the research proxy?

The research uses long-history US-listed proxies; a Taiwan client would
hold Taiwan-listed funds. For Taiwan equity the snapshot has both: EWT
(iShares MSCI Taiwan, USD) and 0050 (Yuanta Taiwan Top 50, TWD). Monthly
returns are compared in TWD (EWT converted at USD/TWD) over the period both
exist. Monthly, not daily, because the two trade on different calendars and
in different time zones (Taipei closes before New York opens).

Reported: correlation, tracking error (volatility of the monthly return
difference, annualised), and the average yearly return gap. The two indexes
also differ (MSCI Taiwan includes mid-caps), so some gap is expected.

The Taiwan-listed series is first checked for unadjusted splits (repaired)
and for moves beyond Taiwan's daily price limit (flagged, see ``quality.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest import metrics
from ..data.quality import implausible_moves, repair_splits


def tracking_report(frames: dict[str, pd.DataFrame], proxy: str = "EWT", fund: str = "0050.TW") -> dict:
    """Monthly TWD returns of the proxy and the Taiwan-listed fund, and how closely they move."""
    px = frames["prices"][proxy].dropna()
    fx = frames["fred"]["usdtwd"].reindex(px.index.union(frames["fred"].index)).ffill().reindex(px.index)
    proxy_twd = (px * fx).resample("ME").last().pct_change()
    fund_px, repairs = repair_splits(frames["taiwan_listed"][fund])
    flagged = implausible_moves(fund_px, 0.10)
    fund_m = fund_px.resample("ME").last().pct_change()
    both = pd.DataFrame({"proxy": proxy_twd, "fund": fund_m}).dropna()
    diff = both["fund"] - both["proxy"]
    years = len(both) / 12
    return {
        "first": str(both.index[0].date()), "last": str(both.index[-1].date()), "months": len(both),
        "correlation": float(both.corr().iloc[0, 1]),
        "tracking_error": float(diff.std(ddof=1) * np.sqrt(12)),
        "proxy_ann_return": float((1 + both["proxy"]).prod() ** (1 / years) - 1),
        "fund_ann_return": float((1 + both["fund"]).prod() ** (1 / years) - 1),
        "proxy_max_drawdown": metrics.max_drawdown(both["proxy"]),
        "fund_max_drawdown": metrics.max_drawdown(both["fund"]),
        "data_repairs": repairs,
        "flagged_moves": {str(d.date()): round(float(v), 4) for d, v in flagged.items()},
        "monthly": both,
    }


# ---------------------------------------------------------------------------------------------
# The frozen portfolios, held with the real Taiwan-listed funds
# ---------------------------------------------------------------------------------------------

def fund_returns(funds: pd.DataFrame, mapping: dict[str, str], flag_limit: float = 0.10) -> tuple[pd.DataFrame, dict]:
    """Daily TWD returns of each sleeve's Taiwan-listed fund, after the data-quality checks.

    Returns the returns (columns = sleeves, Taiwan trading calendar) and a record
    of repaired splits and flagged moves per fund.
    """
    out, quality = {}, {}
    for sleeve, ticker in mapping.items():
        px, repairs = repair_splits(funds[ticker])
        flagged = implausible_moves(px, flag_limit)
        quality[ticker] = {"repairs": repairs, "flagged": {str(d.date()): round(float(v), 4) for d, v in flagged.items()}}
        out[sleeve] = px.pct_change()
    return pd.DataFrame(out), quality


def sleeve_drift(research_twd: pd.DataFrame, funds_twd: pd.DataFrame, start: str) -> pd.DataFrame:
    """How far each fund drifts from its research proxy (both in TWD), monthly, from ``start``.

    Monthly because the two trade on different calendars and in different time zones.
    """
    rows = {}
    for s in funds_twd.columns:
        a = metrics.monthly(research_twd[s].loc[start:].dropna())
        b = metrics.monthly(funds_twd[s].loc[start:].dropna())
        both = pd.DataFrame({"proxy": a, "fund": b}).dropna()
        yrs = len(both) / 12
        rows[s] = {"months": len(both), "correlation": both.corr().iloc[0, 1],
                   "tracking_error": (both["fund"] - both["proxy"]).std(ddof=1) * np.sqrt(12),
                   "proxy_ann_return": (1 + both["proxy"]).prod() ** (1 / yrs) - 1,
                   "fund_ann_return": (1 + both["fund"]).prod() ** (1 / yrs) - 1}
    t = pd.DataFrame(rows).T
    t["return_gap"] = t["fund_ann_return"] - t["proxy_ann_return"]
    return t


def replay_trades(returns: pd.DataFrame, targets: pd.DataFrame, costs, start: pd.Timestamp,
                  next_day: bool = False) -> pd.Series:
    """Hold a recorded sequence of target portfolios on another set of returns, with costs.

    ``targets`` are the weights a portfolio traded to, by decision date. With
    ``next_day=True`` each trade is executed on the first trading day *after*
    the decision date in ``returns`` (a decision at the New York close can only
    be traded in Taipei the next day). The portfolio starts in cash at the first
    target on or after ``start`` and pays to buy it.
    """
    idx = returns.index
    R = returns.to_numpy(float)
    sleeves = list(returns.columns)
    tgt = targets.reindex(columns=sleeves).fillna(0.0)
    tgt = tgt[tgt.index >= start]
    pos = idx.searchsorted(tgt.index, side="right" if next_day else "left")
    exec_at = {}
    for p, (_, row) in zip(pos, tgt.iterrows()):
        if p < len(idx):
            exec_at[p] = row.to_numpy()                 # if two decisions map to one day, the later wins
    first = min(exec_at)
    w = np.zeros(len(sleeves))
    if "cash" in sleeves:
        w[sleeves.index("cash")] = 1.0
    out = np.zeros(len(idx) - first)
    for k, i in enumerate(range(first, len(idx))):
        g = h = 0.0
        if k > 0:
            g = float(w @ R[i])
            h = costs.daily_holding_cost(w)
            w = w * (1 + R[i]) / (1 + g)
        t = 0.0
        if i in exec_at:
            new = exec_at[i]
            t = costs.trading_cost(w, new)
            w = new
        out[k] = (1 + g - h) * (1 - t) - 1
    return pd.Series(out, idx[first:], name="net")
