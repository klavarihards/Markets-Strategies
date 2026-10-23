"""Performance metrics, each with one definition used everywhere.

All functions take a Series of *daily* returns (0.01 = +1%) and annualise
with ``periods_per_year`` (252 trading days).

Definitions worth knowing by heart:

- **Annualised return (CAGR):** the constant yearly growth rate that would
  turn the starting value into the ending value. Geometric, so a +50% year
  followed by a -50% year is correctly a loss (1.5 x 0.5 = 0.75).
- **Volatility:** standard deviation of daily returns times sqrt(252).
  Daily risks add up like independent coin flips, so variance grows with
  time and standard deviation with its square root.
- **Sharpe ratio:** average daily return *in excess of cash*, divided by
  the standard deviation of that excess, times sqrt(252). It asks: how much
  return beyond the risk-free rate did each unit of risk earn? (Earlier
  notebooks in this repo sometimes used CAGR / volatility; that mixes a
  geometric mean with an arithmetic risk measure and ignores cash, so it is
  not used here.)
- **Sortino ratio:** like Sharpe, but only *downside* moves count as risk.
- **Maximum drawdown:** the worst fall from a previous peak, the loss a
  client who bought at the top would have seen.
- **Calmar ratio:** CAGR divided by the size of the maximum drawdown.
- **CVaR 95%:** the average return on the worst 5% of days (or months).
  VaR says where the bad tail starts; CVaR says how bad it is inside.
- **Tracking error:** volatility of the *difference* between a portfolio
  and its benchmark: how far it strays from it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PPY = 252


def cagr(r: pd.Series, ppy: int = PPY) -> float:
    """Compound annual growth rate."""
    years = len(r) / ppy
    growth = float((1 + r).prod())
    return growth ** (1 / years) - 1 if years > 0 and growth > 0 else float("nan")


def volatility(r: pd.Series, ppy: int = PPY) -> float:
    """Annualised standard deviation of returns."""
    return float(r.std(ddof=1) * np.sqrt(ppy))


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0, ppy: int = PPY) -> float:
    """Annualised Sharpe ratio of returns in excess of the risk-free rate."""
    ex = _excess(r, rf)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan")


def sortino(r: pd.Series, rf: pd.Series | float = 0.0, ppy: int = PPY) -> float:
    """Annualised Sortino ratio: excess return over downside deviation."""
    ex = _excess(r, rf)
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean())
    return float(ex.mean() / downside * np.sqrt(ppy)) if downside > 0 else float("nan")


def drawdown(r: pd.Series) -> pd.Series:
    """Fall from the running peak of the wealth curve, at every date (0 at a new high)."""
    wealth = (1 + r).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)   # the starting value counts as a peak
    return wealth / peak - 1


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough loss (a negative number)."""
    return float(drawdown(r).min())


def drawdown_stats(r: pd.Series) -> dict[str, Any]:
    """Depth and timing of the worst drawdown, and the longest time under water.

    ``recovery_days`` counts trading days from the trough of the worst
    drawdown until the old peak was regained; ``None`` if it never was.
    """
    dd = drawdown(r)
    trough = dd.idxmin()
    after = dd.loc[trough:]
    recovered = after[after >= 0]
    recovery_days = int(len(dd.loc[trough:recovered.index[0]]) - 1) if len(recovered) else None
    under = (dd < 0).astype(int).to_numpy()
    longest, run = 0, 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return {"max_drawdown": float(dd.min()), "trough_date": trough,
            "recovery_days": recovery_days, "longest_underwater_days": longest}


def calmar(r: pd.Series, ppy: int = PPY) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    mdd = max_drawdown(r)
    return cagr(r, ppy) / abs(mdd) if mdd < 0 else float("nan")


def cvar(r: pd.Series, alpha: float = 0.95) -> float:
    """Average return in the worst ``1 - alpha`` share of periods (historical CVaR, a negative number)."""
    cutoff = r.quantile(1 - alpha)
    return float(r[r <= cutoff].mean())


def monthly(r: pd.Series) -> pd.Series:
    """Compound daily returns into calendar-month returns."""
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1


def tracking_error(r: pd.Series, benchmark: pd.Series, ppy: int = PPY) -> float:
    """Annualised volatility of the return difference against a benchmark."""
    diff = (r - benchmark.reindex(r.index)).dropna()
    return volatility(diff, ppy)


def _excess(r: pd.Series, rf: pd.Series | float) -> pd.Series:
    if isinstance(rf, pd.Series):
        return (r - rf.reindex(r.index)).dropna()
    return r - rf


def summarize(r: pd.Series, rf: pd.Series | float = 0.0, benchmark: pd.Series | None = None,
              annual_turnover: float | None = None, ppy: int = PPY) -> dict[str, Any]:
    """Every standard metric for one return series, as a flat dict."""
    dds = drawdown_stats(r)
    out = {
        "start": r.index[0].date(), "end": r.index[-1].date(),
        "ann_return": cagr(r, ppy), "ann_vol": volatility(r, ppy),
        "sharpe": sharpe(r, rf, ppy), "sortino": sortino(r, rf, ppy),
        "max_drawdown": dds["max_drawdown"], "recovery_days": dds["recovery_days"],
        "longest_underwater_days": dds["longest_underwater_days"],
        "calmar": calmar(r, ppy),
        "cvar95_daily": cvar(r, 0.95), "cvar95_monthly": cvar(monthly(r), 0.95),
    }
    if annual_turnover is not None:
        out["turnover_per_year"] = annual_turnover
    if benchmark is not None:
        out["tracking_error"] = tracking_error(r, benchmark, ppy)
    return out
