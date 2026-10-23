"""Data-quality checks: catching what free data sources get wrong.

Free price data has known failure modes. The one that matters here: a stock
split that the provider has not applied to the earlier history. The price
then appears to fall 75% in a day (a 4-for-1 split) when nothing happened to
the investor. Found in Yahoo's 0050.TW series (2014-01-02, a ratio of
0.249).

``repair_splits`` finds one-day price ratios within 2% of 1/k or k (k = 2 to
10) and rescales all earlier prices by that ratio, so the series is
continuous. ``implausible_moves`` lists any other daily move larger than the
exchange could allow (Taiwan's daily price limit was 7% until June 2015 and
10% since); these are flagged for a human, never silently changed.
"""

from __future__ import annotations

import pandas as pd


def repair_splits(prices: pd.Series, tolerance: float = 0.02) -> tuple[pd.Series, list[dict]]:
    """Undo unadjusted splits. Returns the repaired series and a record of each repair."""
    p = prices.dropna().copy()
    repairs = []
    ratio = p / p.shift(1)
    for date, r in ratio.dropna().items():
        for k in range(2, 11):
            for factor in (1 / k, k):
                if abs(r / factor - 1) < tolerance:
                    before = p.index < date
                    p[before] = p[before] * r
                    kind = f"{k}-for-1" if factor < 1 else f"1-for-{k}"
                    repairs.append({"date": str(date.date()), "ratio": float(r), "split": kind})
                    break
            else:
                continue
            break
    return p, repairs


def implausible_moves(prices: pd.Series, limit: float) -> pd.Series:
    """Daily moves beyond ``limit`` in either direction (e.g. an exchange's price limit)."""
    r = prices.dropna().pct_change()
    return r[r.abs() > limit]
