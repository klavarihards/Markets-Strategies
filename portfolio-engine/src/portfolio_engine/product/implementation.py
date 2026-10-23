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
