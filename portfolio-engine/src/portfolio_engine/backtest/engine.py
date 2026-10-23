"""Walk-forward backtest: rebalance quarterly, using only the past.

How a backtest day works:

1. The portfolio holds weights ``w``. Today's return is ``w @ r``.
2. Fees and withholding for the day are deducted.
3. Prices moved, so the weights drift: an asset that rose now makes up more
   of the portfolio. ``w`` is updated to the drifted weights.
4. On a rebalance date (last trading day of each quarter), the allocator
   is shown the returns from the last ``window_years`` up to and including
   today, and returns new target weights. The trades from the drifted
   weights to the target are paid for (spread, commission, tax, FX), and
   the portfolio now holds the target.

Because the allocator only ever receives data up to the rebalance date,
nothing the backtest reports can depend on the future. That is the whole
point of "walk-forward": every decision is made the way it would have been
made at the time. (Trading is assumed to happen at that day's close, using
that close in the estimate. A stricter variant would trade a day later.)

Every method starts on the same date, the first quarter-end with a full
estimation window behind it, so all methods are compared over exactly the
same days. The first rebalance buys the whole portfolio from cash and pays
the costs of doing so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..allocation.base import Allocator, validate_weights
from .costs import CostModel


@dataclass
class BacktestResult:
    """Output of one walk-forward run.

    - ``returns``: daily portfolio returns after all costs (the number that matters)
    - ``gross_returns``: the same before any costs, to show what costs took away
    - ``weights``: target weights chosen at each rebalance date
    - ``turnover``: one-way turnover at each rebalance, i.e. half the sum of
      absolute weight changes (0.10 = 10% of the portfolio was traded)
    - ``trading_costs`` / ``holding_costs``: daily cost series
    """

    name: str
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    trading_costs: pd.Series
    holding_costs: pd.Series
    window_years: float
    meta: dict = field(default_factory=dict)

    def annual_turnover(self, periods_per_year: int = 252) -> float:
        """Average one-way turnover per year, excluding the initial purchase."""
        years = len(self.returns) / periods_per_year
        return float(self.turnover.iloc[1:].sum() / years) if years > 0 else float("nan")


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "quarterly") -> pd.DatetimeIndex:
    """Last trading day of each period in ``index``.

    Using the last *trading* day (not the calendar quarter-end) matters: the
    calendar date may be a weekend, when no trade can happen.

    If the data stops part-way through a period, that period has no
    rebalance yet: its last available day is not its last trading day.
    """
    period = {"quarterly": "Q", "monthly": "M", "annual": "Y"}[frequency]
    s = pd.Series(index, index=index)
    periods = index.to_period(period)
    last = s.groupby(periods).max()
    end_of_data = index[-1]
    period_end = periods[-1].end_time.normalize()
    if len(pd.bdate_range(end_of_data + pd.Timedelta(days=1), period_end)) > 0:
        last = last.iloc[:-1]
    return pd.DatetimeIndex(last.values)


def first_rebalance(index: pd.DatetimeIndex, window_years: float, frequency: str = "quarterly") -> pd.Timestamp:
    """First rebalance date with at least ``window_years`` of history before it."""
    earliest = index[0] + pd.DateOffset(months=int(round(window_years * 12)))
    dates = rebalance_dates(index, frequency)
    eligible = dates[dates >= earliest]
    if len(eligible) == 0:
        raise ValueError(f"Not enough data for a {window_years}-year estimation window")
    return eligible[0]


def run_backtest(returns: pd.DataFrame, allocator: Allocator, costs: CostModel,
                 window_years: float = 3, frequency: str = "quarterly",
                 start_in_cash: bool = True, cash_sleeve: str = "cash",
                 start: pd.Timestamp | None = None, name: str | None = None) -> BacktestResult:
    """Simulate holding the allocator's portfolio through ``returns``.

    ``start`` lets several runs share one start date (e.g. the latest start
    needed by any window length); by default the run starts at the first
    rebalance with a full window.
    """
    sleeves = list(returns.columns)
    if tuple(sleeves) != costs.sleeves:
        raise ValueError("Cost model sleeves do not match the return columns")
    idx = returns.index
    R = returns.to_numpy(dtype=float)
    first = first_rebalance(idx, window_years, frequency)
    if start is not None:
        later = rebalance_dates(idx, frequency)
        later = later[later >= max(first, pd.Timestamp(start))]
        if len(later) == 0:
            raise ValueError(f"No rebalance date on or after {start}")
        first = later[0]
    reb = set(rebalance_dates(idx, frequency))
    i0 = idx.get_loc(first)
    window = pd.DateOffset(months=int(round(window_years * 12)))

    w = np.zeros(len(sleeves))
    if start_in_cash and cash_sleeve in sleeves:
        w[sleeves.index(cash_sleeve)] = 1.0

    n = len(idx) - i0
    net, gross, tcost, hcost = (np.zeros(n) for _ in range(4))
    targets, turns, reb_days = [], [], []
    for k, i in enumerate(range(i0, len(idx))):
        d = idx[i]
        if k == 0:
            g, h = 0.0, 0.0                       # day 0: we only buy the portfolio at the close
        else:
            r = R[i]
            g = float(w @ r)
            h = costs.daily_holding_cost(w)
            w = w * (1 + r) / (1 + g)             # weights drift with prices
        t = 0.0
        if d in reb:
            history = returns.loc[d - window:d]   # strictly up to and including today
            target = validate_weights(allocator(history), returns.columns).to_numpy()
            t = costs.trading_cost(w, target)
            turns.append(0.5 * np.abs(target - w).sum())
            targets.append(target)
            reb_days.append(d)
            w = target
        net[k] = (1 + g - h) * (1 - t) - 1
        gross[k], tcost[k], hcost[k] = g, t, h

    out_idx = idx[i0:]
    return BacktestResult(
        name=name or getattr(allocator, "name", type(allocator).__name__),
        returns=pd.Series(net, out_idx, name="net"),
        gross_returns=pd.Series(gross, out_idx, name="gross"),
        weights=pd.DataFrame(targets, index=pd.DatetimeIndex(reb_days, name="date"), columns=sleeves),
        turnover=pd.Series(turns, pd.DatetimeIndex(reb_days, name="date"), name="turnover"),
        trading_costs=pd.Series(tcost, out_idx, name="trading_costs"),
        holding_costs=pd.Series(hcost, out_idx, name="holding_costs"),
        window_years=window_years,
    )
