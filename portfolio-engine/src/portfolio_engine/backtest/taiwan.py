"""Taiwan reality check: Taiwan cash rates, and the frozen portfolios held with Taiwan-listed funds.

Two questions a Taiwan wealth manager would ask of this research:

1. **What if cash earns what a Taiwan client really earns?** Cash becomes a
   TWD deposit at the five major banks' 1-month rate (Central Bank data),
   which is also the risk-free rate for Sharpe ratios. The frozen portfolio
   choices are *not* re-selected (the hold-out has been opened); the same
   portfolios, methods and benchmarks are re-run, and their development and
   hold-out results compared with the original runs.
2. **How far do the real funds drift from the research proxies?** From the
   first date all six Taiwan-listed funds trade (June 2017), each frozen
   portfolio's actual trade history is replayed twice: with the proxies (as
   researched) and with 0050, 00646, 00645, 00679B, 00697B and 00635U (as a
   client would hold it). Trades decided at the New York close are executed
   on the next Taipei trading day. The fund version pays spreads, commission
   and the 0.1% sell tax, but no extra fee: fund prices are already net of
   the funds' own fees.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import override
from ..data import load_market_data, read_snapshot
from ..data.loader import twd_daily_rate
from ..product.implementation import fund_returns, replay_trades, sleeve_drift
from . import metrics
from .costs import CostModel
from .experiment import run_experiment, summary_table
from .studies import load_credit_spread, regime_signals, run_overlay_study

BENCHMARKS = ("equal_weight", "sixty_forty", "taiwan_equity_only")


def frozen_runs(cfg: dict[str, Any], data, choice: dict[str, Any], regimes) -> dict:
    """Walk-forward runs of the frozen portfolio per profile, plus the benchmarks (3-year window)."""
    out = {}
    for prof, c in choice["portfolios"].items():
        out.update(run_overlay_study(cfg, data, regimes, variants=[c["overlay"]], profiles=[prof],
                                     methods=[c["method"]]))
    return out


def fund_cost_model(cfg: dict[str, Any], sleeves: list[str]) -> CostModel:
    """Trading costs of the Taiwan-listed funds: spread, commission, sell tax; no extra fee or FX spread."""
    costs = {s: {**v, "fund_fee": 0.0, "proxy_fee": 0.0, "withholding": 0.0, "fx_spread": 0.0}
             for s, v in cfg["costs"].items()}
    return CostModel.build(costs, sleeves, cfg["backtest"].get("periods_per_year", 252))


def _stats(r: pd.Series, rf: pd.Series) -> dict[str, float]:
    return {"ann_return": metrics.cagr(r), "ann_vol": metrics.volatility(r), "sharpe": metrics.sharpe(r, rf),
            "max_drawdown": metrics.max_drawdown(r)}


def taiwan_reality_check(cfg: dict[str, Any], choice: dict[str, Any], log_dir: str = "experiments") -> dict[str, Any]:
    """Run both checks on all data (allowed: the hold-out evaluation is logged)."""
    data = load_market_data(cfg, product=True, log_dir=log_dir)
    credit = load_credit_spread(cfg)
    regimes = regime_signals(cfg, data, credit)
    w = cfg["overlays"]["window_years"]
    hold = (data.holdout_start, None)
    dev = (None, data.holdout_start - pd.Timedelta(days=1))

    # 1. Taiwan cash: the frozen portfolios, every core method and the benchmarks, re-run.
    frozen = frozen_runs(cfg, data, choice, regimes)
    core = run_experiment(override(cfg, **{"backtest.window_years": [w]}), data)
    runs = {**core, **frozen}
    cash_tables = {"dev": summary_table(runs, data, cfg, period=dev),
                   "holdout": summary_table(runs, data, cfg, period=hold)}

    # 2. Taiwan-listed funds.
    tcfg = cfg["taiwan"]
    tw, _ = read_snapshot(tcfg["snapshot_dir"])
    funds, quality = fund_returns(tw["funds"], tcfg["funds"])
    start = funds.dropna().index[0]                        # first day all six funds trade
    funds = funds.loc[start:].fillna(0.0)
    funds["cash"] = twd_daily_rate(tw[tcfg["cash_rate"]["table"]], funds.index, tcfg["cash_rate"]["column"],
                                   tcfg["cash_rate"].get("lag_months", 1))
    funds = funds[list(data.returns.columns)]
    rf_tw = funds["cash"]
    drift = sleeve_drift(data.returns, funds.drop(columns="cash"), str(start.date()))

    research_costs = CostModel.from_config(cfg, list(data.returns.columns))
    tw_costs = fund_cost_model(cfg, list(data.returns.columns))
    replay_start = pd.Timestamp(start) + pd.offsets.QuarterEnd(0)
    rows, series = [], {}
    for (win, name), res in frozen.items():
        proxy = replay_trades(data.returns, res.weights, research_costs, replay_start)
        fund = replay_trades(funds, res.weights, tw_costs, replay_start, next_day=True)
        # Split the gap: the research's holding-cost assumptions (extra fund fee, withholding, FX
        # spread, which the funds' prices already contain), the funds' own returns, next-day trading.
        proxy_tw_costs = replay_trades(data.returns, res.weights, tw_costs, replay_start)
        fund_same_day = replay_trades(funds, res.weights, tw_costs, replay_start)
        split = {"gap_cost_assumptions": metrics.cagr(proxy_tw_costs) - metrics.cagr(proxy),
                 "gap_fund_returns": metrics.cagr(fund_same_day) - metrics.cagr(proxy_tw_costs),
                 "gap_next_day": metrics.cagr(fund) - metrics.cagr(fund_same_day)}
        pm, fm = metrics.monthly(proxy), metrics.monthly(fund)
        both = pd.DataFrame({"proxy": pm, "fund": fm}).dropna()
        series[name] = {"proxy": proxy, "fund": fund}
        p, f = _stats(proxy, data.rf), _stats(fund, rf_tw)
        rows.append({"portfolio": name, "start": str(fund.index[0].date()), "end": str(fund.index[-1].date()),
                     **{f"proxy_{k}": v for k, v in p.items()}, **{f"fund_{k}": v for k, v in f.items()},
                     "return_gap": f["ann_return"] - p["ann_return"], **split,
                     "monthly_correlation": both.corr().iloc[0, 1],
                     "tracking_error": (both["fund"] - both["proxy"]).std(ddof=1) * np.sqrt(12)})
    implementation = pd.DataFrame(rows).set_index("portfolio")
    return {"data": data, "runs": runs, "cash_tables": cash_tables, "drift": drift, "quality": quality,
            "implementation": implementation, "series": series, "funds_start": str(start.date())}


def save_reality_check(out: dict[str, Any], cfg: dict[str, Any]):
    """Save the runs and tables (provenance, registry) like every other experiment."""
    from .experiment import save_run

    tables = {"cash_dev": out["cash_tables"]["dev"], "cash_holdout": out["cash_tables"]["holdout"],
              "fund_drift": out["drift"], "fund_implementation": out["implementation"]}
    return save_run(out["runs"], out["cash_tables"]["dev"], cfg, out["data"], tables=tables)
