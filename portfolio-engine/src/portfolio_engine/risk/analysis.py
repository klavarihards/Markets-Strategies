"""Phase 3 in one call: every risk table for every portfolio of a walk-forward run.

"Today's portfolio" for each method is the target it chose at the last
rebalance of the data it was given (with the hold-out locked, the last
quarter-end before the hold-out). The risk inputs (covariance, daily
returns for CVaR) are the same estimation window the method used for that
decision. The goal simulations use each method's *walk-forward* monthly
returns after costs, so the method's real trading behaviour and costs are
part of the simulated futures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..backtest import metrics
from ..backtest.engine import BacktestResult
from ..backtest.experiment import split_name
from ..data.loader import MarketData
from ..estimators import estimate_cov
from .decomposition import decompose
from .goals import Client, simulate_goal
from .stress import historical_stress, hypothetical_stress


def current_portfolios(results: dict[tuple[float, str], BacktestResult], window: float) -> dict[str, pd.Series]:
    """Last target weights of every run with this estimation window, keyed by run name."""
    return {name: res.weights.iloc[-1] for (w, name), res in results.items() if w == window}


def run_risk_analysis(cfg: dict[str, Any], data: MarketData,
                      results: dict[tuple[float, str], BacktestResult]) -> dict[str, Any]:
    """Stress tests, risk decomposition, CVaR and goal probabilities for every portfolio."""
    rcfg = cfg.get("risk", {})
    window = rcfg.get("window_years", 3)
    alpha = rcfg.get("cvar_alpha", 0.95)
    ppy = cfg["backtest"].get("periods_per_year", 252)
    ports = current_portfolios(results, window)
    as_of = next(iter(ports.values())).name
    hist = data.returns.loc[as_of - pd.DateOffset(months=int(round(window * 12))):as_of]
    cov = estimate_cov(hist, cfg.get("methods", {}).get("covariance"), ppy)

    weights = pd.DataFrame(ports).T
    weights.index = pd.MultiIndex.from_tuples([split_name(n) for n in weights.index], names=["profile", "method"])

    # Risk decomposition, by sleeve and by asset class, plus the portfolio's own CVaR and volatility.
    decomp, by_class, risk_summary = {}, {}, []
    for name, w in ports.items():
        decomp[name] = decompose(w, cov, hist, alpha)
        by_class[name] = decompose(w, cov, hist, alpha, data.asset_class)
        port = hist @ w.reindex(hist.columns).fillna(0.0)
        x = w.reindex(cov.index).fillna(0.0).to_numpy()
        risk_summary.append({"profile": split_name(name)[0], "method": split_name(name)[1],
                             "ex_ante_vol": float(np.sqrt(x @ cov.to_numpy() @ x)),
                             "cvar95_daily": metrics.cvar(port, alpha),
                             "cvar95_monthly": metrics.cvar(metrics.monthly(port), alpha),
                             "equity_share_of_volatility": float(by_class[name].loc["equity", "share_of_volatility"]),
                             "equity_share_of_cvar": float(by_class[name].loc["equity", "share_of_cvar"]),
                             "equity_weight": float(by_class[name].loc["equity", "weight"])})
    risk_summary = pd.DataFrame(risk_summary).set_index(["profile", "method"])

    # Stress tests. The historical replay uses every day of data available (it is a
    # "what if today's portfolio lived through 2008" exercise, not a forecast).
    hist_stress = historical_stress(ports, data.returns, cfg["stress"]["historical"])
    hyp = hypothetical_stress(ports, {n: cov for n in ports}, cfg["stress"], data.asset_class)
    for t in (hist_stress, hyp):
        t["max_loss_limit"] = [cfg["profiles"][split_name(n)[0]]["max_loss"] if split_name(n)[0] in cfg["profiles"]
                               else np.nan for n in t.index.get_level_values(0)]
    hist_stress["beyond_limit"] = hist_stress["max_drawdown"] < hist_stress["max_loss_limit"]
    hyp["beyond_limit"] = hyp["loss"] < hyp["max_loss_limit"]

    # What the walk-forward portfolios actually did in each crisis they lived through.
    realised = []
    for name in ports:
        r = results[(window, name)].returns
        for scen, (start, end) in cfg["stress"]["historical"].items():
            seg = r.loc[start:end]
            covered = len(seg) > 0 and seg.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=7)
            realised.append({"portfolio": name, "scenario": scen,
                             "realised_return": float((1 + seg).prod() - 1) if len(seg) else np.nan,
                             "realised_max_drawdown": metrics.max_drawdown(seg) if len(seg) else np.nan,
                             "fully_covered": covered})
    realised = pd.DataFrame(realised).set_index(["portfolio", "scenario"])

    # Goal probabilities: each client against the methods of their profile, plus the benchmarks.
    goal_rows, fans = [], {}
    mc = cfg["monte_carlo"]
    for profile in cfg.get("clients", {}):
        client = Client.from_config(cfg, profile)
        for name in ports:
            prof, method = split_name(name)
            if prof not in (profile, "benchmark"):
                continue
            monthly = metrics.monthly(results[(window, name)].returns)
            sim = simulate_goal(monthly, client, mc, seed=cfg.get("seed", 0))
            fans[(profile, method)] = sim["fan"]
            for kind in ("bootstrap", "student_t"):
                goal_rows.append({"client": profile, "portfolio": method, "simulation": kind, **sim[kind]})
    goals = pd.DataFrame(goal_rows).set_index(["client", "portfolio", "simulation"])

    return {"as_of": as_of, "weights": weights, "risk_summary": risk_summary, "decomposition": decomp,
            "decomposition_by_class": by_class, "historical_stress": hist_stress, "realised_in_crises": realised,
            "hypothetical_stress": hyp, "goals": goals, "fans": fans}


def tables_for_saving(analysis: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """The flat tables of an analysis, ready for ``save_run(..., tables=...)``."""
    by_class = pd.concat(analysis["decomposition_by_class"], names=["portfolio", "asset_class"])
    return {"current_weights": analysis["weights"], "risk_summary": analysis["risk_summary"],
            "risk_by_asset_class": by_class, "historical_stress": analysis["historical_stress"],
            "realised_in_crises": analysis["realised_in_crises"],
            "hypothetical_stress": analysis["hypothetical_stress"], "goals": analysis["goals"]}
