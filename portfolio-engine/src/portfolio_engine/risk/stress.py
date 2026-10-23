"""Stress tests: how bad could it get?

**Historical replay.** Take today's portfolio and run it through a real
crisis: what if you had held exactly these weights from the start of the
2008 crash to its bottom? The sleeves' actual daily TWD returns are used,
and the portfolio is left alone (no rebalancing) during the episode, the
way most clients actually behave in a crash. Reported: total return over
the episode, the worst fall from a peak inside it, and how many months it
took afterwards to get back to the starting value.

**Hypothetical shocks.** Events that have not happened (yet), applied
instantly to today's weights:

- a price shock to some sleeves (e.g. Taiwan equities -30%);
- a currency move: if TWD weakens 10%, every USD-priced asset is worth 10%
  more in TWD, which *cushions* a Taiwan crash for a diversified client;
- a rise in US interest rates, turned into bond losses with duration: a bond
  with duration D loses about D x (rate rise), so a 2-point rise costs a
  17-year-duration Treasury fund about 34%;
- a correlation shock: all risky sleeves suddenly move together (correlation
  0.8), which is what diversification fails against. This changes risk, not
  prices, so it is reported as the size of a 1-in-20 bad year.

Each loss is compared with the client's maximum acceptable loss.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def replay(w: pd.Series, returns: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    """Hold weights ``w`` without rebalancing through ``[start, end]`` of historical returns."""
    w = w.reindex(returns.columns).fillna(0.0)
    episode = returns.loc[start:end]
    if len(episode) == 0:
        return {"total_return": np.nan, "max_drawdown": np.nan, "months_to_recover": np.nan}
    growth = (1 + episode).cumprod()                 # each sleeve's value, starting from 1
    value = growth @ w                               # buy-and-hold portfolio value
    peak = np.maximum(value.cummax(), 1.0)
    out = {"total_return": float(value.iloc[-1] - 1), "max_drawdown": float((value / peak - 1).min())}
    # After the episode, keep holding and see when the starting value (1.0) comes back.
    after = (1 + returns.loc[episode.index[-1]:].iloc[1:]).cumprod() * growth.iloc[-1]
    path = pd.concat([value, after @ w])
    bottom = value.idxmin()
    back = path.loc[bottom:][path.loc[bottom:] >= 1.0]
    out["months_to_recover"] = (round((back.index[0] - bottom).days / 30.44, 1) if len(back) else None)
    return out


def historical_stress(weights: dict[str, pd.Series], returns: pd.DataFrame,
                      scenarios: dict[str, list[str]]) -> pd.DataFrame:
    """Replay every portfolio through every historical scenario."""
    rows = []
    for name, w in weights.items():
        for scen, (start, end) in scenarios.items():
            rows.append({"portfolio": name, "scenario": scen, **replay(w, returns, start, end)})
    return pd.DataFrame(rows).set_index(["portfolio", "scenario"])


def shock_returns(spec: dict[str, Any], sleeves: list[str], fx_exposed: list[str],
                  durations: dict[str, float]) -> pd.Series:
    """Instant TWD return of each sleeve under a price / FX / rates shock."""
    local = pd.Series(0.0, index=sleeves)
    for s, x in spec.get("shocks", {}).items():
        local[s] += x
    for s, d in durations.items():
        local[s] += -d * spec.get("rates", 0.0)
    fx = pd.Series(0.0, index=sleeves)
    fx[[s for s in fx_exposed if s in sleeves]] = spec.get("fx", 0.0)
    return (1 + local) * (1 + fx) - 1


def correlation_stress_cov(cov: pd.DataFrame, rho: float, risky: list[str]) -> pd.DataFrame:
    """Same volatilities, but every pair of risky sleeves gets correlation ``rho``."""
    sd = np.sqrt(np.diag(cov))
    corr = cov.to_numpy() / np.outer(sd, sd)
    idx = [cov.index.get_loc(s) for s in risky]
    for i in idx:
        for j in idx:
            if i != j:
                corr[i, j] = rho
    return pd.DataFrame(np.outer(sd, sd) * corr, cov.index, cov.columns)


def bad_year_loss(w: pd.Series, cov: pd.DataFrame, confidence: float = 0.95) -> float:
    """A 1-in-20 bad year under a normal approximation: -1.645 x annual volatility (zero mean)."""
    x = w.reindex(cov.index).fillna(0.0).to_numpy()
    return float(-norm.ppf(confidence) * np.sqrt(x @ cov.to_numpy() @ x))


def hypothetical_stress(weights: dict[str, pd.Series], covs: dict[str, pd.DataFrame],
                        cfg_stress: dict[str, Any], asset_class: dict[str, str]) -> pd.DataFrame:
    """Apply every hypothetical scenario to every portfolio."""
    rows = []
    risky = [s for s, c in asset_class.items() if c != "cash"]
    for name, w in weights.items():
        cov = covs[name]
        sleeves = list(cov.index)
        for scen, spec in cfg_stress["hypothetical"].items():
            if "correlation" in spec:
                stressed = correlation_stress_cov(cov, spec["correlation"], risky)
                rows.append({"portfolio": name, "scenario": scen, "kind": "risk",
                             "loss": bad_year_loss(w, stressed), "normal_case": bad_year_loss(w, cov)})
            else:
                r = shock_returns(spec, sleeves, cfg_stress.get("fx_exposed", []), cfg_stress.get("durations", {}))
                rows.append({"portfolio": name, "scenario": scen, "kind": "price",
                             "loss": float(w.reindex(sleeves).fillna(0.0) @ r), "normal_case": np.nan})
    return pd.DataFrame(rows).set_index(["portfolio", "scenario"])
