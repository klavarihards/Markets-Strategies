"""Goal probabilities: how likely is the client to reach their goal?

The client sees one number: "about 7 in 10 simulated futures reach NT$3
million by 2036". It comes from simulating many possible futures:

1. **Monthly returns for the future.** Two ways, both reported, because
   neither is "the truth":
   - *Block bootstrap.* Cut the portfolio's actual monthly return history
     (from the walk-forward backtest, after costs) into 12-month blocks and
     glue randomly chosen blocks together into new 10- or 30-year
     histories. Crashes, recoveries and fat tails are kept exactly as they
     happened, but in new orders.
   - *Fitted distribution.* Draw monthly returns from a Student-t
     distribution with the history's mean and volatility. Fat-tailed by
     construction (5 degrees of freedom), and not limited to what happened.
     Uses antithetic variates, as in the option pricer in
     `monte-carlo/Monte_Carlo.ipynb`: every random draw is used twice, once
     as is and once mirrored, which cancels some of the simulation noise.
2. **Wealth.** Each month the contribution is added and the whole pot earns
   that month's return: ``W[t+1] = (W[t] + contribution) * (1 + r[t+1])``.
3. **Count.** The share of paths that end at or above the goal is the
   probability of reaching it.

Also reported: the median outcome and a 5th-95th percentile range; the
chance that the investments fall further than the client's loss limit at
some point (measured on returns, not on wealth, since contributions would
hide losses); how long the goal takes in a bad case (the time by which 90%
of paths have reached it); and what +NT$2,000 a month or two more years
would do to the probability. Every scenario uses the same random draws
("common random numbers"), so differences come from the change, not from
noise. Amounts are nominal: inflation is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Client:
    name: str
    description: str
    horizon_years: float
    initial: float
    monthly: float
    goal: float
    max_loss: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any], profile: str) -> "Client":
        c = cfg["clients"][profile]
        return cls(profile, c.get("description", ""), float(c["horizon_years"]), float(c["initial"]),
                   float(c["monthly"]), float(c["goal"]), float(cfg["profiles"][profile]["max_loss"]))


def bootstrap_paths(monthly: np.ndarray, n_paths: int, n_months: int, block: int = 12,
                    seed: int = 0) -> np.ndarray:
    """Circular block bootstrap of a monthly return history into ``n_paths`` x ``n_months`` returns."""
    rng = np.random.default_rng(seed)
    t = len(monthly)
    n_blocks = int(np.ceil(n_months / block))
    starts = rng.integers(0, t, size=(n_paths, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block)) % t).reshape(n_paths, -1)[:, :n_months]
    return monthly[idx]


def student_t_paths(mean: float, sd: float, n_paths: int, n_months: int, dof: float = 5.0,
                    seed: int = 0, antithetic: bool = True) -> np.ndarray:
    """Monthly returns from a Student-t with the given mean and standard deviation."""
    rng = np.random.default_rng(seed)
    scale = sd * np.sqrt((dof - 2) / dof)             # a t with `dof` has variance dof/(dof-2)
    if antithetic:
        half = rng.standard_t(dof, size=(n_paths // 2 + n_paths % 2, n_months))
        z = np.vstack([half, -half])[:n_paths]
    else:
        z = rng.standard_t(dof, size=(n_paths, n_months))
    return np.maximum(mean + scale * z, -0.99)        # a fund cannot lose more than everything


def wealth_paths(returns: np.ndarray, initial: float, monthly: float) -> np.ndarray:
    """Wealth at the start and after each month; contributions go in at the start of the month."""
    n_paths, n_months = returns.shape
    w = np.empty((n_paths, n_months + 1))
    w[:, 0] = initial
    for t in range(n_months):
        w[:, t + 1] = (w[:, t] + monthly) * (1 + returns[:, t])
    return w


def goal_report(returns: np.ndarray, client: Client, extra_years: float = 0.0) -> dict[str, float]:
    """Goal statistics for one set of simulated monthly returns (columns past the horizon allowed)."""
    h = int(round(client.horizon_years * 12))
    w = wealth_paths(returns, client.initial, client.monthly)
    final = w[:, h]
    growth = np.cumprod(1 + returns[:, :h], axis=1)
    peak = np.maximum.accumulate(np.maximum(growth, 1.0), axis=1)
    worst_fall = (growth / peak - 1).min(axis=1)
    reached = w[:, 1:] >= client.goal
    first = np.where(reached.any(axis=1), reached.argmax(axis=1) + 1, np.inf)
    bad_case_months = np.quantile(first, 0.90, method="higher")   # an actual path's time, no interpolation
    return {
        "p_reach_goal": float((final >= client.goal).mean()),
        "median_wealth": float(np.median(final)),
        "p5_wealth": float(np.quantile(final, 0.05)),
        "p95_wealth": float(np.quantile(final, 0.95)),
        "p_loss_beyond_limit": float((worst_fall < client.max_loss).mean()),
        "bad_case_years_to_goal": float(bad_case_months / 12) if np.isfinite(bad_case_months) else np.inf,
        "contributed": client.initial + client.monthly * h,
    }


def simulate_goal(monthly_hist: pd.Series, client: Client, mc: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """Bootstrap and Student-t goal reports for one portfolio, plus the two "what would help" variants."""
    sens = mc.get("sensitivity", {})
    extra = float(mc.get("extra_years", 20)) + float(sens.get("extra_years", 0))
    n_months = int(round((client.horizon_years + extra) * 12))
    n = int(mc.get("n_paths", 10000))
    x = monthly_hist.dropna().to_numpy()
    draws = {
        "bootstrap": bootstrap_paths(x, n, n_months, int(mc.get("block_months", 12)), seed),
        "student_t": student_t_paths(x.mean(), x.std(ddof=1), n, n_months, float(mc.get("student_t_dof", 5)), seed),
    }
    out: dict[str, Any] = {}
    for kind, r in draws.items():
        base = goal_report(r, client)
        out[kind] = base
        more = Client(**{**client.__dict__, "monthly": client.monthly + sens.get("extra_monthly", 0)})
        later = Client(**{**client.__dict__, "horizon_years": client.horizon_years + sens.get("extra_years", 0)})
        out[kind]["p_reach_goal_more_monthly"] = goal_report(r, more)["p_reach_goal"]
        out[kind]["p_reach_goal_later"] = goal_report(r, later)["p_reach_goal"]
    out["fan"] = {kind: np.quantile(wealth_paths(r[:, :int(client.horizon_years * 12)], client.initial,
                                                 client.monthly), [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
                  for kind, r in draws.items()}
    return out
