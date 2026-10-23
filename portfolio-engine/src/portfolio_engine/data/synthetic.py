"""Synthetic market data with the same shape as a real snapshot.

Tests must run anywhere, including machines that cannot download market
data, and must never depend on what the market happened to do. So this
module invents price histories with roughly realistic behaviour (equities
volatile and correlated with each other, bonds calmer, fat-tailed daily
moves) and returns them in exactly the format the downloader produces.

These numbers are for testing the plumbing only. Nothing computed from them
says anything about real markets and none of it may appear in a write-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ticker: (annual drift, annual volatility, group)
_SPEC = {
    "EWT": (0.08, 0.24, "eq"), "SPY": (0.09, 0.18, "eq"), "EFA": (0.06, 0.18, "eq"),
    "EEM": (0.07, 0.23, "eq"), "TLT": (0.03, 0.14, "bond"), "IEF": (0.025, 0.07, "bond"),
    "GLD": (0.05, 0.16, "gold"),
}
_CORR = {("eq", "eq"): 0.70, ("bond", "bond"): 0.85, ("eq", "bond"): -0.20,
         ("eq", "gold"): 0.05, ("bond", "gold"): 0.25, ("gold", "gold"): 1.0}


def _correlation(tickers: list[str]) -> np.ndarray:
    groups = [_SPEC[t][2] for t in tickers]
    n = len(tickers)
    c = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            key = (groups[i], groups[j]) if (groups[i], groups[j]) in _CORR else (groups[j], groups[i])
            c[i, j] = c[j, i] = _CORR[key]
    return c


def make_synthetic_frames(start: str = "2004-11-18", end: str = "2026-09-30",
                          seed: int = 0, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Return ``{"prices", "taiwan_listed", "fred"}`` frames like a real snapshot.

    Daily returns are drawn from a Student-t distribution (5 degrees of
    freedom, so crashes happen more often than a normal curve allows) with a
    fixed correlation matrix. The same seed always gives the same data.
    """
    rng = np.random.default_rng(seed)
    tickers = tickers or list(_SPEC)
    dates = pd.bdate_range(start, end, name="date")
    n, k = len(dates), len(tickers)
    mu = np.array([_SPEC[t][0] for t in tickers]) / 252
    vol = np.array([_SPEC[t][1] for t in tickers]) / np.sqrt(252)
    chol = np.linalg.cholesky(_correlation(tickers))
    dof = 5.0
    z = rng.standard_normal((n, k)) @ chol.T
    scale = np.sqrt(rng.chisquare(dof, size=(n, 1)) / dof)      # common shock: fat tails move together
    shocks = z / scale * np.sqrt((dof - 2) / dof)                 # rescale to unit variance
    rets = mu - 0.5 * vol**2 + vol * shocks
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)

    # Taiwan-listed fund: tracks EWT in TWD with small noise (holiday calendars ignored).
    fx = 30 * np.exp(np.cumsum(rng.normal(0, 0.05 / np.sqrt(252), n)))
    tw = prices[["EWT"]].mul(fx / fx[0], axis=0) * np.exp(np.cumsum(rng.normal(0, 0.001, n)))[:, None]
    tw.columns = ["0050.TW"]

    rate = np.clip(2.0 + np.cumsum(rng.normal(0, 0.03, n)), 0.0, 6.0)
    fred = pd.DataFrame({"cash_rate": rate, "usdtwd": fx}, index=dates)
    # FRED leaves holidays blank; mimic that so the loader's gap handling is exercised.
    holes = rng.random(n) < 0.02
    fred.loc[holes] = np.nan
    return {"prices": prices, "taiwan_listed": tw, "fred": fred}
