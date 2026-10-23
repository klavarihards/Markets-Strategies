"""Expected returns: the hardest input to estimate, handled two ways.

**Historical mean.** Average past return, annualised. It is what plain
mean-variance uses, and it is very noisy: with 20% volatility, even 10
years of data pins the average return down only to about +/-6% a year.
This noise is why mean-variance portfolios swing between extremes.

**Black-Litterman.** Start from returns the *market* implies instead of
from history, then adjust them with explicit views:

1. *Reverse optimisation.* If investors with risk aversion ``delta`` hold
   the market portfolio ``w_mkt`` willingly, the returns they must expect
   are ``pi = delta * Sigma @ w_mkt`` (in excess of cash). Assets that add
   a lot of risk to the market portfolio must be expected to earn more.
2. *Views.* A view says, for example, "Taiwan equity will beat US equity by
   2% a year". It is a row of ``P`` (which sleeves, with what signs), an
   entry of ``Q`` (by how much), and an entry of ``Omega`` (how unsure we
   are). By default ``Omega`` is set in proportion to the view portfolio's
   own variance (He and Litterman, 1999).
3. *Blend.* The posterior returns are a precision-weighted average of the
   equilibrium and the views: confident views move them far, vague views
   barely at all. ``tau`` scales how uncertain the equilibrium itself is.

With no views, Black-Litterman simply returns the equilibrium ``pi``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def historical_mean(returns: pd.DataFrame, ppy: int = 252) -> pd.Series:
    """Arithmetic average daily return times 252."""
    return returns.mean() * ppy


def implied_returns(cov: pd.DataFrame, market_weights: pd.Series, risk_aversion: float) -> pd.Series:
    """Equilibrium excess returns ``pi = delta * Sigma @ w_mkt`` (reverse optimisation)."""
    w = market_weights.reindex(cov.index).fillna(0.0)
    w = w / w.sum()
    return risk_aversion * cov @ w


def views_matrices(views: list[dict[str, Any]], assets: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Build ``P`` (one row per view) and ``Q`` from config-style views.

    ``{"weights": {"taiwan_equity": 1, "us_equity": -1}, "q": 0.02}`` reads
    "Taiwan equity beats US equity by 2% a year".
    """
    P = np.zeros((len(views), len(assets)))
    Q = np.zeros(len(views))
    for i, v in enumerate(views):
        for asset, wgt in v["weights"].items():
            P[i, assets.get_loc(asset)] = wgt
        Q[i] = v["q"]
    return P, Q


def black_litterman(cov: pd.DataFrame, pi: pd.Series, P: np.ndarray | None = None,
                    Q: np.ndarray | None = None, omega: np.ndarray | None = None,
                    tau: float = 0.05) -> tuple[pd.Series, pd.DataFrame]:
    """Posterior expected returns and covariance.

    Posterior mean:  pi + tau*Sigma P' (P tau*Sigma P' + Omega)^-1 (Q - P pi)
    Posterior cov:   Sigma + M, where M = tau*Sigma - tau*Sigma P' (P tau*Sigma P' + Omega)^-1 P tau*Sigma
                     is the remaining uncertainty about the mean.
    """
    S = cov.to_numpy()
    p = pi.reindex(cov.index).to_numpy()
    tS = tau * S
    if P is None or len(P) == 0:
        return pd.Series(p, cov.index), pd.DataFrame(S + tS, cov.index, cov.columns)
    if omega is None:
        omega = np.diag(np.diag(P @ tS @ P.T))      # He-Litterman default
    A = P @ tS @ P.T + omega
    post = p + tS @ P.T @ np.linalg.solve(A, Q - P @ p)
    M = tS - tS @ P.T @ np.linalg.solve(A, P @ tS)
    return pd.Series(post, cov.index), pd.DataFrame(S + M, cov.index, cov.columns)
