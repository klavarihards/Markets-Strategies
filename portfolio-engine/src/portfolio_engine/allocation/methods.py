"""The five allocation methods, from the simplest to the most elaborate.

Each method is a callable object: give it the returns history at a
rebalance date and it returns the target weights for the client's profile.
Every method uses the same shrunk covariance estimate, so differences in
results come from the method, not from different inputs.

1. **Equal weight**: the same amount in every sleeve. No estimates at all.
2. **Mean-variance** (Markowitz, 1952): trade expected return against
   variance, using historical average returns and the Ledoit-Wolf
   covariance. Optimal *if* the inputs were right; fragile because they
   are not.
3. **Black-Litterman**: the same trade-off, but with expected returns
   implied by a neutral market portfolio (plus any explicit views) instead
   of historical averages.
4. **Risk parity** (equal risk contribution): ignore expected returns and
   size each sleeve so it adds the same amount of risk.
5. **Hierarchical risk parity** (Lopez de Prado, 2016): group similar
   sleeves by correlation into a tree, then split money down the tree,
   giving less to riskier groups. No matrix inversion, so it is stable when
   assets are highly correlated.

Methods 2 and 3 have the profile limits inside their optimisation. Methods
1, 4 and 5 compute their own weights first and are then moved to the
nearest portfolio that satisfies the profile (see ``optimizer.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from ..estimators import (black_litterman, correlation_from_cov, estimate_cov, historical_mean,
                          implied_returns, views_matrices)
from .optimizer import solve
from .profiles import RiskProfile


# ---------------------------------------------------------------------------
# Pure weight rules (no profile), usable and testable on their own
# ---------------------------------------------------------------------------

def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Share of portfolio variance coming from each asset: w_i (Sigma w)_i / w' Sigma w.

    The shares add up to 1. A portfolio that is 50% bonds by weight can get
    90% of its risk from equities; this is the number that shows it.
    """
    m = cov @ w
    return w * m / (w @ m)


def risk_parity_weights(cov: np.ndarray, budgets: np.ndarray | None = None,
                        tol: float = 1e-14, max_iter: int = 200) -> np.ndarray:
    """Long-only weights whose risk contributions match ``budgets`` (equal by default).

    Solved with Spinu's (2013) convex reformulation: minimise
    ``0.5 y' Sigma y - sum(b_i log y_i)`` over y > 0. At the minimum,
    ``y_i (Sigma y)_i = b_i`` for every i, which is exactly "each asset's
    risk contribution equals its budget". Rescaling y to sum to 1 gives the
    weights. Newton's method converges in a handful of steps.
    """
    n = len(cov)
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, float) / np.sum(budgets)
    y = 1.0 / np.sqrt(np.diag(cov))
    y = y / np.sqrt(y @ cov @ y)
    for _ in range(max_iter):
        grad = cov @ y - b / y
        hess = cov + np.diag(b / y ** 2)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while np.any(y - t * step <= 0):          # stay strictly positive
            t *= 0.5
        y = y - t * step
        if np.max(np.abs(step)) < tol * np.max(np.abs(y)):
            break
    return y / y.sum()


def hrp_weights(cov: pd.DataFrame, linkage: str = "single") -> pd.Series:
    """Hierarchical risk parity weights (Lopez de Prado, 2016).

    1. *Tree clustering.* Turn correlations into distances,
       ``d = sqrt((1 - rho) / 2)`` (0 for perfectly correlated, 1 for
       perfectly opposite), and join the closest assets into clusters, then
       clusters into bigger clusters, building a tree.
    2. *Quasi-diagonalisation.* Reorder the assets so that similar ones sit
       next to each other (the order of the tree's leaves).
    3. *Recursive bisection.* Split the ordered list in half. Give each half
       money in inverse proportion to its variance (a riskier half gets
       less), then repeat inside each half until single assets remain.
    """
    corr = correlation_from_cov(cov)
    dist = np.sqrt(np.clip((1.0 - corr.to_numpy()) / 2.0, 0.0, 1.0))
    link = sch.linkage(ssd.squareform(dist, checks=False), linkage)
    order = [cov.index[i] for i in sch.to_tree(link, rd=False).pre_order()]

    def cluster_var(items: list[str]) -> float:
        c = cov.loc[items, items].to_numpy()
        ivp = 1.0 / np.diag(c)
        ivp /= ivp.sum()
        return float(ivp @ c @ ivp)

    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[a:b] for c in clusters for a, b in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
    return w.reindex(cov.index)


# ---------------------------------------------------------------------------
# Allocators: method + profile, called by the backtest at each rebalance
# ---------------------------------------------------------------------------

class ProfileMethod:
    """Shared plumbing: estimate the covariance, run the method, log diagnostics."""

    method = "base"

    def __init__(self, profile: RiskProfile | None, asset_class: dict[str, str],
                 cov_settings: dict | None = None, vol_cap: bool = True, ppy: int = 252):
        self.profile = profile
        self.asset_class = asset_class
        self.cov_settings = cov_settings
        self.vol_cap = vol_cap
        self.ppy = ppy
        self.name = f"{profile.name}:{self.method}" if profile else self.method
        self.log: list[dict[str, Any]] = []

    def __call__(self, history: pd.DataFrame) -> pd.Series:
        cov = estimate_cov(history, self.cov_settings, self.ppy)
        w, info = self.weights(history, cov)
        self.log.append({"date": history.index[-1], **info})
        return pd.Series(w, index=history.columns)

    def weights(self, history: pd.DataFrame, cov: pd.DataFrame) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def _nearest(self, raw: np.ndarray, cov: pd.DataFrame) -> tuple[np.ndarray, dict]:
        return solve(cov.to_numpy(), list(cov.index), self.asset_class, self.profile, "nearest",
                     target=raw, vol_cap=self.vol_cap)

    def _risky(self, cov: pd.DataFrame, exclude: list[str]) -> list[str]:
        return [s for s in cov.index if self.asset_class[s] not in exclude]


class EqualWeightMethod(ProfileMethod):
    """1/N across all sleeves, then the nearest portfolio the profile allows."""

    method = "equal_weight"

    def weights(self, history, cov):
        raw = np.full(len(cov), 1.0 / len(cov))
        return self._nearest(raw, cov)


class MeanVariance(ProfileMethod):
    """Maximise ``mu' w - (lambda/2) w' Sigma w`` with historical means and shrunk covariance."""

    method = "mean_variance"

    def __init__(self, *args, risk_aversion: float = 2.5, **kw):
        super().__init__(*args, **kw)
        self.risk_aversion = risk_aversion

    def weights(self, history, cov):
        mu = historical_mean(history, self.ppy).reindex(cov.index).to_numpy()
        return solve(cov.to_numpy(), list(cov.index), self.asset_class, self.profile, "utility",
                     mu=mu, risk_aversion=self.risk_aversion, vol_cap=self.vol_cap)


class BlackLitterman(ProfileMethod):
    """Mean-variance on Black-Litterman posterior returns and covariance."""

    method = "black_litterman"

    def __init__(self, *args, market_weights: dict[str, float], risk_aversion: float = 2.5,
                 tau: float = 0.05, views: list[dict] | None = None, **kw):
        super().__init__(*args, **kw)
        self.market_weights = pd.Series(market_weights, dtype=float)
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.views = views or []

    def weights(self, history, cov):
        pi = implied_returns(cov, self.market_weights, self.risk_aversion)
        P, Q = views_matrices(self.views, cov.index) if self.views else (None, None)
        mu, post_cov = black_litterman(cov, pi, P, Q, tau=self.tau)
        return solve(post_cov.to_numpy(), list(cov.index), self.asset_class, self.profile, "utility",
                     mu=mu.to_numpy(), risk_aversion=self.risk_aversion, vol_cap=self.vol_cap)


class RiskParity(ProfileMethod):
    """Equal risk contribution across the risky sleeves, then the nearest profile portfolio."""

    method = "risk_parity"

    def __init__(self, *args, exclude_asset_classes: list[str] | None = None, **kw):
        super().__init__(*args, **kw)
        self.exclude = exclude_asset_classes or []

    def weights(self, history, cov):
        risky = self._risky(cov, self.exclude)
        raw = pd.Series(0.0, index=cov.index)
        raw[risky] = risk_parity_weights(cov.loc[risky, risky].to_numpy())
        return self._nearest(raw.to_numpy(), cov)


class HierarchicalRiskParity(ProfileMethod):
    """HRP across the risky sleeves, then the nearest profile portfolio."""

    method = "hrp"

    def __init__(self, *args, linkage: str = "single", exclude_asset_classes: list[str] | None = None, **kw):
        super().__init__(*args, **kw)
        self.linkage = linkage
        self.exclude = exclude_asset_classes or []

    def weights(self, history, cov):
        risky = self._risky(cov, self.exclude)
        raw = pd.Series(0.0, index=cov.index)
        raw[risky] = hrp_weights(cov.loc[risky, risky], self.linkage)
        return self._nearest(raw.to_numpy(), cov)


METHODS = {c.method: c for c in (EqualWeightMethod, MeanVariance, BlackLitterman,
                                  RiskParity, HierarchicalRiskParity)}


def build_method_allocators(cfg: dict[str, Any]) -> dict[str, ProfileMethod]:
    """Every (profile, method) pair named in the config, keyed ``"profile:method"``."""
    asset_class = {k: v["asset_class"] for k, v in cfg["sleeves"].items()}
    m = cfg["methods"]
    common = {"cov_settings": m.get("covariance"), "vol_cap": m.get("vol_cap", True),
              "ppy": cfg["backtest"].get("periods_per_year", 252)}
    out: dict[str, ProfileMethod] = {}
    for pname in cfg["profiles"]:
        profile = RiskProfile.from_config(cfg, pname)
        for method, cls in METHODS.items():
            if method not in m:
                continue
            kw = dict(m[method] or {})
            if method in ("mean_variance", "black_litterman"):
                kw["risk_aversion"] = m.get("risk_aversion", 2.5)
            alloc = cls(profile, asset_class, **common, **kw)
            out[alloc.name] = alloc
    return out
