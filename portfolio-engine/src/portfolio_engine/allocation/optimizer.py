"""One constrained solver shared by every method.

All methods end in the same kind of convex problem, solved with cvxpy:

- **Utility** (mean-variance, Black-Litterman): choose ``w`` to maximise
  ``w' mu - (lambda / 2) w' Sigma w``: expected return minus a penalty for
  variance. ``lambda`` (risk aversion) sets the exchange rate between them.
- **Min-CVaR**: choose ``w`` to minimise the average loss on the worst
  ``1 - alpha`` share of historical days. CVaR is not a smooth formula of
  the weights, but Rockafellar and Uryasev (2000) showed it equals the
  minimum over a threshold ``z`` of ``z + mean(max(loss - z, 0)) / (1 - alpha)``,
  which becomes a linear programme with one helper variable per day. At the
  optimum, ``z`` is the VaR and the objective is the CVaR.
- **Nearest** (equal weight, risk parity, HRP): these methods pick weights
  by their own rule, without the profile in mind. The solver then finds the
  portfolio *closest* to those weights (smallest sum of squared differences)
  that satisfies the profile. If the method's weights already satisfy it,
  nothing changes. How far it had to move is logged, so the reports can
  show how much of a result comes from the method and how much from the
  profile.

Both problems share the profile's hard limits, plus a cap on estimated
volatility at the top of the profile's band. If the cap cannot be met
together with the hard limits, the cap is dropped for that date and the
event is logged (``vol_cap_relaxed``). Convex problems have a single best
answer and cvxpy finds it reliably, which is why the methods are written
this way.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

from .profiles import RiskProfile, ex_ante_vol

_OK = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


def _chol(cov: np.ndarray) -> np.ndarray:
    """Cholesky factor L (cov = L L'), with a tiny ridge if the matrix is nearly singular."""
    n = len(cov)
    for ridge in (0.0, 1e-12, 1e-10, 1e-8):
        try:
            return np.linalg.cholesky(cov + ridge * np.eye(n))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Covariance matrix is not positive definite")


def solve(cov: np.ndarray, sleeves: list[str], asset_class: dict[str, str],
          profile: RiskProfile | None, kind: str, mu: np.ndarray | None = None,
          risk_aversion: float | None = None, target: np.ndarray | None = None,
          scenarios: np.ndarray | None = None, alpha: float = 0.95,
          vol_cap: bool = True, lower: np.ndarray | None = None,
          upper: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a ``"utility"``, ``"min_cvar"`` or ``"nearest"`` problem under the profile.

    ``scenarios`` (days x sleeves returns) and ``alpha`` are used by ``"min_cvar"``.
    ``lower`` / ``upper`` add per-sleeve bounds (used by the tactical tilt to
    stay within a few points of the strategic portfolio).

    Returns the weights and a small dict of diagnostics.
    """
    n = len(sleeves)
    L = _chol(cov)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0]
    if profile is not None:
        cons.append(w <= profile.max_weight)
        for g, idx in profile.members(sleeves, asset_class).items():
            lo, hi = profile.bounds[g]
            total = cp.sum(w[idx]) if idx else 0
            cons += [total >= lo, total <= hi]
    if lower is not None:
        cons.append(w >= lower)
    if upper is not None:
        cons.append(w <= upper)
    if kind == "utility":
        obj = cp.Maximize(mu @ w - 0.5 * risk_aversion * cp.sum_squares(L.T @ w))
    elif kind == "min_cvar":
        R = np.asarray(scenarios, float)
        z = cp.Variable()                          # the VaR threshold
        excess = cp.Variable(len(R), nonneg=True)  # loss beyond the threshold, per day
        cons = cons + [excess >= -R @ w - z]
        obj = cp.Minimize(z + cp.sum(excess) / ((1 - alpha) * len(R)))
    elif kind == "nearest":
        # Minimising the distance itself (not its square) gives the same answer, but keeps
        # the objective well scaled near the optimum, so the solver is accurate to ~1e-8.
        obj = cp.Minimize(cp.norm(w - target, 2))
    else:
        raise ValueError(kind)

    attempts = [True, False] if (vol_cap and profile is not None) else [False]
    for use_cap in attempts:
        extra = [cp.norm(L.T @ w, 2) <= profile.vol_band[1]] if use_cap else []
        prob = cp.Problem(obj, cons + extra)
        prob.solve(solver=cp.CLARABEL)
        if prob.status in _OK:
            x = np.clip(np.asarray(w.value).ravel(), 0.0, None)
            x = x / x.sum()
            info = {"status": prob.status, "vol_cap_relaxed": bool(vol_cap and profile is not None and not use_cap),
                    "ex_ante_vol": ex_ante_vol(x, cov)}
            if kind == "nearest":
                info["moved_by_profile"] = float(0.5 * np.abs(x - target).sum())
            return x, info
    raise RuntimeError(f"No feasible portfolio for profile {profile.name if profile else None}: {prob.status}")
