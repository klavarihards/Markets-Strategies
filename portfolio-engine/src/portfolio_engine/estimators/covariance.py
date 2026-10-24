"""Covariance estimates: how the sleeves move together.

Every optimiser needs a covariance matrix, and the sample covariance of past
returns is noisy. With 7 sleeves there are 28 numbers to estimate, and an
optimiser treats every one of them as exact. It hunts for combinations
that *look* low-risk in the sample, often because of estimation error, and
piles into them.

**Ledoit-Wolf shrinkage** pulls the noisy sample matrix ``S`` part of the
way towards a simple, structured target ``F``:

    shrunk = delta * F + (1 - delta) * S

The target used by default is the *constant-correlation* matrix: every
sleeve keeps its own variance, but every pair gets the same (average)
correlation. It is wrong in detail but has very little estimation error.
The shrinkage intensity ``delta`` (between 0 and 1) is not a tuning knob: it
is the value that minimises the expected error of the shrunk matrix, and
Ledoit and Wolf give a formula to estimate it from the data. Noisy data
(few observations, fat tails) gives a larger ``delta``.

Reference: Ledoit and Wolf, "Honey, I Shrunk the Sample Covariance Matrix"
(2004); this is a direct translation of their published code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_cov(returns: pd.DataFrame, ppy: int = 252) -> pd.DataFrame:
    """Annualised sample covariance (daily covariance times 252)."""
    return returns.cov() * ppy


def ledoit_wolf(returns: pd.DataFrame, target: str = "constant_correlation",
                ppy: int = 252) -> tuple[pd.DataFrame, float]:
    """Annualised shrunk covariance and the shrinkage intensity ``delta``.

    A sleeve whose return never changed in the window (a Taiwan deposit rate
    can stay unchanged for years) has zero variance: it is riskless, so it gets
    zero variance and zero covariance with everything, and the shrinkage is
    estimated on the other sleeves. (Its correlation is undefined, and the
    constant-correlation formula would divide by zero.)
    """
    var = returns.var(ddof=0)
    live = var > 1e-20 * max(float(var.max()), 1e-300)
    if not live.all():
        sub, delta = _ledoit_wolf(returns.loc[:, live], target, ppy)
        full = pd.DataFrame(0.0, index=returns.columns, columns=returns.columns)
        full.loc[sub.index, sub.columns] = sub
        return full, delta
    return _ledoit_wolf(returns, target, ppy)


def _ledoit_wolf(returns: pd.DataFrame, target: str = "constant_correlation",
                 ppy: int = 252) -> tuple[pd.DataFrame, float]:
    """Annualised shrunk covariance and the shrinkage intensity ``delta``.

    ``target`` is ``"constant_correlation"`` (Ledoit-Wolf 2004, the default)
    or ``"constant_variance"`` (a multiple of the identity matrix: same
    variance for everything and no correlation, Ledoit-Wolf 2004b).
    """
    X = returns.to_numpy(dtype=float)
    t, n = X.shape
    Xm = X - X.mean(axis=0)
    S = Xm.T @ Xm / t                          # sample covariance, 1/t as in the paper

    if target == "constant_correlation":
        var = np.diag(S)
        std = np.sqrt(var)
        r_bar = (np.sum(S / np.outer(std, std)) - n) / (n * (n - 1))
        F = r_bar * np.outer(std, std)
        np.fill_diagonal(F, var)
        # pi: sum of asymptotic variances of the sample covariance entries
        y = Xm ** 2
        pi_mat = y.T @ y / t - S ** 2
        pi_hat = pi_mat.sum()
        # rho: asymptotic covariances between the target and the sample
        term1 = (Xm ** 3).T @ Xm / t
        term2 = var[:, None] * S
        theta = term1 - term2
        np.fill_diagonal(theta, 0.0)
        rho_hat = np.trace(pi_mat) + r_bar * np.sum(np.outer(1 / std, std) * theta)
        # gamma: how far the target is from the sample
        gamma_hat = np.linalg.norm(S - F, "fro") ** 2
        delta = max(0.0, min(1.0, (pi_hat - rho_hat) / gamma_hat / t))
    elif target == "constant_variance":
        mu = np.trace(S) / n
        F = mu * np.eye(n)
        d2 = np.linalg.norm(S - F, "fro") ** 2
        b2_bar = sum(np.linalg.norm(np.outer(x, x) - S, "fro") ** 2 for x in Xm) / t ** 2
        b2 = min(b2_bar, d2)
        delta = b2 / d2 if d2 > 0 else 1.0
    else:
        raise ValueError(f"Unknown shrinkage target {target!r}")

    shrunk = delta * F + (1 - delta) * S
    cols = returns.columns
    return pd.DataFrame(shrunk * ppy, index=cols, columns=cols), float(delta)


def estimate_cov(returns: pd.DataFrame, settings: dict | None = None, ppy: int = 252) -> pd.DataFrame:
    """The covariance every method uses, chosen by the config's ``methods.covariance`` section."""
    settings = settings or {"estimator": "ledoit_wolf", "target": "constant_correlation"}
    if settings.get("estimator", "ledoit_wolf") == "sample":
        return sample_cov(returns, ppy)
    return ledoit_wolf(returns, settings.get("target", "constant_correlation"), ppy)[0]


def correlation_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    """Turn a covariance matrix into a correlation matrix (a riskless sleeve gets correlation 0)."""
    sd = np.sqrt(np.diag(cov))
    denom = np.outer(sd, sd)
    corr = np.divide(cov.to_numpy(), denom, out=np.zeros_like(denom), where=denom > 0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, cov.index, cov.columns)
