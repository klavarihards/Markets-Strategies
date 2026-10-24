"""Performance metrics, each with one definition used everywhere.

All functions take a Series of *daily* returns (0.01 = +1%) and annualise
with ``periods_per_year`` (252 trading days).

Definitions worth knowing by heart:

- **Annualised return (CAGR):** the constant yearly growth rate that would
  turn the starting value into the ending value. Geometric, so a +50% year
  followed by a -50% year is correctly a loss (1.5 x 0.5 = 0.75).
- **Volatility:** standard deviation of daily returns times sqrt(252).
  Daily risks add up like independent coin flips, so variance grows with
  time and standard deviation with its square root.
- **Sharpe ratio:** average daily return *in excess of cash*, divided by
  the standard deviation of that excess, times sqrt(252). It asks: how much
  return beyond the risk-free rate did each unit of risk earn? (Earlier
  notebooks in this repo sometimes used CAGR / volatility; that mixes a
  geometric mean with an arithmetic risk measure and ignores cash, so it is
  not used here.)
- **Sortino ratio:** like Sharpe, but only *downside* moves count as risk.
- **Maximum drawdown:** the worst fall from a previous peak, the loss a
  client who bought at the top would have seen.
- **Calmar ratio:** CAGR divided by the size of the maximum drawdown.
- **CVaR 95%:** the average return on the worst 5% of days (or months).
  VaR says where the bad tail starts; CVaR says how bad it is inside.
- **Tracking error:** volatility of the *difference* between a portfolio
  and its benchmark: how far it strays from it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PPY = 252


def cagr(r: pd.Series, ppy: int = PPY) -> float:
    """Compound annual growth rate."""
    years = len(r) / ppy
    growth = float((1 + r).prod())
    return growth ** (1 / years) - 1 if years > 0 and growth > 0 else float("nan")


def volatility(r: pd.Series, ppy: int = PPY) -> float:
    """Annualised standard deviation of returns."""
    return float(r.std(ddof=1) * np.sqrt(ppy))


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0, ppy: int = PPY) -> float:
    """Annualised Sharpe ratio of returns in excess of the risk-free rate."""
    ex = _excess(r, rf)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan")


def sortino(r: pd.Series, rf: pd.Series | float = 0.0, ppy: int = PPY) -> float:
    """Annualised Sortino ratio: excess return over downside deviation."""
    ex = _excess(r, rf)
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean())
    return float(ex.mean() / downside * np.sqrt(ppy)) if downside > 0 else float("nan")


def drawdown(r: pd.Series) -> pd.Series:
    """Fall from the running peak of the wealth curve, at every date (0 at a new high)."""
    wealth = (1 + r).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)   # the starting value counts as a peak
    return wealth / peak - 1


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough loss (a negative number)."""
    return float(drawdown(r).min())


def drawdown_stats(r: pd.Series) -> dict[str, Any]:
    """Depth and timing of the worst drawdown, and the longest time under water.

    ``recovery_days`` counts trading days from the trough of the worst
    drawdown until the old peak was regained; ``None`` if it never was.
    """
    dd = drawdown(r)
    trough = dd.idxmin()
    after = dd.loc[trough:]
    recovered = after[after >= 0]
    recovery_days = int(len(dd.loc[trough:recovered.index[0]]) - 1) if len(recovered) else None
    under = (dd < 0).astype(int).to_numpy()
    longest, run = 0, 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return {"max_drawdown": float(dd.min()), "trough_date": trough,
            "recovery_days": recovery_days, "longest_underwater_days": longest}


def calmar(r: pd.Series, ppy: int = PPY) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    mdd = max_drawdown(r)
    return cagr(r, ppy) / abs(mdd) if mdd < 0 else float("nan")


def cvar(r: pd.Series, alpha: float = 0.95) -> float:
    """Average return in the worst ``1 - alpha`` share of periods (historical CVaR, a negative number)."""
    cutoff = r.quantile(1 - alpha)
    return float(r[r <= cutoff].mean())


def monthly(r: pd.Series) -> pd.Series:
    """Compound daily returns into calendar-month returns."""
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1


def tracking_error(r: pd.Series, benchmark: pd.Series, ppy: int = PPY) -> float:
    """Annualised volatility of the return difference against a benchmark."""
    diff = (r - benchmark.reindex(r.index)).dropna()
    return volatility(diff, ppy)


def _excess(r: pd.Series, rf: pd.Series | float) -> pd.Series:
    if isinstance(rf, pd.Series):
        return (r - rf.reindex(r.index)).dropna()
    return r - rf


def summarize(r: pd.Series, rf: pd.Series | float = 0.0, benchmark: pd.Series | None = None,
              annual_turnover: float | None = None, ppy: int = PPY) -> dict[str, Any]:
    """Every standard metric for one return series, as a flat dict."""
    dds = drawdown_stats(r)
    out = {
        "start": r.index[0].date(), "end": r.index[-1].date(),
        "ann_return": cagr(r, ppy), "ann_vol": volatility(r, ppy),
        "sharpe": sharpe(r, rf, ppy), "sortino": sortino(r, rf, ppy),
        "max_drawdown": dds["max_drawdown"], "recovery_days": dds["recovery_days"],
        "longest_underwater_days": dds["longest_underwater_days"],
        "calmar": calmar(r, ppy),
        "cvar95_daily": cvar(r, 0.95), "cvar95_monthly": cvar(monthly(r), 0.95),
    }
    if annual_turnover is not None:
        out["turnover_per_year"] = annual_turnover
    if benchmark is not None:
        out["tracking_error"] = tracking_error(r, benchmark, ppy)
    return out


def sharpe_difference_ci(r_a: pd.Series, r_b: pd.Series, rf: pd.Series | float = 0.0,
                         n_boot: int = 2000, block: int = 21, ci: float = 0.90,
                         seed: int = 0, ppy: int = PPY) -> dict[str, float]:
    """Is method A's Sharpe ratio really higher than B's? A bootstrap confidence interval.

    A Sharpe ratio from 15 years of data is uncertain by roughly +/-0.25,
    so a gap of 0.05 between two methods can easily be luck. To see how
    large the gap could plausibly be, we rebuild many alternative histories
    by resampling the actual days, recompute both Sharpe ratios on each, and
    look at the spread of their difference.

    - Days are resampled in *blocks* of ``block`` consecutive days (a
      circular block bootstrap), because volatile days cluster together;
      resampling single days would make history look calmer than it was.
    - A and B are resampled on the *same* days, so what they share (both
      fell in 2008) cancels out and only their difference is tested.

    If the interval contains zero, the data cannot tell the two apart.
    """
    ex = pd.DataFrame({"a": _excess(r_a, rf), "b": _excess(r_b, rf)}).dropna().to_numpy()
    t = len(ex)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(t / block))
    last = t - (n_blocks - 1) * block                 # the final block is cut to fit t days
    starts = rng.integers(0, t, size=(n_boot, n_blocks))
    # Each resampled history is n_blocks blocks glued together, so its sum (and sum of
    # squares) is the sum of its blocks' sums. Precompute every block's sums once instead
    # of building every resampled history day by day.
    wrap = np.vstack([ex, ex[:block]])
    c1 = np.vstack([np.zeros((1, 2)), np.cumsum(wrap, axis=0)])
    c2 = np.vstack([np.zeros((1, 2)), np.cumsum(wrap ** 2, axis=0)])
    s = np.arange(t)
    full1, full2 = c1[s + block] - c1[s], c2[s + block] - c2[s]
    part1, part2 = c1[s + last] - c1[s], c2[s + last] - c2[s]
    sum1 = full1[starts[:, :-1]].sum(axis=1) + part1[starts[:, -1]]
    sum2 = full2[starts[:, :-1]].sum(axis=1) + part2[starts[:, -1]]
    mean = sum1 / t
    sd = np.sqrt(np.maximum(sum2 - t * mean ** 2, 0.0) / (t - 1))
    sr = mean / sd * np.sqrt(ppy)
    diffs = sr[:, 0] - sr[:, 1]
    point = (ex[:, 0].mean() / ex[:, 0].std(ddof=1) - ex[:, 1].mean() / ex[:, 1].std(ddof=1)) * np.sqrt(ppy)
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"sharpe_diff": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "share_boot_positive": float((diffs > 0).mean())}


EULER_GAMMA = 0.5772156649015329


def _psr_z(sr: float, sr0: float, t: int, skew: float, kurt: float) -> float:
    return (sr - sr0) * np.sqrt(t - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)


def probabilistic_sharpe_ratio(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """Probability that the true Sharpe ratio is above ``sr_benchmark`` (both per-period, e.g. daily).

    A Sharpe ratio measured on T days is itself an estimate. Its uncertainty
    grows when returns are skewed to the left or fat-tailed, which hedge-fund
    style returns often are (Bailey and Lopez de Prado, 2012). PSR turns the
    measured Sharpe into a probability, accounting for the sample length,
    skewness and kurtosis.
    """
    from scipy.stats import norm
    x = r.dropna()
    sr = x.mean() / x.std(ddof=1)
    return float(norm.cdf(_psr_z(sr, sr_benchmark, len(x), x.skew(), x.kurt() + 3)))


def expected_max_sharpe(trial_srs: np.ndarray) -> float:
    """The best Sharpe ratio you would expect from ``N`` trials *of pure noise*.

    If you try many strategies, the best one looks good by luck alone. With N
    independent trials whose Sharpe ratios vary with variance V, the expected
    maximum is about ``sqrt(V) * ((1 - g) * z(1 - 1/N) + g * z(1 - 1/(N e)))``,
    with g the Euler-Mascheroni constant and z the normal quantile function.
    """
    from scipy.stats import norm
    srs = np.asarray(trial_srs, float)
    n = len(srs)
    if n < 2:
        return 0.0
    v = srs.var(ddof=1)
    return float(np.sqrt(v) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n)
                               + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe_ratio(r: pd.Series, trial_srs: np.ndarray) -> dict[str, float]:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014).

    The probability that a strategy's true Sharpe ratio is above zero *after*
    accounting for how many configurations were tried to find it: its PSR
    measured against the best Sharpe that the same number of noise trials
    would be expected to produce, instead of against zero. ``trial_srs`` are
    the per-period Sharpe ratios of every configuration tried (the trial
    registry). A DSR above 0.95 is the usual bar.
    """
    from scipy.stats import norm
    x = r.dropna()
    sr = x.mean() / x.std(ddof=1)
    sr0 = expected_max_sharpe(trial_srs)
    return {"sharpe_daily": float(sr), "sr0_daily": sr0, "n_trials": int(len(trial_srs)),
            "dsr": float(norm.cdf(_psr_z(sr, sr0, len(x), x.skew(), x.kurt() + 3)))}
