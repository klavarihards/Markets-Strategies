"""Phase 4 overlays: adjustments on top of the strategic portfolio.

The strategic ("core") portfolio is set quarterly by one of the methods.
Overlays react faster (monthly here) to protect the client or lean into a
signal, and are judged by one question: does the client end up better off
after costs? Three layers, as a private bank would use them:

1. **Tactical tilt** (Black-Litterman momentum views). Time-series momentum:
   asset classes that rose over the past 12 months (beyond cash) tend to
   keep rising for a while, and vice versa (Moskowitz, Ooi and Pedersen,
   2012). Each risky sleeve gets a view "expected return = equilibrium
   return ± 2% a year", with the sign from its 12-month excess return. The
   Black-Litterman posterior is optimised under the profile, but never more
   than 5 percentage points away from the core in any sleeve.
2. **Regime filter.** A two-state view of the world, calm or stressed. In the
   stressed state the tilt is switched off (back to the core) and 10% of the
   portfolio moves from equities to cash. Two detectors are compared:
   - *Rule* (the simple version, like the moving-average regime project in
     this repo): stressed when at least two of three alarms ring: the
     equity sleeves are below their 200-day average; their 1-month
     volatility is above its 3-year 80th percentile; the credit spread
     (Baa corporate minus 10-year Treasury yield) is above its 3-year 80th
     percentile.
   - *Hidden Markov model:* a two-state Gaussian HMM fitted on weekly
     equity returns, volatility and credit spreads, re-fitted each month on
     data up to that date; "stressed" is the state with the higher variance,
     and the regime is stressed when its probability on the last week is
     above 50%.
3. **Volatility targeting** (GARCH, reused from the QQQ project in this
   repo). A GARCH(1,1) with Student-t shocks is fitted on the last two years
   of the portfolio's daily returns and forecasts its volatility over the
   next month. If the forecast is above the top of the profile's band, the
   equity weights are scaled down by (band / forecast) and the money goes to
   bonds and cash in their current proportions. No leverage: the scale never
   goes above 1, so in calm markets nothing happens.

De-risking may take equity below the profile's minimum or cash above the
single-sleeve cap for a while; that is a deliberate exception, because it
only ever lowers the client's risk.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..estimators import black_litterman, estimate_cov, implied_returns
from .optimizer import solve
from .profiles import RiskProfile


def equity_index_returns(returns: pd.DataFrame, asset_class: dict[str, str]) -> pd.Series:
    """Daily return of an equal-weighted mix of the equity sleeves: the 'market' for regime signals."""
    eq = [s for s in returns.columns if asset_class.get(s) == "equity"]
    return returns[eq].mean(axis=1)


def move_from_equity(w: pd.Series, amount: float, asset_class: dict[str, str],
                     to: str = "defensive") -> pd.Series:
    """Take ``amount`` of weight out of equities (proportionally) and put it into cash or bonds+cash."""
    w = w.copy()
    eq = [s for s in w.index if asset_class[s] == "equity"]
    total_eq = w[eq].sum()
    amount = min(amount, total_eq)
    if amount <= 0:
        return w
    w[eq] *= (total_eq - amount) / total_eq
    if to == "cash":
        dest = [s for s in w.index if asset_class[s] == "cash"]
        w[dest[0]] += amount
    else:
        dest = [s for s in w.index if asset_class[s] in ("bond", "cash")]
        base = w[dest]
        share = base / base.sum() if base.sum() > 0 else pd.Series(1.0 / len(dest), dest)
        w[dest] += amount * share
    return w


# ---------------------------------------------------------------------------------------------
# Regime detection (computed once per date, walk-forward, shared by every portfolio)
# ---------------------------------------------------------------------------------------------

def rule_regime(returns: pd.DataFrame, credit_spread: pd.Series | None, asset_class: dict[str, str],
                dates: pd.DatetimeIndex) -> pd.Series:
    """Stressed (True) when at least two of three alarms ring, using only data up to each date."""
    r = equity_index_returns(returns, asset_class)
    level = (1 + r).cumprod()
    vol21 = r.rolling(21).std() * np.sqrt(252)
    spread = credit_spread.shift(1).reindex(returns.index, method="ffill") if credit_spread is not None else None
    out = {}
    for d in dates:
        i = returns.index.get_indexer([d], method="pad")[0]
        if i < 756:
            out[d] = False
            continue
        lv, vv = level.iloc[:i + 1], vol21.iloc[:i + 1]
        alarms = [lv.iloc[-1] < lv.iloc[-200:].mean(),
                  vv.iloc[-1] > vv.iloc[-756:].quantile(0.8)]
        if spread is not None:
            sp = spread.iloc[:i + 1].dropna()
            alarms.append(len(sp) > 0 and sp.iloc[-1] > sp.iloc[-756:].quantile(0.8))
        out[d] = sum(alarms) >= 2
    return pd.Series(out, name="stressed")


def hmm_regime(returns: pd.DataFrame, credit_spread: pd.Series | None, asset_class: dict[str, str],
               dates: pd.DatetimeIndex, seed: int = 0, min_weeks: int = 156) -> pd.Series:
    """Two-state Gaussian HMM on weekly features, re-fitted at each date on data up to that date."""
    from hmmlearn.hmm import GaussianHMM

    r = equity_index_returns(returns, asset_class)
    weekly = (1 + r).resample("W-FRI").prod() - 1
    feats = pd.DataFrame({"ret": weekly,
                          "vol": np.log(r.rolling(21).std().resample("W-FRI").last() * np.sqrt(252))})
    if credit_spread is not None:
        feats["spread"] = credit_spread.shift(1).resample("W-FRI").last()
    feats = feats.dropna()
    out = {}
    for d in dates:
        x = feats.loc[:d]
        if len(x) < min_weeks:
            out[d] = False
            continue
        z = ((x - x.mean()) / x.std()).to_numpy()
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(z)
            prob = model.predict_proba(z)[-1]
        stressed_state = int(np.argmax([model.covars_[k][0, 0] for k in range(2)]))
        out[d] = bool(prob[stressed_state] > 0.5)
    return pd.Series(out, name="stressed")


# ---------------------------------------------------------------------------------------------
# The three layers
# ---------------------------------------------------------------------------------------------

def garch_vol_forecast(port_returns: pd.Series, horizon: int = 21) -> float:
    """Annualised volatility forecast for the next ``horizon`` days from a GARCH(1,1) with t shocks.

    Same set-up as the QQQ volatility-targeting notebook: returns in percent,
    Student-t errors, fitted on a rolling window. Falls back to an EWMA
    (lambda 0.94, RiskMetrics) estimate if the fit fails.
    """
    from arch import arch_model

    x = 100 * port_returns.dropna()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(x, vol="Garch", p=1, q=1, dist="t", rescale=False).fit(disp="off")
            var = res.forecast(horizon=horizon, reindex=False).variance.iloc[-1].mean()
        vol = np.sqrt(var * 252) / 100
        if np.isfinite(vol) and vol > 0:
            return float(vol)
    except Exception:
        pass
    ewma = (x / 100) ** 2
    return float(np.sqrt(ewma.ewm(alpha=0.06).mean().iloc[-1] * 252))


class VolTarget:
    """Scale equities down when the GARCH forecast is above the profile's volatility band."""

    def __init__(self, asset_class: dict[str, str], cap: float, window: int = 504, horizon: int = 21):
        self.asset_class, self.cap, self.window, self.horizon = asset_class, cap, window, horizon
        self.log: list[dict[str, Any]] = []

    def __call__(self, history: pd.DataFrame, w: pd.Series, d) -> pd.Series:
        port = history.iloc[-self.window:] @ w.reindex(history.columns).fillna(0.0)
        forecast = garch_vol_forecast(port, self.horizon)
        scale = min(1.0, self.cap / forecast)
        eq = [s for s in w.index if self.asset_class[s] == "equity"]
        out = move_from_equity(w, (1 - scale) * w[eq].sum(), self.asset_class) if scale < 1 else w
        self.log.append({"date": d, "garch_vol": forecast, "equity_scale": scale})
        return out


class MomentumTilt:
    """Black-Litterman momentum views, at most ``max_tilt`` away from the core in any sleeve."""

    def __init__(self, profile: RiskProfile, asset_class: dict[str, str], market_weights: dict[str, float],
                 risk_aversion: float = 2.5, tau: float = 0.05, strength: float = 0.02, max_tilt: float = 0.05,
                 lookback_months: int = 12, window_years: float = 3, cov_settings: dict | None = None,
                 vol_cap: bool = True):
        self.profile, self.asset_class = profile, asset_class
        self.mkt = pd.Series(market_weights, dtype=float)
        self.risk_aversion, self.tau, self.strength, self.max_tilt = risk_aversion, tau, strength, max_tilt
        self.lookback, self.window, self.cov_settings, self.vol_cap = lookback_months, window_years, cov_settings, vol_cap
        self.log: list[dict[str, Any]] = []

    def signals(self, history: pd.DataFrame, d) -> pd.Series:
        """12-month return of each sleeve minus cash's: +1 if positive, -1 if negative."""
        past = history.loc[d - pd.DateOffset(months=self.lookback):d]
        growth = (1 + past).prod() - 1
        cash = [s for s in history.columns if self.asset_class[s] == "cash"]
        excess = growth - (growth[cash[0]] if cash else 0.0)
        return np.sign(excess.drop(cash))

    def __call__(self, history: pd.DataFrame, core: pd.Series, d) -> pd.Series:
        hist = history.loc[d - pd.DateOffset(months=int(self.window * 12)):d]
        cov = estimate_cov(hist, self.cov_settings)
        pi = implied_returns(cov, self.mkt, self.risk_aversion)
        sig = self.signals(history, d)
        P = np.zeros((len(sig), len(cov)))
        for k, s in enumerate(sig.index):
            P[k, cov.index.get_loc(s)] = 1.0
        Q = pi[sig.index].to_numpy() + self.strength * sig.to_numpy()
        mu, post_cov = black_litterman(cov, pi, P, Q, tau=self.tau)
        c = core.reindex(cov.index).to_numpy()
        lo = np.clip(c - self.max_tilt, 0.0, None)
        hi = np.minimum(c + self.max_tilt, max(self.profile.max_weight, c.max()))
        w, info = solve(post_cov.to_numpy(), list(cov.index), self.asset_class, self.profile, "utility",
                        mu=mu.to_numpy(), risk_aversion=self.risk_aversion, vol_cap=self.vol_cap,
                        lower=lo, upper=hi)
        self.log.append({"date": d, **{f"signal_{s}": int(v) for s, v in sig.items()}, **info})
        return pd.Series(w, cov.index)


class OverlayStack:
    """Tilt (unless stressed), then the regime's cash buffer (if stressed), then volatility targeting."""

    def __init__(self, asset_class: dict[str, str], tilt: MomentumTilt | None = None,
                 regime: pd.Series | None = None, cash_buffer: float = 0.10, vol: VolTarget | None = None,
                 name: str = "overlay"):
        self.asset_class, self.tilt, self.regime, self.cash_buffer, self.vol = asset_class, tilt, regime, cash_buffer, vol
        self.name = name
        self.log: list[dict[str, Any]] = []

    def stressed(self, d) -> bool:
        if self.regime is None:
            return False
        past = self.regime.loc[:d]
        return bool(past.iloc[-1]) if len(past) else False

    def __call__(self, history: pd.DataFrame, core: pd.Series, d) -> pd.Series:
        stressed = self.stressed(d)
        w = core
        if self.tilt is not None and not stressed:
            w = self.tilt(history, core, d)
        if stressed:
            w = move_from_equity(w, self.cash_buffer, self.asset_class, to="cash")
        if self.vol is not None:
            w = self.vol(history, w, d)
        equity = float(w[[s for s in w.index if self.asset_class[s] == "equity"]].sum())
        self.log.append({"date": d, "stressed": stressed, "equity": equity})
        return w
