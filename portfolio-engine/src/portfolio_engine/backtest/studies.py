"""Phase 4 studies: overlays, the pre-registered decision rules, robustness and diagnostics.

Everything here runs on development data only (the hold-out stays locked)
and every backtest it saves is recorded in the trial registry.

- ``run_overlay_study``: every profile x method x overlay variant.
- ``overlay_effects`` / ``apply_keep_rule``: Rule 1 from RESEARCH_LOG.md,
  an overlay is kept for a profile only if, for the median method, it
  improves max drawdown or monthly CVaR by >= 10% and costs <= 0.05 Sharpe.
- ``select_methods``: Rule 2, the recommended portfolio per profile.
- ``robustness_study``: the same methods under other reasonable settings.
- ``regime_breakdown``: performance in calm vs stressed periods.
- ``factor_exposures``: what each portfolio is really betting on.
- ``deflated_sharpe_table``: the best results, deflated for the number of trials.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from ..allocation import RiskProfile, benchmark_allocators, build_method_allocators
from ..allocation.overlays import MomentumTilt, OverlayStack, VolTarget, hmm_regime, rule_regime
from ..config import override
from ..data.loader import MarketData
from ..data.snapshot import find_snapshot, read_snapshot
from . import metrics
from .costs import CostModel
from .engine import BacktestResult, first_rebalance, rebalance_dates, run_backtest
from .experiment import run_experiment, summary_table

SIMPLICITY = ["equal_weight", "risk_parity", "hrp", "min_cvar", "mean_variance", "black_litterman"]


def load_credit_spread(cfg: dict[str, Any]) -> pd.Series | None:
    """The credit-spread series from the macro snapshot, or None if there is none."""
    path = find_snapshot(cfg.get("macro", {}).get("snapshot_dir", "data/macro/latest"))
    return read_snapshot(path)[0]["macro"]["credit_spread"] if path else None


def regime_signals(cfg: dict[str, Any], data: MarketData, credit: pd.Series | None) -> dict[str, pd.Series]:
    """Walk-forward calm/stressed flags at every month-end, from both detectors."""
    dates = rebalance_dates(data.returns.index, "monthly")
    return {"rule": rule_regime(data.returns, credit, data.asset_class, dates),
            "hmm": hmm_regime(data.returns, credit, data.asset_class, dates, seed=cfg.get("seed", 0))}


def make_overlay(variant: str, profile: str, cfg: dict[str, Any], data: MarketData,
                 regimes: dict[str, pd.Series]) -> OverlayStack | None:
    """A fresh overlay object for one variant and profile (None for the core portfolio)."""
    if variant == "core":
        return None
    oc = cfg["overlays"]
    prof = RiskProfile.from_config(cfg, profile)
    ac = data.asset_class
    bl = cfg["methods"]["black_litterman"]

    def vol():
        return VolTarget(ac, prof.vol_band[1], oc["vol_target"]["garch_window_days"], oc["vol_target"]["horizon_days"])

    def tilt():
        t = oc["tilt"]
        return MomentumTilt(prof, ac, bl["market_weights"], cfg["methods"].get("risk_aversion", 2.5), bl["tau"],
                            t["strength"], t["max_tilt"], t["lookback_months"], oc["window_years"],
                            cfg["methods"].get("covariance"), cfg["methods"].get("vol_cap", True))

    buffer = oc["regime"]["cash_buffer"]
    if variant == "vol_target":
        return OverlayStack(ac, vol=vol(), name=variant)
    if variant == "regime_rule":
        return OverlayStack(ac, regime=regimes["rule"], cash_buffer=buffer, name=variant)
    if variant == "regime_hmm":
        return OverlayStack(ac, regime=regimes["hmm"], cash_buffer=buffer, name=variant)
    if variant == "tilt":
        return OverlayStack(ac, tilt=tilt(), name=variant)
    if variant == "all":
        return OverlayStack(ac, tilt=tilt(), regime=regimes[oc["all"]["regime_model"]], cash_buffer=buffer,
                            vol=vol(), name=variant)
    raise ValueError(variant)


def run_overlay_study(cfg: dict[str, Any], data: MarketData, regimes: dict[str, pd.Series],
                      variants: list[str] | None = None, profiles: list[str] | None = None,
                      methods: list[str] | None = None) -> dict[tuple[float, str], BacktestResult]:
    """Backtest every profile x method x variant (3-year window). Keys: ``(window, "profile:method|variant")``."""
    oc = cfg["overlays"]
    w = oc["window_years"]
    variants = variants or oc["variants"]
    costs = CostModel.from_config(cfg, list(data.returns.columns))
    start = first_rebalance(data.returns.index, w, cfg["backtest"].get("rebalance", "quarterly"))
    out = {}
    for name, alloc in build_method_allocators(cfg).items():
        prof, method = name.split(":")
        if (profiles and prof not in profiles) or (methods and method not in methods):
            continue
        for v in variants:
            fresh = copy.deepcopy(alloc)
            ov = make_overlay(v, prof, cfg, data, regimes)
            res = run_backtest(data.returns, fresh, costs, window_years=w,
                               frequency=cfg["backtest"].get("rebalance", "quarterly"),
                               start_in_cash=cfg["backtest"].get("start_in_cash", True), start=start,
                               name=f"{name}|{v}", overlay=ov, overlay_frequency=oc["frequency"],
                               overlay_tolerance=oc["tolerance"])
            res.meta["log"] = pd.DataFrame(fresh.log).set_index("date")
            if ov is not None:
                res.meta["overlay_log"] = pd.DataFrame(ov.log)
            out[(w, f"{name}|{v}")] = res
    for bname, b in benchmark_allocators(cfg).items():
        out[(w, bname)] = run_backtest(data.returns, b, costs, window_years=w, start=start, name=bname)
    return out


def _variant_frame(table: pd.DataFrame) -> pd.DataFrame:
    t = table.drop(index="benchmark", level="profile", errors="ignore").reset_index()
    t[["method", "variant"]] = t["method"].str.split("|", expand=True)
    return t.set_index(["profile", "method", "variant"])


def overlay_effects(summary: pd.DataFrame) -> pd.DataFrame:
    """Change against the core portfolio for every profile x method x overlay variant."""
    t = _variant_frame(summary.xs(summary.index.get_level_values(0)[0], level="window_years"))
    rows = []
    for (prof, method, variant), r in t.iterrows():
        if variant == "core":
            continue
        c = t.loc[(prof, method, "core")]
        rows.append({"profile": prof, "method": method, "variant": variant,
                     "d_sharpe": r.sharpe - c.sharpe, "d_ann_return": r.ann_return - c.ann_return,
                     "maxdd_improvement": 1 - r.max_drawdown / c.max_drawdown,
                     "cvar_improvement": 1 - r.cvar95_monthly / c.cvar95_monthly,
                     "d_turnover": r.turnover_per_year - c.turnover_per_year})
    return pd.DataFrame(rows).set_index(["profile", "method", "variant"])


def apply_keep_rule(effects: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Rule 1: keep an overlay for a profile if, for the median method, risk improves >= 10%
    (max drawdown or monthly CVaR) and the Sharpe ratio falls by no more than 0.05."""
    rule = cfg["decision_rules"]["keep_overlay"]
    med = effects.groupby(level=["profile", "variant"]).median()
    med["risk_improvement"] = med[["maxdd_improvement", "cvar_improvement"]].max(axis=1)
    med["keep"] = (med["risk_improvement"] >= rule["min_risk_improvement"]) & (med["d_sharpe"] >= -rule["max_sharpe_loss"])
    return med


def select_methods(summary: pd.DataFrame, compliance: pd.DataFrame, kept: pd.DataFrame,
                   cfg: dict[str, Any]) -> pd.DataFrame:
    """Rule 2: per profile, among portfolios within 0.05 Sharpe of the best (core or kept overlays):
    those within the client's loss limit first, then the lowest monthly CVaR; if none stayed within
    the limit, the smallest max drawdown; ties to the simpler method."""
    tol = cfg["decision_rules"]["select_method"]["sharpe_tolerance"]
    w = summary.index.get_level_values(0)[0]
    s = _variant_frame(summary.xs(w, level="window_years"))
    c = _variant_frame(compliance.xs(w, level="window_years"))
    s = s.join(c[["max_loss_breached"]])
    rows = []
    for prof in cfg["profiles"]:
        allowed = {"core"} | set(kept.xs(prof, level="profile").query("keep").index)
        cand = s.xs(prof, level="profile").reset_index()
        cand = cand[cand["variant"].isin(allowed)]
        cand = cand[cand["sharpe"] >= cand["sharpe"].max() - tol]
        within = cand[~cand["max_loss_breached"]]
        pool, key, why = ((within, "cvar95_monthly", "within loss limit, lowest monthly CVaR") if len(within)
                          else (cand, "max_drawdown", "none within loss limit, smallest max drawdown"))
        pool = pool.assign(simplicity=pool["method"].map(SIMPLICITY.index),
                           overlay_count=(pool["variant"] != "core").astype(int))
        best = pool.sort_values([key, "simplicity", "overlay_count"], ascending=[False, True, True]).iloc[0]
        rows.append({"profile": prof, "method": best["method"], "variant": best["variant"], "reason": why,
                     "sharpe": best["sharpe"], "max_drawdown": best["max_drawdown"],
                     "cvar95_monthly": best["cvar95_monthly"], "candidates": len(cand)})
    return pd.DataFrame(rows).set_index("profile")


def robustness_study(cfg: dict[str, Any], data: MarketData) -> tuple[pd.DataFrame, dict]:
    """The core methods (3-year window) under other reasonable settings; returns a long table and the runs."""
    rb = cfg["robustness"]
    prof = rb["profile"]
    base = override(cfg, **{"backtest.window_years": [cfg["overlays"]["window_years"]],
                            "profiles": {prof: cfg["profiles"][prof]}})
    settings = {f"rebalance={f}": {"backtest.rebalance": f} for f in rb["rebalance"]}
    settings |= {f"risk_aversion={a}": {"methods.risk_aversion": a} for a in rb["risk_aversion"]}
    settings |= {f"max_weight={m}": {f"profiles.{prof}.max_weight": m} for m in rb["max_weight"]}
    rows, runs = [], {}
    for label, change in settings.items():
        c = override(base, **change)
        c["name"] = f"phase4_robustness[{label}]"
        res = run_experiment(c, data)
        runs[label] = (c, res)
        summ = summary_table(res, data, c)
        for (w, p, m), r in summ.iterrows():
            rows.append({"setting": label, "profile": p, "method": m, "sharpe": r.sharpe,
                         "ann_return": r.ann_return, "max_drawdown": r.max_drawdown,
                         "turnover_per_year": r.turnover_per_year})
    cal = override(cfg, **{"backtest.window_years": [cfg["overlays"]["window_years"]],
                           "profiles": {"conservative": {**cfg["profiles"]["conservative"],
                                        "vol_band": [cfg["profiles"]["conservative"]["vol_band"][0],
                                                     rb["calibrated_conservative_vol_cap"]]}}})
    cal["name"] = "phase4_robustness[conservative_vol_cap=6%]"
    res = run_experiment(cal, data)
    runs["conservative_vol_cap=6%"] = (cal, res)
    for (w, p, m), r in summary_table(res, data, cal).iterrows():
        rows.append({"setting": "conservative_vol_cap=6%", "profile": p, "method": m, "sharpe": r.sharpe,
                     "ann_return": r.ann_return, "max_drawdown": r.max_drawdown,
                     "turnover_per_year": r.turnover_per_year})
    return pd.DataFrame(rows), runs


def daily_regime(regime: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """A month-end regime flag applies from the next trading day until the next month-end."""
    return regime.astype(float).reindex(index.union(regime.index)).ffill().shift(1).reindex(index).fillna(0).astype(bool)


def regime_breakdown(results: dict[tuple[float, str], BacktestResult], regime: pd.Series,
                     rf: pd.Series) -> pd.DataFrame:
    """Annualised return, volatility and Sharpe of each run in calm and in stressed periods."""
    rows = []
    for (w, name), res in results.items():
        flag = daily_regime(regime, res.returns.index)
        for state, mask in (("calm", ~flag), ("stressed", flag)):
            r = res.returns[mask]
            if len(r) < 20:
                continue
            rows.append({"run": name, "regime": state, "days": len(r), "ann_return": r.mean() * 252,
                         "ann_vol": metrics.volatility(r), "sharpe": metrics.sharpe(r, rf)})
    return pd.DataFrame(rows).set_index(["run", "regime"])


def factor_returns(data: MarketData, credit: pd.Series | None) -> pd.DataFrame:
    """Monthly factor returns in TWD: equity, rates (duration), credit, USD/TWD and gold, all over cash."""
    m = lambda s: metrics.monthly(s)  # noqa: E731
    cash = m(data.returns["cash"])
    f = pd.DataFrame({"equity": m(data.returns["us_equity"]) - cash,
                      "rates": m(data.returns["intermediate_bond"]) - cash,
                      "usd_twd": m(data.fx),
                      "gold": m(data.returns["gold"]) - cash})
    if credit is not None:
        spread = credit.resample("ME").last()
        spread.index = spread.index.to_period("M")
        f["credit"] = -spread.diff().reindex(f.index)          # positive = spreads tightened
    return f.dropna()


def factor_exposures(results: dict[tuple[float, str], BacktestResult], data: MarketData,
                     credit: pd.Series | None) -> pd.DataFrame:
    """OLS betas (with Newey-West t-statistics) of each run's monthly excess return on the factors."""
    import statsmodels.api as sm

    f = factor_returns(data, credit)
    cash = metrics.monthly(data.returns["cash"])
    rows = []
    for (w, name), res in results.items():
        y = (metrics.monthly(res.returns) - cash).reindex(f.index).dropna()
        X = sm.add_constant(f.loc[y.index])
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        row = {"run": name, "r2": fit.rsquared, "alpha_annual": fit.params["const"] * 12}
        for k in f.columns:
            row[f"beta_{k}"] = fit.params[k]
            row[f"t_{k}"] = fit.tvalues[k]
        rows.append(row)
    return pd.DataFrame(rows).set_index("run")


def deflated_sharpe_table(results: dict[tuple[float, str], BacktestResult], names: list[str],
                          registry_path: str = "experiments/registry.csv", snapshot_id: str | None = None) -> pd.DataFrame:
    """DSR of the named runs, deflated by every distinct development trial in the registry."""
    reg = pd.read_csv(registry_path)
    reg = reg[reg["tag"] == "dev"]
    if snapshot_id:
        reg = reg[reg["snapshot_id"] == snapshot_id]
    distinct = reg.assign(sr=reg["sharpe_daily"].round(10)).drop_duplicates(["run", "sr", "first", "last"])
    trials = distinct["sharpe_daily"].to_numpy()
    rows = []
    for key, res in results.items():
        if key[1] in names:
            d = metrics.deflated_sharpe_ratio(res.returns, trials)
            rows.append({"run": key[1], "sharpe_annual": d["sharpe_daily"] * np.sqrt(252),
                         "expected_max_sharpe_annual": d["sr0_daily"] * np.sqrt(252),
                         "n_trials": d["n_trials"], "psr_vs_zero": metrics.probabilistic_sharpe_ratio(res.returns),
                         "dsr": d["dsr"]})
    return pd.DataFrame(rows).set_index("run")
