"""The advice engine behind the API, the dashboard and the client report.

It does not invent anything new: for each profile it runs the portfolio
frozen in ``configs/final_choice.yaml`` (chosen by the pre-registered rules
on development data) walk-forward through all the data, and then answers
the client's questions with the Phase 3 tools:

- **What should I hold?** The portfolio's weights at its latest rebalance,
  minus any sleeves the client wants to avoid (the nearest portfolio within
  the profile without them), with a one-line reason for each holding.
- **How bad could a year be?** Today's weights replayed through 2008 and 2022.
- **How likely am I to reach my goal?** 10,000 simulated futures built from
  market history applied to today's weights, after the portfolio's measured
  costs; the client is shown the *lower* of the bootstrap and Student-t
  probabilities, plus what +NT$2,000 a month or two more years would do.
- **When will you contact me?** The rebalancing schedule that was backtested,
  and the loss that triggers an advisor review (75% of the client's limit).
  Nothing is ever sold automatically: a person decides.
"""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..allocation import RiskProfile, build_method_allocators
from ..allocation.optimizer import solve
from ..backtest import CostModel, metrics, run_backtest
from ..backtest.studies import load_credit_spread, make_overlay
from ..config import load_config
from ..data import load_market_data
from ..data.loader import MarketData
from ..estimators import estimate_cov
from ..risk import Client, replay, simulate_goal
from .questionnaire import Answers, assess_profile

HOLDING_TEXT = {
    "taiwan_equity": ("Taiwan stocks", "0050 (Yuanta Taiwan Top 50)", "Growth from the home market."),
    "us_equity": ("US stocks", "00646 (Yuanta S&P 500)", "Growth from the world's largest companies, in USD."),
    "intl_equity": ("Other markets' stocks", "e.g. 00645 (Fubon TOPIX)", "Spreads equity risk beyond the US and Taiwan."),
    "long_treasury": ("Long US Treasuries", "00679B (Yuanta US Treasury 20+ Year)", "Cushion when stocks fall sharply."),
    "intermediate_bond": ("Medium-term bonds", "a Taiwan-listed investment-grade bond ETF", "Steady income with less rate risk."),
    "gold": ("Gold", "00635U (Yuanta S&P GSCI Gold ER Futures)", "Protection against inflation and crises."),
    "cash": ("Cash", "TWD deposit or money-market fund", "Money you can use at any time; buffer for the goal."),
}

LIMITS = ("These figures come from simulations based on {first}-{last} market history, after estimated fees and "
          "taxes. They are not guarantees: markets can do things they have not done before. This is an "
          "educational research project using hypothetical clients, not investment advice; the funds named "
          "are representative Taiwan-listed instruments, not recommendations.")


class Advisor:
    """Builds the frozen portfolio of every profile once, then answers clients quickly."""

    def __init__(self, config: str | Path = "configs/product.yaml", data: MarketData | None = None,
                 choice: dict | None = None, credit: pd.Series | None = None, log_dir: str | Path = "results"):
        self.cfg = load_config(config) if not isinstance(config, dict) else config
        pcfg = self.cfg["product"]
        self.choice = choice or yaml.safe_load(Path(pcfg["final_choice"]).read_text())
        self.data = data or load_market_data(self.cfg, product=True, log_dir=log_dir)
        self.credit = credit if credit is not None or data is not None else load_credit_spread(self.cfg)
        self.costs = CostModel.from_config(self.cfg, list(self.data.returns.columns))
        self.allocs = build_method_allocators(self.cfg)
        self.window = self.cfg["overlays"]["window_years"]
        self.runs = {p: self._run(p) for p in self.cfg["profiles"]}

    def _regimes(self, model: str) -> dict[str, pd.Series]:
        from ..allocation.overlays import hmm_regime, rule_regime
        from ..backtest import rebalance_dates
        dates = rebalance_dates(self.data.returns.index, "monthly")
        fn = rule_regime if model == "rule" else hmm_regime
        kw = {"seed": self.cfg.get("seed", 0)} if model == "hmm" else {}
        return {model: fn(self.data.returns, self.credit, self.data.asset_class, dates, **kw)}

    def _run(self, profile: str):
        pick = self.choice["portfolios"][profile]
        overlay_name = pick["overlay"]
        regimes = {}
        if overlay_name in ("regime_rule", "regime_hmm", "all"):
            model = {"regime_rule": "rule", "regime_hmm": "hmm"}.get(overlay_name, self.cfg["overlays"]["all"]["regime_model"])
            regimes = self._regimes(model)
        overlay = make_overlay(overlay_name, profile, self.cfg, self.data, regimes)
        alloc = copy.deepcopy(self.allocs[f"{profile}:{pick['method']}"])
        return run_backtest(self.data.returns, alloc, self.costs, window_years=self.window, overlay=overlay,
                            overlay_frequency=self.cfg["overlays"]["frequency"],
                            overlay_tolerance=self.cfg["overlays"]["tolerance"], name=f"{profile}:{pick['method']}")

    def weights(self, profile: str, avoid: list[str] | None = None) -> pd.Series:
        """Today's weights for a profile; sleeves to avoid are removed and the rest re-fitted to the profile."""
        w = self.runs[profile].weights.iloc[-1]
        if not avoid:
            return w
        d = self.data.returns.index[-1]
        hist = self.data.returns.loc[d - pd.DateOffset(years=self.window):d]
        cov = estimate_cov(hist, self.cfg["methods"].get("covariance"))
        upper = np.array([0.0 if s in avoid else 1.0 for s in cov.index])
        prof = RiskProfile.from_config(self.cfg, profile)
        x, _ = solve(cov.to_numpy(), list(cov.index), self.data.asset_class, prof, "nearest",
                     target=w.reindex(cov.index).to_numpy(), upper=upper,
                     vol_cap=self.cfg["methods"].get("vol_cap", True))
        return pd.Series(x, cov.index).round(10)

    def schedule_text(self, profile: str) -> str:
        """The rebalancing routine exactly as it was backtested for this profile's portfolio."""
        text = ("We rebalance your portfolio back to its targets on the last trading day of every quarter "
                "(March, June, September, December).")
        if self.choice["portfolios"][profile]["overlay"] != "core":
            text += (" We also check it at the end of every month and adjust it when our risk signals "
                     "change by more than two percentage points.")
        return text

    def advise(self, answers: Answers, today: date | None = None) -> dict[str, Any]:
        """Everything the one-page report needs, as plain JSON-friendly values."""
        prof_info = assess_profile(answers, today)
        profile = prof_info["profile"]
        bad = [s for s in answers.avoid if s not in self.data.returns.columns]
        if bad:
            raise ValueError(f"Unknown holdings to avoid: {bad}")
        w = self.weights(profile, answers.avoid)
        pcfg = self.cfg["product"]
        d = self.data.returns.index[-1]
        hist = self.data.returns.loc[d - pd.DateOffset(years=self.window):d]
        cov = estimate_cov(hist, self.cfg["methods"].get("covariance"))
        port_hist = hist @ w.reindex(hist.columns)
        run = self.runs[profile]

        # Goal simulation: today's weights applied to the whole market history (rebalanced monthly),
        # minus the cost drag measured on this portfolio's walk-forward run. The same basis for every
        # client, with or without excluded holdings, so their probabilities can be compared. (An
        # earlier version used the walk-forward returns when nothing was excluded and this basis
        # otherwise; excluding gold then appeared to *raise* a client's odds.)
        drag = metrics.cagr(run.gross_returns) - metrics.cagr(run.returns)
        monthly = metrics.monthly(self.data.returns @ w.reindex(self.data.returns.columns).fillna(0.0)) - drag / 12
        prof_cfg = self.cfg["profiles"][profile]
        client = Client("client", "", max(answers.horizon_years(today), 1 / 12), answers.current_savings,
                        answers.monthly_contribution, answers.goal_amount, prof_cfg["max_loss"])
        mc = {**self.cfg["monte_carlo"], "n_paths": pcfg.get("n_paths", 10000)}
        sim = simulate_goal(monthly, client, mc, seed=self.cfg.get("seed", 0))
        cautious = min(("bootstrap", "student_t"), key=lambda k: sim[k]["p_reach_goal"])
        other = "student_t" if cautious == "bootstrap" else "bootstrap"
        g = sim[cautious]

        crises = {}
        for k in pcfg["report_crises"]:
            start, end = self.cfg["stress"]["historical"][k]
            if self.data.returns.index[0] <= pd.Timestamp(start) and self.data.returns.index[-1] >= pd.Timestamp(end):
                crises[k] = replay(w, self.data.returns, start, end)      # only crises fully inside the data
        holdings = [{"sleeve": s, "name": HOLDING_TEXT[s][0], "fund": HOLDING_TEXT[s][1],
                     "why": HOLDING_TEXT[s][2], "weight": float(x)} for s, x in w.items() if x >= 0.005]
        x = w.reindex(cov.index).to_numpy()
        return {
            "as_of": str(d.date()), "rebalanced_on": str(run.weights.index[-1].date()),
            "profile": prof_info,
            "portfolio": {"method": self.choice["portfolios"][profile]["method"],
                          "overlay": self.choice["portfolios"][profile]["overlay"], "holdings": holdings},
            "risk": {"expected_volatility": float(np.sqrt(x @ cov.to_numpy() @ x)),
                     "cvar95_monthly": metrics.cvar(metrics.monthly(port_hist), 0.95),
                     "max_loss_limit": prof_cfg["max_loss"],
                     "review_trigger": pcfg["review_trigger_share"] * prof_cfg["max_loss"],
                     "crises": {k: {k2: (None if v2 is None else float(v2)) for k2, v2 in v.items()}
                                for k, v in crises.items()}},
            "goal": {"amount": answers.goal_amount, "year": answers.goal_year, "simulation_used": cautious,
                     "n_paths": int(mc["n_paths"]),
                     "probability": g["p_reach_goal"], "median": g["median_wealth"], "p5": g["p5_wealth"],
                     "p95": g["p95_wealth"], "p_loss_beyond_limit": g["p_loss_beyond_limit"],
                     "probability_other_simulation": sim[other]["p_reach_goal"],
                     "contributed": g["contributed"],
                     "with_more_monthly": {"extra": mc["sensitivity"]["extra_monthly"],
                                           "probability": g["p_reach_goal_more_monthly"]},
                     "two_years_later": {"years": mc["sensitivity"]["extra_years"], "probability": g["p_reach_goal_later"]},
                     "fan": {"months": int(sim["fan"][cautious].shape[1] - 1),
                             "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
                             "values": sim["fan"][cautious].round(0).tolist()}},
            "schedule": {"rebalancing": self.schedule_text(profile)},
            "limits": LIMITS.format(first=self.data.returns.index[0].year, last=d.year),
        }
