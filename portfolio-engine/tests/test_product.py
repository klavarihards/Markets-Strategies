"""Phase 5: questionnaire, advisor, report, API and the product's hold-out guard (synthetic data)."""

import json
from datetime import date

import pandas as pd
import pytest
from conftest import ROOT

from portfolio_engine.config import load_config
from portfolio_engine.data import load_market_data, make_synthetic_frames, write_snapshot
from portfolio_engine.product import Advisor, Answers, assess_profile

TODAY = date(2026, 10, 23)
CHOICE = {"portfolios": {"conservative": {"method": "min_cvar", "overlay": "core"},
                         "balanced": {"method": "risk_parity", "overlay": "vol_target"},
                         "growth": {"method": "hrp", "overlay": "regime_rule"}}}


def _answers(**kw):
    base = dict(goal_amount=3_000_000, goal_year=2036, current_savings=600_000, monthly_contribution=15_000,
                income_stability="stable", reaction_to_20pct_fall="hold")
    return Answers(**{**base, **kw})


@pytest.fixture(scope="module")
def advisor():
    cfg = load_config(ROOT / "configs" / "product.yaml")
    cfg["product"]["n_paths"] = 500
    frames = make_synthetic_frames(end="2016-12-31", seed=8)
    data = load_market_data(cfg, frames=frames, product=True)
    return Advisor(cfg, data=data, choice=CHOICE)


def test_profile_is_the_more_cautious_of_capacity_and_tolerance():
    assert assess_profile(_answers(), TODAY)["profile"] == "balanced"
    assert assess_profile(_answers(reaction_to_20pct_fall="sell"), TODAY)["profile"] == "conservative"
    assert assess_profile(_answers(goal_year=2056, income_stability="very_stable",
                                   reaction_to_20pct_fall="buy_more"), TODAY)["profile"] == "growth"
    # Loves risk, but needs the money soon: conservative.
    assert assess_profile(_answers(goal_year=2028, reaction_to_20pct_fall="buy_more"), TODAY)["profile"] == "conservative"
    # Long horizon, but would sell in a panic: not growth.
    assert assess_profile(_answers(goal_year=2056, income_stability="very_stable",
                                   reaction_to_20pct_fall="sell"), TODAY)["profile"] == "conservative"


def test_advice_is_complete_and_suitable(advisor):
    adv = advisor.advise(_answers(), TODAY)
    assert adv["profile"]["profile"] == "balanced"
    w = pd.Series({h["sleeve"]: h["weight"] for h in adv["portfolio"]["holdings"]})
    assert w.sum() == pytest.approx(1, abs=0.02)        # holdings under 0.5% are not listed
    assert 0 <= adv["goal"]["probability"] <= 1
    assert adv["goal"]["probability"] <= adv["goal"]["probability_other_simulation"]       # the cautious one is shown
    assert adv["goal"]["with_more_monthly"]["probability"] >= adv["goal"]["probability"]
    assert adv["risk"]["review_trigger"] == pytest.approx(0.75 * -0.20)
    # Synthetic data ends in 2016: only crises fully inside the data are reported.
    assert set(adv["risk"]["crises"]) == {"global_financial_crisis"}
    json.dumps(adv)                                                                        # JSON-ready


def test_avoided_holdings_are_removed_within_the_profile(advisor):
    from portfolio_engine.allocation import RiskProfile
    adv = advisor.advise(_answers(avoid=["gold", "taiwan_equity"]), TODAY)
    shown = {h["sleeve"] for h in adv["portfolio"]["holdings"]}
    assert "gold" not in shown and "taiwan_equity" not in shown
    w = advisor.weights("balanced", ["gold", "taiwan_equity"])          # the full weight vector
    assert w["gold"] == 0 and w["taiwan_equity"] == 0
    prof = RiskProfile.from_config(advisor.cfg, "balanced")
    assert prof.violations(w, advisor.data.asset_class) == []
    with pytest.raises(ValueError):
        advisor.advise(_answers(avoid=["bitcoin"]), TODAY)


def test_pdf_report(advisor):
    from portfolio_engine.product.report import render_html, render_pdf
    adv = advisor.advise(_answers(), TODAY)
    html = render_html(adv)
    assert f"About {round(10 * adv['goal']['probability'])} in 10" in html
    assert "Nothing is sold automatically" in html
    pdf = render_pdf(adv)
    assert pdf[:4] == b"%PDF" and len(pdf) > 10_000


def test_api(advisor):
    from fastapi.testclient import TestClient

    from portfolio_engine.product.api import create_app
    client = TestClient(create_app(advisor))
    assert client.get("/health").json()["status"] == "ok"
    assert set(client.get("/profiles").json()) == {"conservative", "balanced", "growth"}
    body = {"goal_amount": 3_000_000, "goal_year": 2036, "current_savings": 600_000, "monthly_contribution": 15_000}
    r = client.post("/advice", json=body)
    assert r.status_code == 200 and r.json()["profile"]["profile"] in ("conservative", "balanced", "growth")
    r = client.post("/report", json=body)
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content[:4] == b"%PDF"
    assert client.post("/advice", json={**body, "avoid": ["bitcoin"]}).status_code == 422
    assert client.post("/advice", json={**body, "income_stability": "rich"}).status_code == 422


def test_product_needs_the_holdout_to_have_been_evaluated(tmp_path):
    cfg = load_config(ROOT / "configs" / "product.yaml")
    snap = write_snapshot(make_synthetic_frames(end="2012-12-31", seed=1), tmp_path / "snap")
    cfg["product"]["holdout_log"] = str(tmp_path / "holdout_log.jsonl")
    with pytest.raises(PermissionError):
        load_market_data(cfg, snapshot_dir=snap, product=True)
    dev = load_market_data(cfg, snapshot_dir=snap)
    (tmp_path / "holdout_log.jsonl").write_text(json.dumps({"snapshot_id": dev.snapshot_id}) + "\n")
    full = load_market_data(cfg, snapshot_dir=snap, product=True)
    assert full.includes_holdout and full.returns.index[-1] > dev.returns.index[-1]


def test_implementation_tracking_on_synthetic_data():
    from portfolio_engine.product.implementation import tracking_report
    rep = tracking_report(make_synthetic_frames(end="2014-12-31", seed=2))
    # The synthetic 0050 is built as EWT in TWD plus small noise, so it must track closely.
    assert rep["correlation"] > 0.95 and rep["tracking_error"] < 0.05 and rep["months"] > 100
