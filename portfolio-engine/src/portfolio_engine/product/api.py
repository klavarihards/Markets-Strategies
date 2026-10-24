"""FastAPI service: submit a client's answers, receive the plan or the one-page PDF.

Endpoints
- ``GET  /health``     liveness check, with the data date
- ``GET  /profiles``   the three risk profiles and the frozen portfolio for each
- ``POST /advice``     questionnaire answers in, advice JSON out
- ``POST /report``     questionnaire answers in, one-page PDF out

The engine is separate from the interface: the same ``Advisor`` serves this
API, the Streamlit dashboard and the notebooks. Start it with
``uvicorn portfolio_engine.product.api:app`` from the ``portfolio-engine``
folder (``PORTFOLIO_ENGINE_CONFIG`` selects another config).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .advisor import Advisor
from .questionnaire import Answers


class AnswersIn(BaseModel):
    goal_amount: float = Field(gt=0, description="Goal in TWD", examples=[3_000_000])
    goal_year: int = Field(ge=2000, le=2100, examples=[2036])
    current_savings: float = Field(ge=0, examples=[600_000])
    monthly_contribution: float = Field(ge=0, examples=[15_000])
    income_stability: Literal["unstable", "stable", "very_stable"] = "stable"
    reaction_to_20pct_fall: Literal["sell", "hold", "buy_more"] = "hold"
    avoid: list[str] = Field(default_factory=list, description="Holdings to exclude, e.g. ['gold']")
    client_name: str = "Hypothetical client"


def create_app(advisor: Advisor | None = None) -> FastAPI:
    app = FastAPI(title="Goal-based portfolio engine",
                  description="Educational research project with hypothetical clients. Not investment advice.",
                  version="0.1.0")

    @lru_cache(maxsize=1)
    def get_advisor() -> Advisor:
        return advisor or Advisor(os.environ.get("PORTFOLIO_ENGINE_CONFIG", "configs/product.yaml"))

    def advise(a: AnswersIn) -> dict:
        try:
            return get_advisor().advise(Answers(**a.model_dump(exclude={"client_name"})))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=f"No portfolio satisfies these constraints: {e}")

    @app.get("/health")
    def health() -> dict:
        adv = get_advisor()
        return {"status": "ok", "data_until": str(adv.data.returns.index[-1].date()), "snapshot": adv.data.snapshot_id}

    @app.get("/profiles")
    def profiles() -> dict:
        adv = get_advisor()
        return {p: {**adv.cfg["profiles"][p], **adv.choice["portfolios"][p]} for p in adv.cfg["profiles"]}

    @app.post("/advice")
    def advice(a: AnswersIn) -> dict:
        return advise(a)

    @app.post("/report", response_class=Response)
    def report(a: AnswersIn) -> Response:
        from .report import render_pdf
        pdf = render_pdf(advise(a), a.client_name)
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": 'attachment; filename="plan.pdf"'})

    return app


app = create_app()
