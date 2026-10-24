"""Dated research views for Black-Litterman: entered with a confidence, used walk-forward, scored afterwards.

A view is a statement such as "Taiwan equity will beat US equity by 3% a
year over the next 12 months, and I am 60% confident". It is written down in
``configs/views.yaml`` with the date it was made, and then:

- **used only from that date on**, for its horizon, so a backtest can never
  benefit from a view before it existed;
- **turned into Black-Litterman inputs**: the sleeves it is about (a row of
  P), the expected excess return (Q), and its uncertainty (Omega) from the
  stated confidence with Idzorek's method in the closed form given by
  Walters (2014): ``omega = tau * (1 - c) / c * p' Sigma p``. A confidence of
  100% means "certain" (omega near 0, the view is imposed); a confidence
  near 0% means the view is ignored;
- **scored when its horizon is over**: did the sleeves move in the stated
  direction, and how far was the realised excess return from the forecast?

That turns research notes into a track record: views entered in a
disciplined way, used mechanically, and judged afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class View:
    id: str
    entered: pd.Timestamp
    statement: str
    weights: dict[str, float]          # +1 / -1 per sleeve: "long minus short"
    q: float                           # expected excess return of that combination, per year
    confidence: float                  # 0 < c <= 1
    horizon_months: int
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ends(self) -> pd.Timestamp:
        return self.entered + pd.DateOffset(months=self.horizon_months)

    def active(self, d: pd.Timestamp) -> bool:
        return self.entered <= d < self.ends


def load_views(path: str | Path) -> list[View]:
    """Read the views file (an empty list is fine)."""
    p = Path(path)
    items = (yaml.safe_load(p.read_text()) or {}).get("views", []) if p.exists() else []
    out = []
    for v in items or []:
        c = float(v["confidence"])
        if not 0 < c <= 1:
            raise ValueError(f"View {v['id']}: confidence must be in (0, 1], got {c}")
        out.append(View(id=str(v["id"]), entered=pd.Timestamp(v["entered"]), statement=v["statement"],
                        weights={k: float(x) for k, x in v["weights"].items()}, q=float(v["q"]),
                        confidence=c, horizon_months=int(v["horizon_months"]), source=v.get("source", "")))
    return out


def idzorek_omega(P: np.ndarray, cov: np.ndarray, confidences: np.ndarray, tau: float) -> np.ndarray:
    """Diagonal Omega from view confidences (Idzorek's method, Walters' closed form)."""
    c = np.clip(np.asarray(confidences, float), 1e-6, 1.0)
    var = np.array([p @ cov @ p for p in P])
    return np.diag(tau * (1 - c) / c * var + 1e-12)


def active_views(views: list[View], d: pd.Timestamp, assets: pd.Index) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """P, Q, confidences and ids of the views active on date ``d``."""
    act = [v for v in views if v.active(d)]
    P = np.zeros((len(act), len(assets)))
    for i, v in enumerate(act):
        for s, w in v.weights.items():
            P[i, assets.get_loc(s)] = w
    return P, np.array([v.q for v in act]), np.array([v.confidence for v in act]), [v.id for v in act]


def score_views(views: list[View], returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Realised result of each view: direction right or wrong, and the error against the forecast.

    The realised value is the annualised excess return of the view's long-minus-short
    combination (or of a single sleeve over cash) from the entry date to the end of its
    horizon, or to the last available date if the view is still open.
    """
    rows = []
    last = returns.index[-1]
    for v in views:
        seg = returns.loc[v.entered + pd.Timedelta(days=1):min(v.ends, last)]
        if len(seg) == 0:
            rows.append({"id": v.id, "entered": v.entered.date(), "status": "not started", "statement": v.statement})
            continue
        combo = sum(w * seg[s] for s, w in v.weights.items())
        if abs(sum(v.weights.values())) > 1e-9:            # a single-sleeve view is measured over cash
            combo = combo - rf.reindex(seg.index)
        years = len(seg) / 252
        realised = float((1 + combo).prod() ** (1 / years) - 1) if years > 0 else np.nan
        rows.append({"id": v.id, "entered": v.entered.date(), "ends": v.ends.date(),
                     "status": "closed" if v.ends <= last else "open", "statement": v.statement,
                     "forecast": v.q, "confidence": v.confidence, "realised": realised,
                     "direction_right": bool(np.sign(realised) == np.sign(v.q)), "error": realised - v.q})
    return pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame()


def main() -> None:
    """Score every view in configs/views.yaml against the latest data (TWD, Taiwan cash).

    Run from the project folder: ``python -m portfolio_engine.estimators.views``.
    Uses all data including the hold-out, so it needs the logged hold-out evaluation.
    """
    from ..config import load_config
    from ..data import load_market_data

    views = load_views("configs/views.yaml")
    if not views:
        print("configs/views.yaml has no views yet.")
        return
    data = load_market_data(load_config("configs/taiwan.yaml"), product=True, log_dir="experiments")
    print(score_views(views, data.returns, data.rf).to_string())


if __name__ == "__main__":
    main()
