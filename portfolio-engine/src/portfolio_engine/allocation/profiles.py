"""Client risk profiles: the suitability limits every portfolio must respect.

Know-your-client comes first. A conservative client must never receive a
growth portfolio, however good the backtest of that portfolio looks. So each
profile sets hard limits that every allocation method has to work inside:

- the total weight in **equity**, in **bonds and cash**, and in **gold**
  must stay within the profile's ranges;
- no single sleeve may exceed ``max_weight`` (diversification);
- no leverage and no short positions (weights between 0 and 1, summing to 1).

Two softer numbers describe the client but are not hard limits here:

- ``vol_band``: the typical annual volatility for the profile. Its upper
  end is used as a cap on the *estimated* volatility when that is possible
  without breaking the hard limits. (In a crisis, when every equity is
  volatile, a growth client's minimum equity share can make the cap
  impossible; the hard limits then win and the event is logged.)
- ``max_loss``: the loss the client says they could live with. It is
  checked against realised drawdowns in the reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_GROUPS = {"equity": ["equity"], "bonds_cash": ["bond", "cash"], "gold": ["gold"]}


@dataclass(frozen=True)
class RiskProfile:
    name: str
    bounds: dict[str, tuple[float, float]]       # group -> (min, max) total weight
    groups: dict[str, list[str]]                 # group -> asset classes it contains
    vol_band: tuple[float, float]
    max_loss: float
    max_weight: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any], name: str) -> "RiskProfile":
        p = cfg["profiles"][name]
        groups = cfg.get("profile_groups", DEFAULT_GROUPS)
        return cls(name=name,
                   bounds={g: (float(p[g][0]), float(p[g][1])) for g in groups},
                   groups={g: list(v) for g, v in groups.items()},
                   vol_band=(float(p["vol_band"][0]), float(p["vol_band"][1])),
                   max_loss=float(p["max_loss"]), max_weight=float(p["max_weight"]))

    def members(self, sleeves: list[str], asset_class: dict[str, str]) -> dict[str, list[int]]:
        """Positions of the sleeves that belong to each group."""
        return {g: [i for i, s in enumerate(sleeves) if asset_class[s] in classes]
                for g, classes in self.groups.items()}

    def exposures(self, w: pd.Series, asset_class: dict[str, str]) -> dict[str, float]:
        """Total weight in each group."""
        idx = self.members(list(w.index), asset_class)
        return {g: float(w.iloc[i].sum()) for g, i in idx.items()}

    def violations(self, w: pd.Series, asset_class: dict[str, str], tol: float = 1e-6) -> list[str]:
        """Plain-language list of the hard limits a weight vector breaks (empty if none)."""
        out = []
        for g, x in self.exposures(w, asset_class).items():
            lo, hi = self.bounds[g]
            if x < lo - tol or x > hi + tol:
                out.append(f"{g} {x:.1%} outside {lo:.0%}-{hi:.0%}")
        if (w > self.max_weight + tol).any():
            out.append(f"sleeve above {self.max_weight:.0%}: {w[w > self.max_weight + tol].round(3).to_dict()}")
        if (w < -tol).any() or abs(w.sum() - 1) > tol:
            out.append("not long-only and fully invested")
        return out


def ex_ante_vol(w: np.ndarray, cov: np.ndarray) -> float:
    """Estimated annual volatility of a portfolio: sqrt(w' Sigma w)."""
    return float(np.sqrt(max(w @ cov @ w, 0.0)))
