"""Frictions: what it costs to hold and to trade each sleeve.

Two kinds of cost, charged differently:

**Holding costs** accrue every day, whether you trade or not:

- *Fund fee (expense ratio).* The research proxies' adjusted prices are
  already net of the proxy's own fee, so only the *extra* fee of the fund
  the client would really hold is charged: ``fund_fee - proxy_fee``. A
  cheaper real fund gives a small credit. Charging the full fund fee would
  count the fee twice.
- *Dividend withholding.* Tax withheld at source on dividends (e.g. 30% on
  US dividends for a non-US investor), entered as an annual drag.

**Trading costs** are paid on the value traded at each rebalance:

- *Half the bid-ask spread*: you buy at the ask and sell at the bid, so each
  trade loses about half the spread against the mid price.
- *Broker commission*, on buys and sells.
- *Taiwan securities transaction tax*: 0.1% on ETF **sales** only.
- *FX spread*: converting TWD to USD (or back) for foreign-asset sleeves.

Costs are fractions of portfolio value, so a 10% trade into a sleeve with
total buy costs of 0.2% costs 0.02% of the portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Per-sleeve cost rates, aligned to a fixed sleeve order."""

    sleeves: tuple[str, ...]
    holding: np.ndarray      # annual drag charged daily on each sleeve's weight
    buy: np.ndarray          # cost per unit of value bought
    sell: np.ndarray         # cost per unit of value sold
    periods_per_year: int = 252

    @classmethod
    def build(cls, cost_cfg: dict[str, dict[str, float]], sleeves: list[str],
              periods_per_year: int = 252) -> "CostModel":
        """Turn the ``costs`` section of the config into rate vectors."""
        missing = set(sleeves) - set(cost_cfg)
        if missing:
            raise KeyError(f"No cost settings for sleeves {sorted(missing)}")
        g = lambda s, k: float(cost_cfg[s].get(k, 0.0))  # noqa: E731
        holding = np.array([g(s, "fund_fee") - g(s, "proxy_fee") + g(s, "withholding") for s in sleeves])
        per_trade = np.array([g(s, "half_spread") + g(s, "commission") + g(s, "fx_spread") for s in sleeves])
        sell_tax = np.array([g(s, "sell_tax") for s in sleeves])
        return cls(tuple(sleeves), holding, per_trade, per_trade + sell_tax, periods_per_year)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], sleeves: list[str]) -> "CostModel":
        """Cost model for a full run config."""
        return cls.build(cfg["costs"], sleeves, cfg["backtest"].get("periods_per_year", 252))

    @classmethod
    def zero(cls, sleeves: list[str]) -> "CostModel":
        """A frictionless world, for measuring how much costs matter."""
        z = np.zeros(len(sleeves))
        return cls(tuple(sleeves), z, z, z)

    def daily_holding_cost(self, w: np.ndarray) -> float:
        """Today's fee and withholding drag as a fraction of portfolio value."""
        return float(w @ self.holding) / self.periods_per_year

    def trading_cost(self, w_from: np.ndarray, w_to: np.ndarray) -> float:
        """Cost of moving from weights ``w_from`` to ``w_to``, as a fraction of portfolio value."""
        trade = w_to - w_from
        return float(np.clip(trade, 0, None) @ self.buy + np.clip(-trade, 0, None) @ self.sell)

    def as_frame(self) -> pd.DataFrame:
        """The annual holding drag and per-trade costs per sleeve, for display."""
        return pd.DataFrame({"holding_per_year": self.holding, "buy_cost": self.buy,
                             "sell_cost": self.sell}, index=list(self.sleeves))
