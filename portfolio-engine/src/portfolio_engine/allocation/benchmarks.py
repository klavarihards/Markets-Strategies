"""The three benchmarks every method must face.

- **Equal weight (1/N).** The same amount in every sleeve. It needs no
  estimates at all, so it cannot suffer from estimation error. DeMiguel,
  Garlappi and Uppal (2009) found that none of 14 optimisation models
  reliably beat it out of sample, which makes it the honest bar to clear.
- **60/40.** The classic balanced portfolio: 60% equities, 40% bonds. It is
  what most "moderate risk" products look like, so a method that cannot beat
  it gives an advisor no reason to switch.
- **100% Taiwan equity.** What many Taiwan retail investors actually hold
  (one 0050-style fund). Beating it on risk is easy; the question is how much
  return a diversified portfolio gives up, if any.

All three hold fixed target weights and are rebalanced back to them each
quarter, so they still pay trading costs and taxes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import Allocator


class FixedWeights:
    """Always return the same target weights, whatever the history says."""

    def __init__(self, weights: dict[str, float], name: str = "fixed"):
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Fixed weights must have a positive total")
        self.weights = {k: v / total for k, v in weights.items()}
        self.name = name

    def __call__(self, history: pd.DataFrame) -> pd.Series:
        return pd.Series(self.weights).reindex(history.columns).fillna(0.0)


class EqualWeight:
    """1/N across every sleeve in the history (or a chosen subset)."""

    name = "equal_weight"

    def __init__(self, sleeves: list[str] | None = None):
        self.sleeves = sleeves

    def __call__(self, history: pd.DataFrame) -> pd.Series:
        cols = self.sleeves or list(history.columns)
        return pd.Series(1.0 / len(cols), index=cols).reindex(history.columns).fillna(0.0)


def benchmark_allocators(cfg: dict[str, Any]) -> dict[str, Allocator]:
    """Build the benchmark allocators named in the config's ``benchmarks`` section."""
    out: dict[str, Allocator] = {}
    for name, spec in cfg["benchmarks"].items():
        if spec == "equal":
            out[name] = EqualWeight()
        else:
            unknown = set(spec) - set(cfg["sleeves"])
            if unknown:
                raise KeyError(f"Benchmark {name} uses unknown sleeves {sorted(unknown)}")
            out[name] = FixedWeights(spec, name)
    return out
