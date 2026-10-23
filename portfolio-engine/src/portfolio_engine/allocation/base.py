"""What every allocation method looks like to the backtest.

An allocator is anything callable as ``allocator(history) -> weights``:

- ``history`` is a DataFrame of daily sleeve returns up to and including
  the rebalance date. Nothing after that date is ever passed in, which is
  what makes the backtest walk-forward.
- ``weights`` is a Series indexed by sleeve, non-negative, summing to 1
  (client portfolios use no leverage and no short positions).

Keeping this interface tiny means the five methods in Phase 2 and the
benchmarks here are interchangeable inside the same backtest loop.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Allocator = Callable[[pd.DataFrame], pd.Series]


def validate_weights(w: pd.Series, sleeves: pd.Index, tol: float = 1e-6) -> pd.Series:
    """Check a weight vector is a valid long-only, fully invested portfolio.

    Missing sleeves get weight 0. Tiny negative values from a numerical
    solver (e.g. -1e-12) are clipped to zero; anything materially negative,
    or a total that is not 1, is an error rather than something to hide.
    """
    w = w.reindex(sleeves).fillna(0.0).astype(float)
    if not np.isfinite(w.values).all():
        raise ValueError(f"Non-finite weights: {w.to_dict()}")
    if (w < -tol).any():
        raise ValueError(f"Negative weights are not allowed in client portfolios: {w[w < 0].to_dict()}")
    w = w.clip(lower=0.0)
    if abs(w.sum() - 1.0) > tol:
        raise ValueError(f"Weights sum to {w.sum():.6f}, not 1")
    return w / w.sum()
