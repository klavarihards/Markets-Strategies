"""Where does a portfolio's risk come from?

A portfolio can look diversified by weight and still get almost all of its
risk from one place: 40% in equities can be 80% of the risk, because
equities are so much more volatile than bonds. This module splits two risk
measures into one contribution per sleeve, and the contributions add up
exactly to the total:

- **Volatility contribution.** Sleeve i contributes ``w_i (Sigma w)_i``
  to the portfolio variance ``w' Sigma w``: its weight times how much it
  moves with the whole portfolio. A sleeve that is volatile *and* moves with
  everything else contributes most; a sleeve that tends to rise when the
  rest falls can contribute less than nothing.
- **CVaR contribution.** Take the worst 5% of days for the portfolio. On
  those days, how much of the average loss came from each sleeve? That is
  ``w_i * (average return of sleeve i on the portfolio's worst days)``.
  This looks only at the tail, where clients actually get hurt.
"""

from __future__ import annotations

import pandas as pd


def volatility_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Each sleeve's share of portfolio variance (shares sum to 1)."""
    x = w.reindex(cov.index).fillna(0.0).to_numpy()
    m = cov.to_numpy() @ x
    return pd.Series(x * m / (x @ m), index=cov.index)


def cvar_contributions(w: pd.Series, returns: pd.DataFrame, alpha: float = 0.95) -> tuple[pd.Series, float]:
    """Each sleeve's contribution to historical CVaR, and the CVaR itself.

    The contributions (in return units, negative = loss) add up to the
    portfolio's CVaR exactly, because the portfolio return on each tail day
    is the sum of the sleeves' weighted returns on that day.
    """
    x = w.reindex(returns.columns).fillna(0.0)
    port = returns @ x
    tail = port <= port.quantile(1 - alpha)
    contrib = returns[tail].mul(x, axis=1).mean()
    return contrib, float(port[tail].mean())


def decompose(w: pd.Series, cov: pd.DataFrame, returns: pd.DataFrame, alpha: float = 0.95,
              asset_class: dict[str, str] | None = None) -> pd.DataFrame:
    """Weight, share of volatility and share of CVaR for each sleeve (optionally grouped by asset class)."""
    contrib, total = cvar_contributions(w, returns, alpha)
    out = pd.DataFrame({"weight": w.reindex(cov.index).fillna(0.0),
                        "share_of_volatility": volatility_contributions(w, cov),
                        "share_of_cvar": contrib / total})
    if asset_class is not None:
        out = out.groupby(pd.Series(asset_class)).sum()
    return out
