"""Goal-based multi-asset portfolio engine.

The package is split the same way the research question is split:

- ``data``        download, snapshot and load market data (with the hold-out lock)
- ``estimators``  turn past returns into inputs (covariances, expected returns)
- ``allocation``  turn inputs into portfolio weights
- ``risk``        stress tests, risk decomposition, goal probabilities
- ``backtest``    walk-forward simulation, costs and performance metrics
"""

__version__ = "0.1.0"
