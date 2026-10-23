# Goal-Based Multi-Asset Portfolio Engine

Educational research project, not investment advice. Fund names are used only
as liquid, representative Taiwan-listed instruments, not as recommendations.

The engine builds portfolios for a hypothetical Taiwan-based client from seven
building blocks, compares allocation methods walk-forward with costs and
taxes, and reports the chance of reaching a goal. How each part works, and
what can go wrong, is in [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Status

| Phase | Content | Status |
|---|---|---|
| 1 | Data snapshots, TWD returns, walk-forward backtest with costs, three benchmarks | Built; awaiting first run on real data |
| 2 | Five allocation methods under three client risk profiles, cross-checked against PyPortfolioOpt / Riskfolio-Lib | Built; awaiting first run on real data |
| 3 | Stress tests, risk decomposition, CVaR, Monte Carlo goal probabilities, hold-out | Not started |

## Layout

```
configs/                 one YAML file per run
src/portfolio_engine/
  data/                  download (Colab), snapshots, synthetic test data, loader + hold-out lock
  estimators/            Ledoit-Wolf covariance, historical means, Black-Litterman
  allocation/            benchmarks, risk profiles, the five methods, shared cvxpy solver
  backtest/              cost model, walk-forward engine, metrics, experiment runner
  risk/                  Phase 3
notebooks/               one Colab notebook per phase
tests/                   pytest, all on synthetic data
```

## Running it

In Colab, open `notebooks/01_data_and_baseline.ipynb` (then `02_allocation_methods.ipynb`), set `DOWNLOAD = True`
for the first run, and keep the snapshot folder it writes (commit it or copy
it to Drive). Every later run reads that frozen snapshot.

Locally:

```
pip install -e ".[dev]"
pytest
```

## Results

Results are added only after a run on real data; every number will come from
a saved run (`results/<config>_<hash>_<snapshot>/`).

- Baselines (3-year window, after costs, TWD): [RESULT]
- Five methods x three profiles, comparison and compliance tables: [RESULT]
