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
| 1 | Data snapshots, TWD returns, walk-forward backtest with costs, three benchmarks | Done; run on real data |
| 2 | Five allocation methods under three client risk profiles, cross-checked against PyPortfolioOpt / Riskfolio-Lib | Done; run on real data |
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

From the runs in `notebooks/01_data_and_baseline.ipynb` and
`notebooks/02_allocation_methods.ipynb`: snapshot `570bdef9e8ea` (downloaded
2026-10-23, committed in `data/snapshots/`), development data to 22 Oct 2024;
hold-out (23 Oct 2024 – 22 Oct 2026) not yet opened. Config hashes
`eb292432cba0` (Phase 1) and `03f45b1dbe9e` (Phase 2). All figures in TWD,
after costs. Cash is treated as a TWD deposit (see HOW_IT_WORKS section 2;
the first run converted it from USD and was redone before Phase 3).

**Baselines, 3-year window (31 Dec 2007 – 22 Oct 2024)**

| | Annual return | Volatility | Sharpe | Max drawdown | Cost drag / yr |
|---|---|---|---|---|---|
| Equal weight | 5.4% | 8.8% | 0.52 | -22.4% | 0.40% |
| 60/40 | 5.9% | 11.3% | 0.47 | -28.9% | 0.41% |
| 100% Taiwan equity | 8.2% | 23.6% | 0.41 | -58.1% | -0.16% |

(The Taiwan cost drag is negative because the assumed 0050 fee is below EWT's.)

**Five methods under three profiles, 3-year window: Sharpe (max drawdown)**

| Profile | Equal weight | Mean-variance | Black-Litterman | Risk parity | HRP |
|---|---|---|---|---|---|
| Conservative | 0.50 (-15.4%) | 0.50 (-14.8%) | 0.43 (-16.6%) | 0.52 (-11.8%) | 0.53 (-11.6%) |
| Balanced | 0.50 (-22.9%) | 0.45 (-27.0%) | 0.48 (-21.8%) | 0.54 (-18.3%) | 0.53 (-18.3%) |
| Growth | 0.44 (-35.1%) | 0.46 (-36.4%) | 0.50 (-30.8%) | 0.47 (-33.1%) | 0.46 (-33.4%) |

**5-year window (31 Dec 2009 – 22 Oct 2024)**

| Profile | Equal weight | Mean-variance | Black-Litterman | Risk parity | HRP |
|---|---|---|---|---|---|
| Conservative | 0.66 (-11.8%) | 0.70 (-12.3%) | 0.60 (-10.2%) | 0.62 (-12.9%) | 0.60 (-12.2%) |
| Balanced | 0.70 (-13.1%) | 0.79 (-17.3%) | 0.68 (-13.9%) | 0.68 (-13.2%) | 0.70 (-12.4%) |
| Growth | 0.64 (-18.4%) | 0.75 (-21.8%) | 0.72 (-19.3%) | 0.67 (-18.2%) | 0.65 (-18.2%) |

What the tables say:

- **No method is distinguishable from the benchmarks.** None of the 60 Sharpe
  differences against 1/N and 60/40 has a 90% bootstrap interval that excludes
  zero; all gaps lie between -0.08 and +0.14.
- **Mean-variance** has the highest Sharpe in every profile with the 5-year
  window, but trades 28-69% of the portfolio a year (others: 9-15%).
  With the 3-year window it is never the best.
- **Risk parity and HRP** gave the smallest drawdowns for conservative and
  balanced clients with the 3-year window, which includes 2008; with the
  5-year window that advantage is gone.
- **Black-Litterman without views** was the weakest for conservative clients.
- **Client limits:** the conservative -10% loss limit was breached by every
  method in both windows; the growth -30% limit by every method in the 3-year
  window; the balanced -20% limit by equal weight, mean-variance and
  Black-Litterman in the 3-year window. The volatility cap could not be met in
  16-18% of quarters for balanced and growth with the 3-year window, and
  20-25% for growth with the 5-year window.
- **The profile does much of the work:** risk parity and HRP were moved
  7-44% of the portfolio, on average, to fit the profile limits.

Hold-out (23 Oct 2024 – 22 Oct 2026): [RESULT], opened once at the end of Phase 3.
