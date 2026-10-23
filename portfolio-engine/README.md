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
| 3 | Stress tests, risk decomposition, CVaR variant, Monte Carlo goal probabilities | Done; run on real data |

## Layout

```
configs/                 one YAML file per run
src/portfolio_engine/
  data/                  download (Colab), snapshots, synthetic test data, loader + hold-out lock
  estimators/            Ledoit-Wolf covariance, historical means, Black-Litterman
  allocation/            benchmarks, risk profiles, the five methods, shared cvxpy solver
  backtest/              cost model, walk-forward engine, metrics, experiment runner
  risk/                  stress tests, risk decomposition, Monte Carlo goals, analysis runner
notebooks/               one Colab notebook per phase
tests/                   pytest, all on synthetic data
```

## Running it

In Colab, open `notebooks/01_data_and_baseline.ipynb` (then `02_allocation_methods.ipynb` and `03_risk_and_goals.ipynb`), set `DOWNLOAD = True`
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

### Phase 3: risk and goals

From `notebooks/03_risk_and_goals.ipynb` (config `847159c84c0e`). Today's
portfolios are those chosen at the last quarter-end before the hold-out
(30 Sep 2024), with the 3-year window.

**Worst fall when today's weights are replayed through past crises (TWD)**

| Portfolio (today's weights) | 2008 GFC | 2011 euro | 2013 taper | 2020 COVID | 2022 inflation |
|---|---|---|---|---|---|
| Conservative: min-CVaR | -5.3% | -1.8% | -4.0% | -4.7% | -4.6% |
| Conservative: risk parity | -10.7% | -2.1% | -5.0% | -8.5% | -8.0% |
| Balanced: risk parity | -19.0% | -3.6% | -6.8% | -13.4% | -13.2% |
| Balanced: mean-variance | -31.8% | -9.3% | -4.6% | -18.6% | -10.5% |
| Growth: Black-Litterman | -29.0% | -7.8% | -7.3% | -19.6% | -14.3% |
| 60/40 | -26.5% | -6.7% | -7.6% | -16.7% | -16.8% |
| 100% Taiwan | -61.7% | -25.8% | -10.6% | -26.5% | -29.0% |

**Chance of reaching the goal (10,000 block-bootstrap paths of each portfolio's walk-forward returns after costs)**

| Client (goal) | Equal weight | Mean-variance | Black-Litterman | Risk parity | HRP | Min-CVaR | 60/40 |
|---|---|---|---|---|---|---|---|
| Conservative (NT$1.7M in 5 yrs) | 88% | 85% | 85% | 90% | 90% | 89% | 86% |
| Balanced (NT$3M in 10 yrs) | 78% | 74% | 79% | 83% | 81% | 79% | 80% |
| Growth (NT$12M in 30 yrs) | 51% | 63% | 60% | 56% | 53% | 68% | 46% |

What the analysis says:

- **The CVaR variant was the only method that kept a conservative client
  inside the -10% loss limit**: worst walk-forward drawdown -6.0% (3-year
  window) and -5.9% (5-year), and at most -5.3% in every replayed crisis.
  It gave up return for it (3.3% a year, Sharpe 0.54).
- **The conservative profile is internally inconsistent.** Its 7% volatility cap
  implies a 1-in-20 bad year of about 11.5% (1.645 x 7%), already beyond
  the client's own -10% limit. A calibration change is tested in Phase 4.
- **Money is not risk.** 60/40 has 60% of its money in equities and 75% of its
  volatility from them; balanced mean-variance 60% and 96%.
- **Shocks:** a 2-point rise in US rates costs up to -9.8% (60/40); if every risky
  sleeve moved together (correlation 0.8), a 1-in-20 bad year for the conservative
  portfolios would be -11.0% to -16.1%, all beyond their limit.
- **Goals:** conservative and balanced clients reach their goals in 74-90% of
  simulated futures whatever the method; the 30-year growth client is where the
  choice matters (46% with 60/40, 63% mean-variance, 68% min-CVaR). For the balanced
  client with risk parity, NT$2,000 more a month raises the odds from 83%
  to 94%, two more years to 98%.
- **Model choice matters in the tail.** For the growth client, the chance of a
  fall beyond -30% is 23%-56% with the bootstrap but 7%-24% with a fitted
  Student-t: resampling whole years keeps crashes clustered; a fitted distribution
  spreads them out. Both are reported.
- **Statistics:** 3 of the 72 Sharpe comparisons (now including min-CVaR) are
  "significant" at 90%, but about 7 would be by chance alone. Not evidence of an edge.

Hold-out (23 Oct 2024 – 22 Oct 2026): [RESULT], opened once after Phase 4, when every choice is frozen.
