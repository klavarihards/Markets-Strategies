# Which Portfolio Holds Up for Real Clients? A Walk-Forward Study for a Taiwan-Based Investor

*Educational research project. Hypothetical clients; not investment advice.*

## Summary

Six ways of building a multi-asset portfolio were tested for three client
risk profiles, walk-forward with costs and taxes, on 2004-2026 data in TWD:
equal weight, mean-variance with Ledoit-Wolf shrinkage, Black-Litterman,
risk parity, hierarchical risk parity, and a CVaR-minimising portfolio.
Then three overlays were tested on top: GARCH volatility targeting, a
regime filter, and momentum tilts. The last two years were locked away
before development and opened once at the end.

**Findings**
1. **No method reliably beat equal weight on risk-adjusted return**, in
   development (0 of 60 Sharpe comparisons significant; 3 of 72 once the
   CVaR method is added, about what chance alone gives) or in the hold-out
   (every interval contains zero).
2. **Downside control is where the methods differed.** The CVaR portfolio
   was the only method that kept a conservative client within a -10% loss
   limit: worst drawdown -6.0% (2008-2024), against -11.6% to -16.6% for the
   others. A volatility cap could not do it, even when tightened so that its
   own arithmetic fits the limit.
3. **Overlays helped in one crisis and cost money otherwise.** Regime
   filters and volatility targeting cut the median balanced portfolio's
   2008-09 drawdown from -20% to -15%, but changed almost nothing from 2010
   on, and in the calm hold-out they only added turnover. Momentum tilts
   failed every test.
4. **Out of sample, the frozen portfolios did what they promised clients**
   (every loss limit held, with worst drawdowns of -5.2% to -10.0%) but
   lagged a two-year boom in which 100% Taiwan equity returned 42% a year.

## 1. Question and design

*What should a Taiwan retail client hold to reach a goal (a home deposit in
10 years, retirement in 30) without breaching the loss they can tolerate,
and which method of choosing the weights holds up out of sample?*

- **Universe:** Taiwan, US and international equity, long and intermediate
  US Treasuries, gold, cash. Long-history US-listed proxies (EWT, SPY,
  EFA+EEM, TLT, IEF, GLD), 3-month T-bill; returns converted to TWD; cash
  treated as a TWD deposit.
- **Profiles:** conservative / balanced / growth, with limits on equity,
  bonds+cash and gold, a 40% cap per holding, no leverage, a volatility band
  and a maximum acceptable loss (-10% / -20% / -30%).
- **Backtest:** quarterly walk-forward with 3- and 5-year estimation
  windows; fund fees above the proxy's own, dividend withholding, spreads,
  commission, the 0.1% Taiwan ETF sell tax, FX spreads. Monthly overlay
  checks in Phase 4.
- **Discipline:** frozen, hashed data snapshots; hold-out locked from the
  start; Phase 4 decision rules pre-registered and committed before running;
  every backtest in a trial registry (336 development trials); deflated
  Sharpe ratio; every optimiser cross-checked against PyPortfolioOpt or
  Riskfolio-Lib (agreement 1e-6 or better).

## 2. Results on development data (Dec 2007 – Oct 2024, 3-year window)

**Baselines (TWD, after costs).** Equal weight 5.4% a year, volatility 8.8%,
Sharpe 0.52, worst drawdown -22.4%; 60/40 5.9%, 11.3%, 0.47, -28.9%; 100%
Taiwan equity 8.2%, 23.6%, 0.41, -58.1%. Costs took about 0.4% a year.

**Methods, Sharpe (worst drawdown).**

| Profile | Equal weight | Mean-variance | Black-Litterman | Risk parity | HRP | Min-CVaR |
|---|---|---|---|---|---|---|
| Conservative | 0.50 (-15.4%) | 0.50 (-14.8%) | 0.43 (-16.6%) | 0.52 (-11.8%) | 0.53 (-11.6%) | 0.54 (-6.0%) |
| Balanced | 0.50 (-22.9%) | 0.45 (-27.0%) | 0.48 (-21.8%) | 0.54 (-18.3%) | 0.53 (-18.3%) | 0.55 (-17.1%) |
| Growth | 0.44 (-35.1%) | 0.46 (-36.4%) | 0.50 (-30.8%) | 0.47 (-33.1%) | 0.46 (-33.4%) | 0.54 (-30.8%) |

Mean-variance traded 28-69% of the portfolio a year (the others 8-30%) and was
the most fragile across settings (Sharpe 0.40-0.58 for balanced clients,
depending on rebalancing frequency and the holding cap); min-CVaR had the
best Sharpe in 9 of 11 robustness settings.

**Risk and goals.** Replaying today's weights through 2008: 60/40 -26.5%,
100% Taiwan -61.7%, conservative min-CVaR -5.3%. Most portfolios' risk is
equity risk: 60/40 has 60% of its money but 75% of its volatility in
equities. A 30-year growth client reaches NT$12 million in 46% of simulated
futures with 60/40 and 63-68% with the best methods; the balanced client's
chance rises from 83% to 94% with NT$2,000 more a month. Tail estimates
depend on the simulation: the chance of a fall beyond -30% for growth
clients is 23-56% by block bootstrap but 7-24% with a fitted Student-t.

**Overlays (Phase 4, pre-registered rule: keep if the median method's risk
improves ≥10% at a cost of ≤0.05 Sharpe).** Regime filters kept for every
profile, volatility targeting for balanced and growth, momentum tilts for
none. The benefit is almost all 2008-09.

**Frozen choice (pre-registered rule).** Conservative: min-CVaR with all
overlays (Sharpe 0.69, worst drawdown -7.4%). Balanced: min-CVaR with the
HMM regime filter (0.60, -12.6%). Growth: Black-Litterman with all overlays
(0.60, -18.2%). Deflated Sharpe ratios after 223 distinct trials: 0.97,
0.94, 0.94 (equal weight 0.89, 60/40 0.84).

## 3. Hold-out (23 Oct 2024 – 22 Oct 2026), opened once

| | Annual return | Volatility | Sharpe | Worst drawdown | Client limit |
|---|---|---|---|---|---|
| Conservative choice | 9.5% | 6.3% | 0.85 | -5.2% | -10% (held) |
| Balanced choice | 11.8% | 7.4% | 1.01 | -6.0% | -20% (held) |
| Growth choice | 13.9% | 12.7% | 0.78 | -10.0% | -30% (held) |
| Equal weight | 15.2% | 10.9% | 0.99 | -7.1% | |
| 60/40 | 15.6% | 12.8% | 0.89 | -10.4% | |
| 100% Taiwan equity | 42.3% | 27.6% | 1.27 | -23.5% | |

No Sharpe difference against 1/N or 60/40 is significant. No overlay passed
the keep rule out of sample; the growth choice's cost drag rose to 1.0% a
year. Mean-variance, the most fragile method in development, had the best
hold-out Sharpe in every profile (1.15-1.26): two trending years favour
historical means, which does not make them reliable.

## 4. What failed, and what it taught

- **Momentum tilts**: more turnover, no risk benefit, in every profile.
- **My own keep rule** let "all overlays" pass as a package although it
  contained the failed tilt; packages should only be built from components
  that pass alone.
- **The conservative profile** as specified was inconsistent: a 7%
  volatility cap implies a -11.5% bad year against a -10% limit.
- **Bugs found in review** and fixed before any decision used them: a
  partial trading day in the first download; mislabelled goal results;
  a deflated Sharpe ratio computed on raw instead of excess returns; a
  factor regression that double-counted the currency; an unadjusted 4-for-1
  split in Yahoo's 0050 series. Each is in `RESEARCH_LOG.md`.

## 5. Limitations

History is one sample path with three real crises; proxies stand in for the
Taiwan-listed funds (0050 vs EWT: correlation 0.92, tracking error 7.7% a
year); cash earns the US T-bill rate; cost, tax and duration figures are
approximations; goals are nominal; the 336 trials are correlated, so the
deflated Sharpe ratio is approximate; the product's numbers are simulations,
not guarantees, and a model cannot see life events, which is the advisor's
job.

## 6. Reproducing every number

`notebooks/01`–`05` (Colab-ready) run on the committed snapshot
`570bdef9e8ea`; every run records its config hash, snapshot id and seed;
`experiments/registry.csv` lists every backtest; `pytest` (96 tests,
synthetic data) runs in CI on every push.
