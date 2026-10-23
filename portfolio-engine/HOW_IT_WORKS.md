# How the Portfolio Engine Works

This explains every part of the engine in plain language: what it does, why
it is there, and what can go wrong. It grows with each phase. Sections for
Phases 2 and 3 are filled in when those phases are built.

---

## 1. The question

A client in Taiwan has a goal (say, NT$3 million in 10 years), some savings,
a monthly contribution, and a limit on how big a loss they can live with. The
engine answers two questions for them:

1. **What should I hold?** Weights across seven building blocks.
2. **How likely am I to reach my goal?** A probability, with a range of
   outcomes, not a single forecast.

The research question underneath is: *which way of choosing the weights
holds up best on data it has never seen, after costs?* Expected returns are
very hard to estimate, and sophisticated optimisers are known to turn small
estimation errors into extreme portfolios. So the engine compares methods
from the simplest (equal weight) to the most elaborate, and reports the
ones that failed as well as the ones that worked.

---

## 2. Data: seven building blocks

| Sleeve | Role | Research proxy | What a client would buy |
|---|---|---|---|
| Taiwan equity | Home-market growth | EWT (iShares MSCI Taiwan) | 0050 |
| US equity | Global growth | SPY | 00646 |
| International equity | Diversify away from US and Taiwan tech | 50% EFA + 50% EEM | e.g. 00645 |
| Long US Treasuries | Crash hedge | TLT | 00679B |
| Intermediate bonds | Stable income | IEF | a Taiwan-listed IG bond ETF |
| Gold | Inflation and crisis hedge | GLD | 00635U |
| Cash | Liquidity, goal buffer | 3-month T-bill (FRED DTB3) | TWD deposit or money-market fund |

**Why proxies?** The Taiwan-listed funds a client would actually buy are
young (some started in 2017). A backtest needs a long history that includes
several crises, so the research uses US-listed funds with long histories,
and checks the real Taiwan funds only over the period where both exist.

**Why EWT and not the TAIEX index?** The TAIEX (`^TWII` on Yahoo) is a
*price* index: it ignores dividends. Taiwan stocks pay roughly 3–4% a year
in dividends, so using it would quietly make Taiwan look several percent a
year worse than it was. EWT's adjusted price includes dividends.

**Why does the data start in November 2004?** That is when GLD launched. A
portfolio can only be backtested from the date when every building block
exists.

**Adjusted prices.** Yahoo's adjusted close treats every dividend as if it
had been reinvested, so returns computed from it are *total* returns.

**What can go wrong**
- A proxy is not the real fund. 00635U holds gold *futures*, which carry
  roll costs that GLD (physical gold) does not have. The gap is measured
  later, not assumed away.
- The 50/50 EFA/EEM blend is rebalanced every day inside the data. That is
  a small simplification of how a real holding would behave.
- Cash is modelled with the US T-bill rate. A Taiwan client's cash would
  earn TWD deposit rates, which have been much lower, and has no currency
  risk. Converting the US T-bill into TWD (as the engine does) gives cash a
  currency risk it would not really have. This is a known limitation.

---

## 3. Currency: everything in TWD

The client spends Taiwan dollars, so returns are measured in TWD. A USD
asset's return to a TWD investor is

```
r_TWD = (1 + r_USD) * (1 + r_FX) - 1
```

where `r_FX` is the change in the number of TWD per USD.

*Example:* SPY gains 10% in a year, but the dollar falls 10% against TWD.
To the client that is `1.10 × 0.90 − 1 = −1%`, roughly nothing.

Results are reported in both TWD and USD, so currency's effect is shown
rather than hidden. Hedged results come in a later phase.

---

## 4. Frozen data snapshots

**What:** data is downloaded once (`data/download.py`, run in Colab), saved
as Parquet files, and each file's SHA-256 fingerprint is written to
`manifest.json`. Every run loads the frozen files and checks the
fingerprints first.

**Why:** data providers change history. Yahoo re-adjusts past prices when
it reprocesses dividends; FRED revises series. Without a frozen snapshot,
the same code could give different answers next month, and nobody could
tell whether a result changed because of the code or the data.

**What can go wrong:** if a snapshot file is edited, even by one value,
the loader refuses it (`SnapshotIntegrityError`). The fix is a new download
into a new folder, never editing the old one.

---

## 5. The hold-out lock

**What:** the last two years of the snapshot are cut off before any
research code sees them. `load_market_data` removes them by default; only
`unlock_holdout=True` returns them, and every unlock is written with a
timestamp to `results/holdout_log.jsonl`. In the notebooks, only the last
cell, switched off by default, unlocks it.

**Why:** every time you look at results and change something, you fit the
method a little more to that particular history. After enough tweaks a
backtest can look excellent purely by chance. A hold-out you never looked
at is the only honest test of whether the method works on new data. The
same discipline was used in the Brent–WTI project in this repo.

**What can go wrong**
- *Opening it more than once.* If the hold-out is opened, a change is
  made, and it is opened again, it has become development data. The log
  makes the number of openings visible.
- *Knowing what happened anyway.* We all know roughly how markets did
  recently, and that knowledge can leak into design choices. A locked
  hold-out reduces this risk but cannot remove it.

---

## 6. Walk-forward backtest

**What happens each day** (`backtest/engine.py`):

1. The portfolio holds weights `w`. Today's return is the weighted average
   of the sleeves' returns, `w · r`.
2. The day's share of fund fees and dividend withholding is deducted.
3. Prices moved, so the weights **drift**: if equities rose, they are now a
   bigger share of the portfolio. Nothing is traded between rebalances.
4. On the **last trading day of each quarter**, the allocation method is
   shown the returns from the past 3 (or 5) years up to that day, and
   returns new target weights. The portfolio trades from its drifted
   weights to the target and pays the trading costs.

**Why walk-forward:** a method is only ever given data up to the day of
the decision, exactly as it would have been in real life. Using the full
history to choose weights and then testing on that same history is
*look-ahead bias*, the most common way backtests flatter themselves.

**How we know it works:** tests in `tests/test_engine.py` check that

- the method is never shown a date later than the rebalance date;
- scrambling all returns after some date leaves every result before that
  date unchanged;
- a run on development data is an exact prefix of a run on the full data,
  so the hold-out evaluation continues the same portfolio.

**Fair comparison:** within one estimation window, all methods start on
the same date (the first quarter-end with a full window of history) and
start from cash, paying the cost of buying the portfolio.

**What can go wrong**
- Trading is assumed to happen at the closing price on the rebalance day,
  using that day's close in the estimate. In reality an order is placed
  slightly later. The effect is small for quarterly rebalancing, but it is
  an assumption.
- Paying trading costs slightly reduces the portfolio, so after a
  rebalance the weights are a hair off target. This second-order effect is
  ignored.

---

## 7. Costs and taxes

Costs are part of the return. Each sleeve has its own rates in the config
(`configs/phase1_baseline.yaml`).

**Holding costs** (charged every day, as a yearly rate divided by 252):

- **Fund fee (expense ratio).** The trap: the research proxy's adjusted
  prices are *already* net of the proxy's own fee. Charging the real
  fund's full fee on top would count fees twice. So the engine charges
  only the difference, `fund_fee − proxy_fee`. If the Taiwan-listed fund
  is cheaper than its proxy (0050 vs EWT), this is a small credit.
- **Dividend withholding.** Tax withheld at source on dividends, entered
  as an annual drag (e.g. about 30% of a ~1.3% US dividend yield).

**Trading costs** (paid on the value traded at each rebalance):

- **Half the bid-ask spread.** You buy at the higher price and sell at the
  lower; each trade loses about half the gap against the middle price.
- **Broker commission.** Taiwan's list rate is 0.1425% per trade, usually
  discounted.
- **Taiwan securities transaction tax.** 0.1% on ETF **sales** only.
  This is why new contributions should be used to rebalance first: buying
  avoids the tax.
- **FX spread** on foreign-asset sleeves: the cost of converting between
  TWD and USD.

*Example:* moving 10% of the portfolio out of US equity (sell costs
0.1% spread + 0.0855% commission + 0.1% tax + 0.05% FX ≈ 0.34%) costs
about 0.034% of the whole portfolio.

**What can go wrong:** the rates in the config are **assumptions**, not
measurements. They must be checked against each fund's latest fact sheet
and the broker's fee schedule before the final hold-out run. Taxes also
depend on each client's situation and are simplified here.

---

## 8. Benchmarks

Every method has to beat three simple portfolios, all rebalanced quarterly
and charged the same costs:

- **Equal weight (1/N):** the same amount in each of the seven sleeves. It
  estimates nothing, so it cannot suffer from estimation error. DeMiguel,
  Garlappi and Uppal (2009) found that none of 14 optimisation models
  reliably beat it out of sample. That makes it the honest bar.
- **60/40:** 60% equity (split equally across Taiwan, US and international)
  and 40% bonds (split equally between long Treasuries and intermediate
  bonds). It is what most "balanced" products look like. The split is a
  choice and is set in the config.
- **100% Taiwan equity:** what many Taiwan retail investors actually hold.
  Beating it on risk is easy; the question is how much return
  diversification gives up, if any.

---

## 9. Metrics

One definition each, used everywhere (`backtest/metrics.py`):

| Metric | Plain meaning |
|---|---|
| Annualised return (CAGR) | The steady yearly growth rate that turns the start value into the end value. Geometric, so +50% then −50% is correctly a loss. |
| Volatility | Standard deviation of daily returns × √252. How much returns bounce around. |
| Sharpe ratio | Average return **above cash**, per unit of volatility. "How much extra return did each unit of risk earn?" |
| Sortino ratio | Like Sharpe, but only falls count as risk. |
| Maximum drawdown | Worst fall from a previous peak: what a client who bought at the top would have lost. |
| Recovery days | Trading days from the bottom of the worst drawdown back to the old peak. |
| Calmar ratio | Annualised return ÷ size of the maximum drawdown. |
| CVaR 95% | The average return on the worst 5% of days (or months). VaR says where the bad tail starts; CVaR says how bad it is inside. |
| Turnover | Share of the portfolio traded per year (one-way). |
| Tracking error | Volatility of the difference from the 60/40 benchmark. |
| Cost drag | Annual return before costs minus after costs. |

**Why √252?** If daily returns are roughly independent, variances add up
over days. Variance over a year is 252 × daily variance, so standard
deviation is √252 × daily standard deviation.

**A consistency note.** Earlier notebooks in this repo sometimes computed
Sharpe as CAGR ÷ volatility. That mixes a geometric average with an
arithmetic risk measure and ignores the return on cash. Here Sharpe is
always the mean excess return over its standard deviation.

---

## 10. Reproducibility

Every number must be rebuildable:

- **Config hash:** each run's settings come from one YAML file; a 12-character
  hash of the settings is stored with the results.
- **Snapshot id:** a hash of all data-file fingerprints.
- **Seed:** anything random (synthetic test data now, Monte Carlo later)
  uses a fixed seed from the config.
- **Run record:** `save_run` writes `run.json` with the config hash,
  snapshot id, seed, data dates, and package versions next to the results.

The Phase 1 gate, "baseline numbers reproduce exactly from a clean run",
is checked by `tests/test_reproducibility.py` (on synthetic data) and by
Step 4 of the Phase 1 notebook (on real data).

**Tests use synthetic data.** `data/synthetic.py` invents fat-tailed,
correlated price histories in the same format as a real snapshot, so every
test runs offline and never depends on what markets did. Nothing computed
from synthetic data is a result.

---

## 11. Coming next

- **Phase 2:** mean-variance with Ledoit-Wolf shrinkage, Black-Litterman,
  risk parity and hierarchical risk parity, under the three client risk
  profiles, each cross-checked against PyPortfolioOpt or Riskfolio-Lib.
- **Phase 3:** stress tests, risk decomposition, CVaR, and Monte Carlo
  goal probabilities; then the one-time hold-out evaluation.
