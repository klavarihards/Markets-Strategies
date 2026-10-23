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
- Cash is a TWD deposit with no currency risk, but it earns the **US**
  T-bill rate, because no long daily TWD rate series is freely available.
  TWD deposit rates have usually been lower, so cash returns are somewhat
  overstated. The first version converted the T-bill into TWD instead; that
  gave "cash" a −18.8% worst drawdown from exchange-rate moves alone, which
  a real deposit never has, so it was changed (`currency: TWD` in the config)
  before any further results were looked at.

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
rather than hidden. The one exception is cash, which is treated as a TWD
deposit (see section 2). Hedged results come in a later phase.

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

## 11. Client risk profiles (Phase 2)

Suitability comes before optimisation. Each client is assigned one of three
profiles from a short questionnaire, and every portfolio must respect that
profile's limits, whatever a backtest says.

| Profile | Horizon | Equity | Bonds and cash | Gold | Volatility band | Max loss the client accepts |
|---|---|---|---|---|---|---|
| Conservative | 3–5 years | 15–35% | 55–80% | 0–10% | 4–7% | −10% |
| Balanced | 5–10 years | 40–60% | 30–50% | 0–10% | 7–11% | −20% |
| Growth | 10+ years | 65–85% | 10–30% | 0–10% | 11–16% | −30% |

These are starting assumptions from the project spec, not regulatory figures.
No single sleeve may exceed 40%, and there is no leverage or short selling.

- The **ranges** are hard limits. No method may break them.
- The top of the **volatility band** is a cap on the portfolio's *estimated*
  volatility. In a crisis, every equity is volatile, so a growth client's
  65% minimum in equity can make the cap impossible. Then the hard limits
  win, the cap is dropped for that quarter, and the event is logged
  (`share_vol_cap_relaxed` in the compliance table).
- The **maximum loss** is not something an optimiser can guarantee. It is
  checked afterwards against realised drawdowns (`max_loss_breached`).

**What can go wrong:** the limits often matter more than the method. If the
profile forces 65% equity, five methods can only differ in how they split
the remaining room. The compliance table measures how far the profile
moved each method (`avg_moved_by_profile`), so this is visible, not hidden.

---

## 12. Estimating risk: Ledoit-Wolf shrinkage

Every method except equal weight needs a covariance matrix: how volatile
each sleeve is and how they move together. The sample covariance from three
years of daily data is noisy. An optimiser treats it as exact and piles into
combinations that only *look* low-risk because of noise.

Shrinkage pulls the noisy sample matrix `S` part of the way towards a
simple, structured target `F`:

```
shrunk = delta x F + (1 - delta) x S
```

The target keeps each sleeve's own variance but gives every pair the same
average correlation. It is wrong in detail but has very little noise. The
intensity `delta` (0 to 1) is not tuned by hand: Ledoit and Wolf derive the
value that minimises the expected error, and estimate it from the data.
Fewer observations or fatter tails give a larger `delta`.

All methods use the **same** shrunk matrix, so differences in results come
from the methods, not from different inputs.

**Checked against:** PyPortfolioOpt (constant-correlation target) and
scikit-learn (identity target). The scikit-learn version matches to machine
precision. PyPortfolioOpt builds its sample matrix with 1/(T−1) instead of
the paper's 1/T; with that convention the two agree exactly, and otherwise
differ by about 0.1%.

---

## 13. The five methods

### Equal weight (1/N)

The same amount in each of the seven sleeves, then moved to the nearest
portfolio the profile allows. **Why it is here:** it estimates nothing, so
it cannot be fooled by noise. DeMiguel, Garlappi and Uppal (2009) found no
optimiser that reliably beat it out of sample. **What can go wrong:** it
ignores risk completely; gold gets as much as US equity.

### Mean-variance with shrinkage (Markowitz, 1952)

Choose weights `w` to maximise

```
expected return − (lambda / 2) × variance  =  w'mu − (lambda / 2) w'Σw
```

with `mu` the historical average returns and `Σ` the shrunk covariance.
`lambda` (risk aversion, 2.5 here) sets how much return the client gives up
to avoid a unit of variance. **Why it is here:** it is the textbook answer,
and the starting point of every other method. **What can go wrong:**
expected returns are very hard to estimate. With 20% volatility, even ten
years of data pin an average return down only to about ±6% a year. The
optimiser trusts those noisy averages, so weights jump to corners (the 40%
cap) and swing from quarter to quarter, which means high turnover and costs.

### Black-Litterman (Black and Litterman, 1990)

Mean-variance again, but with better expected returns:

1. **Reverse optimisation.** If investors with risk aversion `delta` happily
   hold a neutral market portfolio `w_mkt`, the returns they must expect are
   `pi = delta × Σ × w_mkt`. A sleeve that adds a lot of risk to the market
   portfolio must be expected to pay for it.
2. **Views** (none in Phase 2; tactical views arrive in Phase 4). A view
   such as "Taiwan equity beats US equity by 2% a year" is a row of `P`, a
   number in `Q`, and an uncertainty in `Omega`. The posterior returns are a
   confidence-weighted blend of `pi` and the views.
3. Run mean-variance on the blended returns, under the profile.

**Why it is here:** it replaces the noisiest input (historical averages)
with something stable, and gives a disciplined way to add house views.
**What can go wrong:** the neutral portfolio `w_mkt` is an assumption (a
rough global multi-asset portfolio, in the config). With no views, the
result is mostly that assumption projected onto the profile.

### Risk parity (equal risk contribution)

Ignore expected returns. Size each risky sleeve so it contributes the same
share of the portfolio's total risk. A sleeve's **risk contribution** is
its weight times its marginal risk: `w_i × (Σw)_i / w'Σw`; the shares add to
100%. Because bonds are calmer than equities, they get much more weight
than in 60/40. Cash is left out (it has almost no risk of its own, so risk
parity would put nearly everything in it) and is added back only if the
profile needs it. **Solved with:** Spinu's convex reformulation, minimise
`½ y'Σy − Σ b_i log y_i`, whose solution has exactly equal risk
contributions; Newton's method converges in a few steps. **What can go
wrong:** it leans heavily on bonds, so when stocks and bonds fall together
(2022) it has nowhere to hide. Professional risk-parity funds use leverage
to reach a target return; client portfolios here do not.

### Hierarchical risk parity (López de Prado, 2016)

1. **Cluster.** Turn correlations into distances, `d = √((1 − ρ)/2)`, and
   join the closest sleeves into a tree (stocks with stocks, bonds with
   bonds).
2. **Reorder** the sleeves so similar ones sit next to each other.
3. **Split money top-down:** cut the ordered list in half, give each half
   money in inverse proportion to its variance, and repeat inside each
   half.

**Why it is here:** it never inverts the covariance matrix, which is the
step that makes mean-variance unstable when assets are highly correlated.
**What can go wrong:** with only six risky sleeves the tree is small, and
the split-in-half rule can separate assets the tree put together. Results
also depend on the linkage method (single linkage, as in the paper).

**Checked against:** PyPortfolioOpt for mean-variance and Black-Litterman
(agreement about 1e-8 and 1e-6 in weights) and HRP (identical); Riskfolio-
Lib for risk parity (about 5e-6, Riskfolio's solver tolerance).

---

## 14. How a method meets the profile

Two cases, both solved as convex problems with cvxpy, which have a single
best answer:

- **Mean-variance and Black-Litterman** have the profile limits *inside*
  their optimisation: they find the best trade-off among portfolios the
  profile allows.
- **Equal weight, risk parity and HRP** choose weights by their own rule,
  then the solver finds the *closest* portfolio that satisfies the profile
  (smallest distance between weight vectors). If the method's choice
  already fits, nothing changes. How far it moved is logged.

---

## 15. Comparing methods honestly

- **Same days, same costs:** within an estimation window, every method
  starts on the same quarter-end, from cash, with the same cost model.
- **Benchmarks:** every profile method is compared with equal weight and
  60/40.
- **Is a gap real?** A Sharpe ratio from 15 years of data is uncertain by
  roughly ±0.25. `sharpe_difference_ci` rebuilds thousands of alternative
  histories by resampling blocks of 21 trading days (blocks keep calm and
  stormy weeks together) on the *same* days for both methods, and reports
  a 90% interval for the Sharpe difference. If it contains zero, the data
  cannot tell the methods apart, and the write-up must say so.
- **Not only Sharpe:** the compliance table checks whether each profile
  portfolio actually stayed in its volatility band and within the client's
  maximum loss. For a client, a method that wins on average but breaks
  their loss limit is the wrong method.

---

## 16. CVaR, and the CVaR variant (Phase 3)

**Value-at-risk (VaR)** answers "what loss do I exceed on my worst 5% of
days?" It says where the bad tail *starts*. **Conditional value-at-risk
(CVaR)** answers "on those worst 5% of days, what is the *average* loss?"
It says how bad the tail is inside. Two portfolios can share a VaR while
one of them has a much nastier tail; CVaR tells them apart, which is why it
is the main tail measure here. It is reported for days and for months.

**The CVaR variant** (`min_cvar`) is a sixth method aimed at conservative
clients: the portfolio, inside the profile limits, whose historical CVaR
over the estimation window is as small as possible. It uses the real daily
returns as scenarios, so crashes in the data count as they happened, with
no bell-curve assumption.

**How it is solved.** CVaR is not a smooth formula of the weights, but
Rockafellar and Uryasev (2000) showed that

```
CVaR = min over z of [ z + average(max(loss − z, 0)) / (1 − alpha) ]
```

where `z` turns out to be the VaR. With one helper variable per day for
`max(loss − z, 0)`, this is a **linear programme**, which solvers handle
exactly. **Checked against:** Riskfolio-Lib's CVaR minimisation (agreement
to about 1e-10), and it beats 500 random portfolios on the same data.

**What can go wrong:** it is only as good as the history in the window. A
3-year window without a crash has a mild tail, and the method will happily
hold whatever was calm recently.

---

## 17. Where the risk comes from: risk decomposition

A portfolio that looks diversified by weight can get most of its risk from
one place. Two decompositions, each adding up exactly to the total:

- **Share of volatility:** sleeve i contributes `w_i (Σw)_i / w'Σw`, its
  weight times how much it moves with the whole portfolio.
- **Share of CVaR:** on the portfolio's worst 5% of days, how much of the
  average loss came from each sleeve: `w_i × (sleeve i's average return on
  those days)`. Because the portfolio's return each day is the sum of the
  weighted sleeve returns, the pieces add up to the CVaR exactly.

The notebook shows, for each portfolio, the share of **money** in equities
next to the share of **risk** from equities. The gap is usually large,
because equities are three to four times as volatile as intermediate bonds.

---

## 18. Stress tests

**Historical replay.** Today's portfolio (the last quarter-end before the
hold-out) is held, *without rebalancing*, through five real crises:

| Scenario | Period | What it tests |
|---|---|---|
| Global financial crisis | Oct 2007 – Mar 2009 | Equity crash, flight to Treasuries |
| Euro debt crisis | 2011 | Contagion, correlation spikes |
| Taper tantrum | May – Sep 2013 | Rising rates hurting bonds |
| COVID crash | Feb – Mar 2020 | Fast, deep fall and recovery |
| 2022 inflation shock | 2022 | Stocks and bonds falling together |

Reported: total return over the episode, the worst fall inside it, and how
many months it took to get back to the starting value. The worst fall is
compared with the client's maximum acceptable loss. Separately, the
notebook shows what the walk-forward portfolios *actually* did in each
crisis their backtest covered.

**Hypothetical shocks**, applied instantly to today's weights:

- *Taiwan equities −30% with TWD weakening 10%.* A weaker TWD makes every
  USD-priced asset worth 10% more in TWD, which cushions a diversified
  client and does nothing for a 100%-Taiwan one.
- *US rates +2 percentage points.* A bond fund with duration D loses about
  D × 2%: roughly −34% for long Treasuries (duration about 17) and −15% for
  intermediate bonds (about 7.5). The durations are approximations in the
  config.
- *Gold −20%.*
- *Correlations rise to 0.8.* Not a price move but a risk change: every
  pair of risky sleeves suddenly moves together, the situation in which
  diversification fails. Reported as a 1-in-20 bad year (1.645 × annual
  volatility under a normal approximation), next to the same figure with
  normal correlations.

**What can go wrong:** history is one sample path. A replay says what
these weights would have done in *that* crisis, not in the next one. The
hypothetical shocks are instantaneous and ignore second-round effects
(a Taiwan crash would probably also hit US equities).

---

## 19. Goal probabilities: Monte Carlo

The client's real question is "will I get there?" For each hypothetical
client (5, 10 and 30 years; amounts in the config) and each portfolio
option, 10,000 futures are simulated month by month:

1. **Future monthly returns**, two ways, both reported:
   - *Block bootstrap.* The portfolio's own walk-forward monthly returns
     after costs are cut into 12-month blocks and glued together in random
     order. Real crashes and recoveries are kept, just re-ordered.
   - *Student-t.* Random monthly returns with the same mean and volatility
     as that history, but from a fat-tailed distribution (5 degrees of
     freedom), so extreme months happen more often than a bell curve
     allows. *Antithetic variates* (from the option pricer in
     `monte-carlo/Monte_Carlo.ipynb`): every random draw is also used
     mirrored, which cancels part of the simulation noise for free.
2. **Wealth:** each month the contribution goes in and the pot earns that
   month's return: `W[t+1] = (W[t] + contribution) × (1 + r)`. With a
   constant return this reproduces the annuity formula exactly (tested).
3. **Count:** the share of futures that end at or above the goal.

Also reported: the median and the 5th–95th percentile of final wealth;
the chance that the investments fall further than the client's loss limit
at some point (measured on returns, because contributions would hide
losses in the wealth figure); the bad-case time to reach the goal (by when
90% of futures have got there); and the probability with NT$2,000 more a
month or two more years. All variants reuse the same random draws
("common random numbers"), so their differences come from the change, not
from noise. The random seed is fixed in the config.

**What can go wrong**
- Both methods assume the future looks like 2008–2024 rearranged. That
  period includes a long bond bull market that cannot repeat from today's
  starting yields, so bond-heavy portfolios may look better than they will
  be.
- Amounts are nominal: inflation is not modelled, so NT$3 million in 10
  years buys less than NT$3 million today.
- The probability is only as good as the client's inputs: a missed
  contribution or an early withdrawal changes everything, and life events
  are the advisor's job, not the model's.

---

## 20. The hold-out evaluation

The last cell of notebook 03 opens the final two years of data once,
after every method and setting has been frozen. The walk-forward run
simply continues into the hold-out with the same rules, so the portfolio on
its first day is exactly the one the development run held. Only the
hold-out dates are summarised. Opening it is logged with a timestamp in
`results/holdout_log.jsonl`. If anything is changed after that, the
hold-out is spent and can no longer be called out of sample.
