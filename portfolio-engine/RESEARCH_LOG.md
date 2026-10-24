# Research Log

A dated record of every iteration: what was built, what the run showed,
what went wrong, and what was changed because of it. Results are quoted
only from saved runs (`results/<config>_<hash>_<snapshot>_<tag>/`, every one
listed in `experiments/registry.csv`). The hold-out period (23 Oct 2024 –
22 Oct 2026) is not used in any entry until it is opened, once, and that
entry says so.

Data snapshot for all entries: `570bdef9e8ea` (downloaded 2026-10-23,
committed in `data/snapshots/2026-10-23/`).

---

## Iteration 1: Phase 1 build (2026-10-23)

**Built:** data download and frozen snapshots (SHA-256 manifest), TWD
returns for seven sleeves, the hold-out lock, a quarterly walk-forward
backtest with fees, spreads, commission, the 0.1% Taiwan sell tax and FX
spread, metrics, and three benchmarks. 33 tests on synthetic data,
including "the allocator never sees the future" and "a development run is
an exact prefix of the hold-out run".

**Problems found and fixed**
- *First Colab run failed* (`FileNotFoundError`): the notebook expected a
  snapshot that does not exist on a fresh runtime. Fix: `latest` resolves
  to the newest snapshot, the notebook downloads automatically when none
  exists, and in Colab snapshots live on Google Drive.
- *Partial-day prices.* The first real download ran during US trading
  hours and froze a half-finished day. Fix: the downloader ends on the
  previous completed session. The snapshot was re-taken before any result
  was looked at.

## Iteration 2: Phase 1 results, first run (config `3e6527ffbfbb`)

3-year window, 31 Dec 2007 – 22 Oct 2024, TWD, after costs: equal weight
5.4% a year, Sharpe 0.51, max drawdown -21.9%; 60/40 5.9%, 0.47, -28.9%;
100% Taiwan equity 8.2%, 0.41, -58.1%. Cost drag about 0.4% a year.

**Noticed:** the cash sleeve, a US T-bill converted into TWD, had 4.9%
volatility and a **-18.8% worst drawdown** in TWD. A TWD deposit never
does that; the conversion was giving "cash" currency risk.

## Iteration 3: Phase 2 build

**Built:** Ledoit-Wolf shrinkage, Black-Litterman, the three client
profiles, one cvxpy solver for all methods, five methods, compliance and
exposure tables, block-bootstrap intervals for Sharpe differences.

**Cross-checks (tests):** mean-variance vs PyPortfolioOpt ~1e-8, Black-
Litterman ~1e-6, HRP identical, risk parity vs Riskfolio-Lib ~5e-6,
Ledoit-Wolf vs scikit-learn exact.

**Problems found and fixed**
- *Ledoit-Wolf differed from PyPortfolioOpt by 0.15%.* Traced to
  PyPortfolioOpt using a 1/(T-1) sample covariance where the paper uses
  1/T; with the same convention the two agree exactly. Kept the paper's
  version and documented the gap in the test.
- *PyPortfolioOpt's HRP crashes on scipy 1.18* (it uses a removed private
  name). The test restores the name before calling the reference.
- *The "nearest portfolio" solve was only accurate to 4e-5.* Minimising a
  squared distance makes the objective tiny near the answer and the solver
  stops early. Minimising the distance itself gives the same answer to
  ~1e-8.
- *A test failure was committed* because `pytest | tail` hid pytest's exit
  code. Caught on the next run; all later checks use `set -o pipefail`.

## Iteration 4: Phase 2 results, first run (config `ff5ff6328895`)

None of 60 Sharpe comparisons against 1/N and 60/40 was significant.
Mean-variance led with the 5-year window but traded 2-6x more; risk parity
and HRP had the smallest drawdowns for conservative and balanced clients
with the 3-year window; every method broke the conservative client's -10%
loss limit. On synthetic and real data, the conservative profile hit its
7% volatility cap almost always, partly because of the currency risk in
"cash".

## Iteration 5: cash treated as a TWD deposit

**Decision (with the project owner, before Phase 3):** cash becomes a TWD
deposit (`currency: TWD`): no currency risk, but still the US T-bill rate,
since no long daily TWD rate series is freely available, so cash returns
are somewhat overstated. Rationale: a conservative client's cash should
not carry 5% currency volatility. Phases 1 and 2 were re-run (configs
`eb292432cba0`, `03f45b1dbe9e`).

**What changed:** conservative mean-variance Sharpe fell from 0.58 to 0.50
(3-year window); the volatility cap no longer had to be relaxed for
conservative clients (15% of quarters before, 0% after); the conclusions
did not change (still 0 of 60 comparisons significant, Sharpe gaps -0.08
to +0.14).

## Iteration 6: Phase 3 build and results (config `847159c84c0e`)

**Built:** a min-CVaR method (Rockafellar-Uryasev linear programme;
matches Riskfolio-Lib to ~1e-10 and beats 500 random portfolios),
volatility and CVaR decomposition (contributions add up exactly), five
historical crisis replays, four hypothetical shocks, and a Monte Carlo
goal simulator (block bootstrap and Student-t with antithetic variates;
matches the annuity formula on deterministic paths).

**Bug found:** in the goals table, the equal-weight *benchmark* and the
profile's equal-weight *method* were both labelled `equal_weight`, so two
different portfolios shared a row label and the fan-chart data overwrote
each other. Fixed (full run names, a uniqueness test) and re-run.

**Results:** see the Phase 3 section of `README.md`. Headlines: the CVaR
variant was the only method to keep a conservative client within -10%
(worst drawdown -6.0%); a 30-year growth client reaches NT$12M in 46% of
futures with 60/40 and 63-68% with the best methods; the balanced client's
odds rise from 83% to 94% with NT$2,000 more a month.

## Review after Phase 3 (mentor's notes)

1. **The conservative profile does not add up.** A 7% volatility cap means
   a 1-in-20 bad year of about -11.5%, beyond the client's -10% limit, so
   most methods were set up to fail it. *Action (Phase 4):* test a
   calibrated profile (volatility cap 6%, so 1.645 x 6% ≈ -9.9%) as a
   sensitivity, chosen from the arithmetic, not from backtest results.
2. **Multiple testing.** 3 of 72 Sharpe comparisons were "significant",
   about what chance gives. *Action:* a trial registry and the deflated
   Sharpe ratio (Bailey and López de Prado), so the number of things tried
   is counted, not forgotten.
3. **Tail estimates depend on the simulation.** The bootstrap and the
   Student-t give very different chances of a deep fall. *Action:* report
   both; the client report will quote the more cautious one.
4. **Hypothetical shocks are one-dimensional.** A Taiwan crash with a weaker
   TWD *gains* money for diversified portfolios here, because US assets are
   not shocked at the same time. *Action:* state it in the limitations;
   the correlation scenario covers the "everything falls together" case.
5. **Goal simulations reuse 2008-2024**, a period with a long bond rally
   that cannot repeat from today's yields. *Action:* limitation stated.

**Trials so far:** 126 development backtests in the registry (including
the superseded first runs; the Phase 3 config re-runs the Phase 2 methods
under a new hash, so distinct strategies are fewer).

---

## Pre-registration for Phase 4 and the final choice (written before any Phase 4 run)

These rules are fixed *before* the overlays and robustness runs, and
committed, so the choices that follow cannot be fitted to the results.

**Overlays tested** (each on every profile x method, 3-year window,
monthly overlay checks, all costs):
`core` (Phase 3 portfolio), `+vol_target` (GARCH), `+regime_rule`,
`+regime_hmm`, `+tilt` (Black-Litterman momentum views, capped at ±5 points
from the core), and `+all` (tilt switched off in the stressed regime,
regime filter with the *rule* model, the simpler of the two, then volatility
targeting).

**Rule 1: keep an overlay for a profile only if**, for the median of that
profile's six methods, compared with the core portfolio, (a) the maximum
drawdown or the monthly 95% CVaR improves by at least 10% (relative), and
(b) the Sharpe ratio falls by no more than 0.05, all after costs. Otherwise
the overlay is reported and dropped.

**Rule 2: the method recommended for each profile** is chosen among the
methods (with kept overlays) whose 3-year-window Sharpe ratio is within 0.05
of the best for that profile: first those that stayed within the client's
maximum loss in the walk-forward backtest; among them the lowest monthly
CVaR; if none stayed within the limit, the smallest maximum drawdown. Ties
go to the simpler method (fewer estimated inputs).

**Robustness settings** (reported, not used for selection): rebalancing
monthly / quarterly / annually; risk aversion 1, 2.5, 5, 10 (mean-variance
and Black-Litterman); single-sleeve cap 30%, 40%, 50%; and a calibrated
conservative profile with a 6% volatility cap (1.645 x 6% ≈ 9.9%, inside
the -10% limit), chosen from that arithmetic alone.

**The hold-out** is opened once, after Rules 1 and 2 have produced the
frozen configuration, for every method, benchmark and the frozen choice.

---

## Iteration 7: Phase 4 build and first run (config `7a10877139aa`)

**Built:** an engine hook for monthly overlay checks (trade only if the
target moves > 2 points), GARCH(1,1)-t volatility targeting (reused from
the QQQ project), rule-based and HMM regime detectors (with the Baa-Treasury
credit spread, in its own macro snapshot `19d9b758d92e`), Black-Litterman
momentum tilts capped at ±5 points, the two pre-registered rules as code,
robustness sweeps, a calm/stressed breakdown, factor regressions and the
deflated Sharpe ratio. Regimes: the rule flags 17% of months as stressed,
the HMM 23%; they agree 87% of the time; both flag all of 2008.

**Rule 1 (median method, change vs core, 3-year window):**

| Overlay | Conservative | Balanced | Growth | Kept for |
|---|---|---|---|---|
| Volatility target | +3% risk, -0.01 Sharpe | +21% risk, +0.04 | +36% risk, +0.08 | balanced, growth |
| Regime (rule) | +16% risk, +0.05 | +25%, +0.05 | +16%, +0.04 | all |
| Regime (HMM) | +14%, +0.02 | +25%, +0.04 | +16%, +0.03 | all |
| Momentum tilt | +2%, -0.01 | -2%, -0.02 | +1%, -0.01 | none |
| All three | +12%, +0.03 | +29%, +0.07 | +43%, +0.11 | all |

("risk" = the larger of the relative improvements in max drawdown and
monthly CVaR.) The momentum tilt **failed everywhere**: it added 21-29
points of turnover a year and did not reduce risk.

**Weakness in my own rule:** "all" includes the failed tilt, but passed as a
package. The rule was applied as written; the benefit of "all" almost
certainly comes from its regime and volatility parts. A better rule would
have tested packages only from components that passed on their own.

**Rule 2 (frozen in `configs/final_choice.yaml`):** conservative =
min-CVaR + all overlays (Sharpe 0.69, worst drawdown -7.4%); balanced =
min-CVaR + HMM regime filter (0.60, -12.6%); growth = Black-Litterman + all
overlays (0.60, -18.2%). All three stayed within their client's loss limit
on development data. For the conservative profile only one portfolio was
within 0.05 Sharpe of the best, so that choice is close to "pick the best of
36", exactly the situation the deflated Sharpe ratio is for.

**Robustness (balanced, 11 settings):** min-CVaR had the best Sharpe in 9 of
11 (range 0.52-0.57); mean-variance was the most fragile (0.40-0.58,
depending on rebalancing frequency and the single-sleeve cap).

**Calibrated conservative profile (6% volatility cap):** every method except
min-CVaR still broke the -10% limit (-10.5% to -14.5%); min-CVaR -6.0%. A
volatility cap is the wrong tool for a drawdown limit; a tail-risk
objective is the right one.

**Calm vs stressed (rule regime):** in stressed periods the conservative
risk parity, HRP and min-CVaR portfolios kept Sharpe ratios of 0.57-0.72;
balanced mean-variance fell to -0.09; every growth portfolio was near 0.

**Mentor's check: where does the overlay benefit come from?** Almost
entirely from 2008-09. For balanced clients the median method's worst
drawdown in 2008-09 improves from -20.0% (core) to -15.1% (regime filter);
from 2010 onwards worst drawdowns are almost unchanged (-13.1% vs -11.7% to
-12.6%) and so are Sharpe ratios. The evidence for the overlays rests on one
crisis. The hold-out is the first test on new data.

## Iteration 8: two bugs found in review, Phase 4 re-run (same config)

1. **The deflated Sharpe ratio used raw returns** while every other Sharpe in
   the engine uses returns in excess of cash, so the frozen conservative
   portfolio showed 0.94 in the DSR table and 0.69 in the summary. Fix: each
   run now saves the cash rate (`rf.parquet`), the registry stores excess-
   return Sharpe ratios, and the registry was rebuilt for every earlier run.
2. **The factor regression double-counted the currency.** The equity, rates
   and gold factors were measured in TWD, so they already contained the
   USD/TWD move, and every portfolio showed a negative dollar exposure.
   Fix: asset factors in USD, USD/TWD on its own. Dollar betas are now
   0.23-0.71 for the profile portfolios and -1.07 for 100% Taiwan equity
   (Taiwan stocks fall in TWD when the dollar strengthens), which makes sense.

Also caught: an automatic lint fix removed an import that a newly added
function needed; the test for the hold-out evaluation failed and caught it.

The frozen choice from the re-run is identical to the first run (as it must
be: neither fix touches Rules 1 or 2).

**Deflated Sharpe ratios** (223 distinct development trials; expected best
Sharpe from noise alone 0.22): conservative choice 0.97, balanced 0.94,
growth 0.94, equal weight 0.89, 60/40 0.84. Only the conservative choice
clears the usual 0.95 bar; the trials are highly correlated, so this errs
on the strict side.

**Factor exposures:** R² 0.73-0.94 and alphas close to zero: the portfolios
are mostly bundles of equity, duration and dollar exposure, which is what a
strategic allocation should be.

---

## Iteration 9: the hold-out, opened once (2026-10-23 16:57 UTC)

`configs/final_choice.yaml` was committed (commit `e386dd0`) before the
opening; the opening is logged in `experiments/holdout_log.jsonl`; results
in `notebooks/04b_holdout_evaluation.ipynb` and the registry (tag
`holdout`). Hold-out: 23 Oct 2024 – 22 Oct 2026, TWD, after costs. It was a
strong bull market: 100% Taiwan equity returned 42.3% a year.

| | Annual return | Volatility | Sharpe | Worst drawdown | Client limit |
|---|---|---|---|---|---|
| Conservative: min-CVaR + all overlays | 9.5% | 6.3% | 0.85 | -5.2% | -10% (held) |
| Balanced: min-CVaR + HMM regime | 11.8% | 7.4% | 1.01 | -6.0% | -20% (held) |
| Growth: Black-Litterman + all overlays | 13.9% | 12.7% | 0.78 | -10.0% | -30% (held) |
| Equal weight | 15.2% | 10.9% | 0.99 | -7.1% | |
| 60/40 | 15.6% | 12.8% | 0.89 | -10.4% | |
| 100% Taiwan equity | 42.3% | 27.6% | 1.27 | -23.5% | |

**What held up:** every frozen portfolio stayed inside its client's loss
limit and its profile's volatility band, with worst drawdowns of -5.2% to
-10.0%.

**What did not:**
- **The cautious portfolios lagged a boom**, as their lower equity share
  implies. The balanced choice matched equal weight on Sharpe (1.01 vs
  0.99) with less risk; the growth choice trailed it (0.78 vs 0.99).
- **No overlay passed Rule 1 on the hold-out.** There was no stressed period
  for them to protect against, so they mostly added turnover: the growth
  choice traded 64% of the portfolio a year, with a cost drag of 1.0% a year
  (development: 0.8%). Consistent with the finding that their benefit came
  from 2008.
- **Mean-variance had the best hold-out Sharpe in every profile** (1.15-1.26),
  the method that looked most fragile in development. Black-Litterman without
  views had the worst (0.63-0.69). In a two-year trending market, historical
  means pointed the right way.
- **No Sharpe difference is significant.** Every 90% interval for the frozen
  portfolios against 1/N and 60/40 contains zero (widths of ±0.4 to ±0.7):
  two years cannot separate these methods.

**Conclusion of the research (Phases 1-4):** no allocation method reliably
beat equal weight on risk-adjusted return, in development or out of sample.
What the engine *does* deliver is control of downside risk tailored to the
client: tail-risk (CVaR) construction kept conservative and balanced clients
inside their loss limits in every period tested, which no volatility-based
method managed for conservative clients. Overlays helped in one crisis and
cost money otherwise. Nothing was changed after this entry.

---

## Iteration 10: Phase 5, the product, and what driving it revealed

**Built:** a six-question questionnaire (profile = the more cautious of
risk capacity and risk tolerance), an advisor that runs each profile's
frozen portfolio on all data (allowed only because the hold-out is logged as
evaluated), a one-page PDF report (Jinja2 + xhtml2pdf, every figure from the
engine), a FastAPI service, a Streamlit dashboard, Docker files and GitHub
Actions CI (ruff + 96 tests). Results in `notebooks/05_product_demo.ipynb`
and `reports/`.

**Found by using the product rather than testing it**
- *Charts missing and "nan%" in the first PDF*: xhtml2pdf ignores percentage
  image widths, and the report quoted a crisis outside the data. Fixed
  (explicit sizes; only crises fully inside the data are quoted).
- *An unadjusted 4-for-1 split in Yahoo's 0050.TW history* (2014-01-02, a
  -75% "day") made 0050 look like it fell 77%. Added `data/quality.py`:
  split-like moves are repaired, other moves beyond Taiwan's price limit are
  flagged (seven in Jan-Feb 2009, the first weeks of the series). After the
  repair, 0050 vs EWT in TWD: correlation 0.92, tracking error 7.7% a year.
- *Excluding a holding raised a client's odds* (47% to 56% for a
  conservative client avoiding gold), because exclusions switched the goal
  simulation to a different basis. Fixed: every client is simulated on
  today's weights applied to market history minus the measured cost drag;
  regression test added.

**Docker:** no Docker daemon in the development sandbox, so the image was
not built here. The recipe was verified instead: a clean virtualenv with a
non-editable `pip install ".[product]"` and the image's file layout served
`/health` and returned a 68 KB PDF from `/report`.

## Final review (mentor's notes)

**What makes this credible:** walk-forward only, with tests that scramble
the future; frozen and hashed data; a hold-out locked first and opened once
(logged, committed); decision rules committed before the runs they decided;
every trial counted and the Sharpe ratio deflated; every optimiser checked
against an established library; failures and bugs recorded here, not hidden.

**What a sceptical reviewer would still push on**
1. Three real crises in 22 years is a small sample for tail-risk claims.
2. Proxies, not the Taiwan-listed funds, drive the research; cash earns the
   US T-bill rate; cost, tax and duration inputs are approximate.
3. The CVaR method's edge is in drawdowns, not Sharpe, and rests mainly on
   2008, 2020 and 2022.
4. The overlays were kept on the strength of one crisis and did nothing
   useful in the hold-out; a production version would need a longer history
   or a cost-aware rule before using them.
5. The deflated Sharpe ratio treats 223 correlated trials as independent.

**Next steps if continued:** Taiwan-listed fund data and real TWD deposit
rates; more crises (longer proxy histories or other markets); spread and
tax data from a broker; inflation-adjusted goals; threshold rebalancing
(drift > 5 points) and contribution-first rebalancing, which the spec
describes but this backtest does not model; a paper-trading log.

## Iteration 11: Taiwan reality check, dated views, and what a reviewer asked for (2026-10-24)

A review of the project as it appears on `main` raised six points. Two of
them (Phase 3 missing, no CI) came from the fact that pull request 1 was
merged at the Phase 2 stage (`081f4ac`), so `main` showed none of Phases
3-5; the rest were real gaps.

**Taiwan cash.** The CBC Statistical Database has an API that serves the
five major banks' monthly deposit rates back to 1987 (FRED has no Taiwan
rate series). Getting it to download took four fixes: the server resets
connections without a browser User-Agent, it needs retries, and its
certificate lacks a Subject Key Identifier, which Python 3.13's strict
verification rejects (only that strictness flag is cleared; the
certificate is still verified). Snapshot `005d4d80deef` (funds and CBC
tables) is committed in `data/taiwan/`.

*Bug found on the first run:* the TWD deposit rate was unchanged for
months at a time, so within an estimation window cash had zero variance,
and the constant-correlation Ledoit-Wolf target divided by zero, sending NaN
into the solver. Fixed by estimating on the sleeves that move and giving
the riskless one zero (co)variance; a test covers it.

**Result.** Re-running every method, the frozen choices and the benchmarks
with Taiwan cash (not re-choosing anything) changed levels, not
conclusions: Sharpe ratios rose because the TWD rate is lower; min-CVaR is
still the only method inside the conservative -10% limit (-6.1%); hold-out
returns of cash-heavy portfolios fell (conservative 9.5% to 8.5% a year).
In the product, short-horizon clients' goal chances fell from 78% to 73%
and 76% to 70%. The product now uses Taiwan cash.

**Real funds.** From July 2017 (when all six exist), monthly correlations
with the proxies are 0.87-0.96 except 00645 (0.68, Japan only). 0050 beat
EWT by 4.4% a year; 00635U lagged spot gold by 5.3% a year. Replaying the
frozen portfolios' trades with the funds (next Taipei day, Taiwan costs)
gave 1.0-1.4% a year more than the proxies, with similar drawdowns. A test
checks that the replay reproduces the backtest when given the proxies.

**Dated Black-Litterman views.** Built and tested (`estimators/views.py`,
`configs/views.yaml`): each view has a date, horizon and confidence, is
used only while active, gets its uncertainty from Idzorek's method
(matches PyPortfolioOpt), and is scored when it expires. The file is
empty: my macro journal records what happened and what is expected, not
dated, quantified forecasts, and writing views now for past dates would be
hindsight. Views will be entered on the day they are written, and the git
history proves when.

**Also:** a "Key findings" section with one chart (drawn by
`scripts/key_findings_chart.py` from the run files) at the top of the
README; the CI badge; findings cells of notebooks 05 and 06 written from
their outputs.
