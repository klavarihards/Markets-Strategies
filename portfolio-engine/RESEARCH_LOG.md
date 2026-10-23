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
