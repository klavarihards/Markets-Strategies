# Quant Finance Projects

Trading strategies and a pricing model, built and backtested in Python.
Each strategy goes through the same process: hypothesis → data analysis →
strategy construction → stress testing (time windows, transaction costs,
parameter sensitivity) → out-of-sample validation. Results are reported as
they came out, including the ideas that did not work.

Self-directed quantitative finance work by Rihards Klavins, built during summer 2026
before starting a BSc in Intelligent Engineering and Technology at National Taiwan
University.

---

## Projects

### 1. Brent vs WTI Pairs Trading Strategy
**Folder:** `pairs-trading/`

A statistical arbitrage strategy on the spread between Brent and WTI crude oil
futures. A 20-day rolling Z-score flags unusually wide or narrow spreads; the
strategy goes long one contract and short the other and exits as the spread
returns towards its mean.

**Results (0.1% cost per trade):**
- In-sample 2021-2024: Sharpe 1.70, annualised return 12.7%, max drawdown -3.8%
  (vs about -46% for holding either contract)
- Sharpe between 1.46 and 1.88 across 2021-2024 sub-periods; positive Sharpe at
  every entry threshold from Z=1.0 to Z=3.0
- Out-of-sample January 2025 - June 2026 (never used in development):
  Sharpe 1.89, annualised return 14.4%, max drawdown -2.8%; Sharpe above 1.0
  at every entry threshold
- Fails badly in 2019-2020, when WTI went negative during the COVID-19 storage
  crisis: the strategy depends on the Brent-WTI relationship holding

**Key concepts:** Pairs trading, statistical arbitrage, Z-score, market
neutrality, out-of-sample testing

---

### 2. GARCH(1,1) Volatility Targeting — QQQ
**Folder:** `GARCH/`

Sizes a QQQ position inversely to forecast volatility: a GARCH(1,1) model is
re-fitted every week on the previous two years of returns (no look-ahead), and
leverage is set to target volatility / forecast volatility, capped at 2x, with
borrowing and trading costs.

**Results (2012-2026):**
- CAGR 27.4% vs 20.3% for buy-and-hold, but Sharpe 0.94 vs 0.99 and max
  drawdown -43% vs -35%: the extra return came mostly from leverage
- Out-of-sample 2024-2026 (target chosen on 2010-2023): Sharpe 1.02 vs 1.26
  for QQQ
- A VIX-based emergency filter made results worse (out-of-sample Sharpe 0.88)
- A 20% trailing stop improved results slightly, but was chosen on the full
  period, so this is an in-sample result

**Key concepts:** Volatility clustering, GARCH, walk-forward estimation,
volatility targeting, leverage, drawdown control

---

### 3. Z-Score Mean Reversion — Crude Oil
**Folder:** `MA_Z-Score/`

Single-asset mean reversion strategy on WTI crude oil futures. Uses a rolling
Z-score to identify when price has deviated significantly from its recent mean
and enter positions expecting reversion.

**Results (2022-2024):**
- Sharpe 0.68
- Total return 48% vs -6.7% buy and hold

**Key concepts:** Rolling statistics, Z-score, mean reversion vs trend following,
Sharpe ratio, drawdown analysis

---

### 4. Regime-Switching Strategy — SPY
**Folder:** `MA_Z-Score/` (same notebook as project 3)

Classifies the market regime (bullish, bearish or ranging) from the 50- and
200-day moving averages, then applies Z-score signals suited to each regime:
only buys dips in bull markets, only shorts rallies in bear markets, and
trades both sides in ranging markets.

**Results (2022-2024, vs SPY buy and hold):**
- Annualised return 11.7% vs 10.0%
- Sharpe 1.22 vs 0.57, max drawdown -9.2% vs -24.5%

The same notebook also tests a dip-buying strategy on 100 large S&P 500
stocks (132% vs 95% for SPY in 2020-2024 before costs) and explains why it
would not survive transaction costs.

**Key concepts:** Regime detection, golden/death cross, adaptive strategy
selection

---

### 5. Monte Carlo Option Pricer
**File:** `monte-carlo/Monte_Carlo.ipynb`

Simulates stock prices with geometric Brownian motion, prices European calls
and puts by Monte Carlo, and checks the prices against the Black-Scholes
closed-form solution, including a convergence analysis of error vs number of
simulations.

**Key concepts:** Geometric Brownian motion, risk-neutral pricing,
Monte Carlo convergence, Black-Scholes

---

---

### 6. SPY Implied Volatility Surface
**File:** `options_volatility_surface.ipynb`

Pulls SPY option chains across expiries with yfinance, converts strikes to moneyness
and plots the implied volatility surface, the volatility smile for each expiry, and
the ATM term structure with the expected move each implied volatility implies. The
CSV files in the repository root are saved outputs of this notebook and of the Monte
Carlo notebook.

**Key concepts:** Implied volatility, volatility smile and term structure, moneyness

---

### 7. Goal-Based Multi-Asset Portfolio Engine
**Folder:** `portfolio-engine/` ·
[write-up](portfolio-engine/REPORT.md) ·
[research log](portfolio-engine/RESEARCH_LOG.md) ·
[how it works](portfolio-engine/HOW_IT_WORKS.md) ·
[sample client report](portfolio-engine/reports/sample_report_chen.pdf)

A goals-based portfolio engine for a Taiwan-based investor: what should a
client hold, and how likely are they to reach their goal? Seven building
blocks in TWD (Taiwan, US and international equity, long and intermediate
Treasuries, gold, cash); six allocation methods (equal weight, mean-variance
with Ledoit-Wolf shrinkage, Black-Litterman, risk parity, hierarchical risk
parity, minimum CVaR) under conservative, balanced and growth suitability
profiles; GARCH volatility targeting, regime filters (rule-based and hidden
Markov) and momentum tilts as overlays; stress tests, risk decomposition and
Monte Carlo goal probabilities; and a client product (six-question
questionnaire, one-page PDF report, FastAPI service, Streamlit dashboard).

Everything is walk-forward with fund fees, spreads, the 0.1% Taiwan ETF sell
tax and FX costs. The last two years were locked before development and
opened once at the end; the Phase 4 decision rules were committed before the
runs they decided; every backtest (336 development trials) is in a registry
and the best results are reported with the deflated Sharpe ratio. Every
optimiser is cross-checked against PyPortfolioOpt or Riskfolio-Lib. 96 tests
run in CI.

**Results (TWD, after costs; development Jan 2008 – Oct 2024, hold-out Oct 2024 – Oct 2026):**
- No method reliably beat equal weight on risk-adjusted return: 0 of 60
  Sharpe comparisons significant in development (3 of 72 with the CVaR
  method, about what chance gives), and none for the frozen portfolios in
  the hold-out
- Downside control is where methods differed: the CVaR method was the only
  one to keep a conservative client within a -10% loss limit (worst drawdown
  -6.0%, against -11.6% to -16.6% for the others); tightening a volatility
  cap could not do it
- Overlays (regime filters, volatility targeting) cut the median balanced
  portfolio's 2008-09 drawdown from -20% to -15% but changed little after
  2010 and only added turnover in the calm hold-out; momentum tilts failed
  every test
- Hold-out, frozen portfolios: conservative 9.5% a year (worst drawdown
  -5.2%), balanced 11.8% (-6.0%, Sharpe 1.01 vs 0.99 for 1/N), growth 13.9%
  (-10.0%); every client loss limit held, but all lagged a boom in which
  100% Taiwan equity returned 42% a year
- Goal simulations: a 30-year growth client reached NT$12M in 46% of
  simulated futures with 60/40 and 63-68% with the best methods; tail-risk
  estimates differ a lot between block-bootstrap and fitted simulations

**What failed and was fixed along the way** (all in the research log): a
partial trading day in the first download, currency risk hiding in "cash",
mislabelled goal results, a deflated Sharpe on raw instead of excess
returns, a factor regression that double-counted the currency, an
unadjusted 4-for-1 split in Yahoo's 0050 data, and a product bug that made
excluding a holding raise a client's odds.

**Limitations:** long-history US-listed proxies stand in for the Taiwan
funds (0050 vs EWT: correlation 0.92); cash earns the US T-bill rate; cost,
tax and duration inputs are approximate; three real crises in the sample;
goals are nominal. Educational project with hypothetical clients, not
investment advice.

**Key concepts:** Goals-based investing, suitability, walk-forward
backtesting, covariance shrinkage, Black-Litterman, risk parity, HRP, CVaR
optimisation, GARCH, hidden Markov regimes, Monte Carlo simulation, stress
testing, hold-out testing, pre-registration, deflated Sharpe ratio

---

## Stack
- **Python** — pandas, numpy, matplotlib, scipy, arch, statsmodels, cvxpy, hmmlearn, pytest
- **Product** (portfolio engine) — FastAPI, Streamlit, Jinja2, Docker, GitHub Actions
- **Data** — yfinance (market data), investing.com (commodity futures)
- **Environment** — Google Colab
- **AI assistance** — Claude (Anthropic) for development, debugging, and research

---

## Methodology
The strategies follow the same validation pipeline:

1. **Hypothesis** — identify a market phenomenon with economic logic
2. **Data exploration** — understand the asset's statistical properties
3. **Strategy construction** — build signal, position, and return logic
4. **Initial backtest** — evaluate on in-sample data
5. **Stress testing** — multiple time windows + transaction costs
6. **Sensitivity analysis** — test across a range of parameters
7. **Out-of-sample validation** — evaluate on data not used in development
