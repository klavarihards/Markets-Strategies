"""Build docs/project_brief.pdf, the two-page project brief, from the run files.

Every number in the tables and most in the text is read from the committed
results of the Taiwan reality check (notebook 06); the few that come from
earlier runs are quoted: the Sharpe-test count (REPORT.md) and one client's goal
chances (notebooks/05_product_demo.ipynb outputs).

Run from the project folder: ``python scripts/project_brief.py``.
"""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RUN = next(Path("results").glob("taiwan_reality_check_*_holdout"))
OUT = Path("docs/project_brief.pdf")
REPO = "https://github.com/klavarihards/Markets-Strategies"
PROJECT = f"{REPO}/tree/main/portfolio-engine"
COLAB = ("https://colab.research.google.com/github/klavarihards/Markets-Strategies/blob/main/"
         "portfolio-engine/notebooks/")

FONT_DIR = Path("/usr/share/fonts/truetype/crosextra")
for name, file in (("Body", "Carlito-Regular.ttf"), ("Body-Bold", "Carlito-Bold.ttf"),
                   ("Body-Italic", "Carlito-Italic.ttf"), ("Body-BoldItalic", "Carlito-BoldItalic.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                              boldItalic="Body-BoldItalic")

INK, MUTED, RULE, BLUE, TINT = (colors.HexColor(c) for c in ("#1f1f1e", "#5f5f5b", "#d9d9d4", "#2a78d6", "#eef4fc"))
S = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=19, leading=22, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="Body", fontSize=10, leading=13, textColor=MUTED),
    "h": ParagraphStyle("h", fontName="Body-Bold", fontSize=11.5, leading=14, textColor=BLUE,
                        spaceBefore=7, spaceAfter=2.5),
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=12.6, textColor=INK, alignment=TA_LEFT),
    "lead": ParagraphStyle("lead", fontName="Body", fontSize=10.6, leading=14, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=9.6, leading=12.6, textColor=INK,
                             leftIndent=9, bulletIndent=0, spaceAfter=1.5),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.8, leading=10.8, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=8.8, leading=10.8, textColor=INK),
    "note": ParagraphStyle("note", fontName="Body-Italic", fontSize=8.2, leading=10.4, textColor=MUTED),
}


def pct(x: float, d: int = 1, sign: bool = False) -> str:
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header_rows=1, bold_rows=(), shade_rows=()):
    data = [[c if isinstance(c, Paragraph) else Paragraph(str(c), S["cellb" if (i < header_rows or i in bold_rows)
                                                                     else "cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LINEBELOW", (0, header_rows), (-1, -1), 0.3, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header_rows:
        style.append(("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, INK))
    style += [("BACKGROUND", (0, r), (-1, r), TINT) for r in shade_rows]
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 10 * mm,
                      "Educational research project with hypothetical clients; not investment advice. Fund names are "
                      "used as representative Taiwan-listed instruments.")
    canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"{doc.page} / 2")
    canvas.restoreState()


def main() -> None:
    dev = pd.read_csv(RUN / "cash_dev.csv")
    hold = pd.read_csv(RUN / "cash_holdout.csv")
    drift = pd.read_csv(RUN / "fund_drift.csv", index_col=0)
    impl = pd.read_csv(RUN / "fund_implementation.csv", index_col=0)

    def row(df, profile, method):
        return df[(df.profile == profile) & (df.method == method)].iloc[0]

    frozen = {"conservative": "min_cvar|all", "balanced": "min_cvar|regime_hmm", "growth": "black_litterman|all"}
    methods = {"equal_weight": "Equal weight (1/N)", "mean_variance": "Mean-variance (Ledoit-Wolf)",
               "black_litterman": "Black-Litterman", "risk_parity": "Risk parity",
               "hrp": "Hierarchical risk parity", "min_cvar": "Minimum CVaR"}
    period = f"{dev.start.iloc[0][:7]} to {dev.end.iloc[0][:7]}"
    h_period = f"{hold.start.iloc[0][:7]} to {hold.end.iloc[0][:7]}"

    cons_dd = {m: row(dev, "conservative", m).max_drawdown for m in methods}
    others = [v for m, v in cons_dd.items() if m != "min_cvar"]
    tw_hold = row(hold, "benchmark", "taiwan_equity_only")
    fc = {p: row(hold, p, m) for p, m in frozen.items()}

    story = [
        Paragraph("Goal-Based Portfolios for Taiwan Investors", S["title"]),
        Spacer(1, 2),
        Paragraph("Project brief · Rihards Klavins · October 2026 · "
                  f'<a href="{PROJECT}" color="#2a78d6">github.com/klavarihards/Markets-Strategies</a>', S["sub"]),
        Spacer(1, 7),
        Paragraph(
            "<b>The question.</b> What should a Taiwan retail client hold to reach a goal, such as a home deposit "
            "in ten years or retirement in thirty, without ever losing more than they said they could bear? "
            "The project is built for a wealth advisor: it turns a six-question client questionnaire into a "
            "suitable portfolio of Taiwan-listed funds, the chance of reaching the goal, what a bad year would "
            "cost in NTD, and a one-page client report.", S["lead"]),
        Paragraph("What it uses", S["h"]),
    ]
    story += bullets([
        "<b>Seven building blocks, all in TWD:</b> Taiwan, US and international equity; long and intermediate "
        "US Treasuries; gold; cash as a TWD deposit at the Central Bank's five-major-banks rate. Researched on "
        "long-history proxies (EWT, SPY, EFA+EEM, TLT, IEF, GLD, 2004-2026), then checked with the Taiwan-listed "
        "funds a client would actually buy.",
        "<b>Three client profiles</b> (conservative, balanced, growth) with equity, bond and gold limits, a 40% "
        "cap per holding and a maximum loss the client accepts (-10%, -20%, -30%).",
        "<b>Six allocation methods:</b> equal weight, mean-variance with Ledoit-Wolf shrinkage, Black-Litterman "
        "(with dated, confidence-weighted research views scored afterwards), risk parity, hierarchical risk "
        "parity, and a minimum-CVaR portfolio that minimises the average loss on the worst days. "
        "<b>Three overlays:</b> GARCH volatility targeting, regime filters (rule-based and hidden Markov), "
        "momentum tilts.",
        "<b>Walk-forward tests with real frictions:</b> quarterly rebalancing using only past data; fund fees, "
        "bid-ask spreads, commission, the <b>0.1% Taiwan ETF sell tax</b>, US dividend withholding and FX "
        "spreads. Stress replays of 2008, 2020 and 2022; Monte Carlo goal probabilities.",
        "<b>Guards against fooling myself:</b> the last two years were <b>locked away before any research and "
        "opened once</b>, at the end; decision rules were committed before the runs they decided; every one of "
        "336 backtests is logged and the best are judged with the deflated Sharpe ratio; every optimiser is "
        "cross-checked against PyPortfolioOpt or Riskfolio-Lib; 104 automated tests run on every push.",
    ])
    story += [
        Paragraph("Key result: the methods differ in how much a client can lose, not in Sharpe ratio", S["h"]),
        Image("docs/key_findings.png", width=150 * mm, height=150 * mm * 4.6 / 9),
        Paragraph(
            f"For a conservative client, only the minimum-CVaR portfolio stayed inside the -10% loss limit "
            f"(worst drawdown {pct(cons_dd['min_cvar'])}, against {pct(max(others))} to {pct(min(others))} for "
            f"the other methods), with the same or better Sharpe ratio. None of the methods reliably beat equal "
            f"weight on risk-adjusted return: 0 of 60 Sharpe comparisons were significant in development. "
            f"Walk-forward, after costs, {period}.", S["body"]),
        Paragraph("For the client and the advisor", S["h"]),
    ]
    story += bullets([
        "<b>A personalised plan:</b> six questions set the profile (the more cautious of what the client can "
        "afford to lose and what they say they would do after a 20% fall); the report gives the chance of "
        "reaching the goal and what changes it. For one hypothetical client, NTD 2,000 more a month raises "
        "the chance from 73% to 94%.",
        "<b>After-sales service built in:</b> the advisor is alerted at 75% of the client's loss limit, "
        "before it is breached; the rebalancing schedule is the one that was backtested; research views are "
        "dated and scored afterwards, so the house view builds a track record.",
        "<b>Local:</b> Taiwan-listed funds, Taiwan deposit rates for cash, the Taiwan sell tax, all results "
        "in NTD.",
    ])
    story += [
        PageBreak(),
        Paragraph("Results by method (development period, after costs, TWD)", S["h"]),
    ]

    rows = [["Method", "Conservative<br/>Sharpe", "Conservative<br/>worst drawdown", "Balanced<br/>Sharpe",
             "Balanced<br/>worst drawdown"]]
    for m, label in methods.items():
        c, b = row(dev, "conservative", m), row(dev, "balanced", m)
        rows.append([label, f"{c.sharpe:.2f}", pct(c.max_drawdown), f"{b.sharpe:.2f}", pct(b.max_drawdown)])
    c, b = row(dev, "conservative", frozen["conservative"]), row(dev, "balanced", frozen["balanced"])
    rows.append(["Frozen choice (min-CVaR + overlays)", f"{c.sharpe:.2f}", pct(c.max_drawdown), f"{b.sharpe:.2f}",
                 pct(b.max_drawdown)])
    story += [
        table(rows, [62 * mm, 26 * mm, 30 * mm, 26 * mm, 30 * mm], bold_rows=(6, 7), shade_rows=(6, 7)),
        Spacer(1, 3),
        Paragraph(
            f"{period}; Sharpe ratio against the TWD deposit rate; client loss limits -10% (conservative) and "
            f"-20% (balanced). <b>Locked hold-out ({h_period}), opened once:</b> the frozen portfolios returned "
            f"{pct(fc['conservative'].ann_return)}, {pct(fc['balanced'].ann_return)} and "
            f"{pct(fc['growth'].ann_return)} a year (conservative, balanced, growth), with worst drawdowns of "
            f"{pct(fc['conservative'].max_drawdown)}, {pct(fc['balanced'].max_drawdown)} and "
            f"{pct(fc['growth'].max_drawdown)}: every client limit held.", S["note"]),
        Paragraph("Taiwan-fund check: the same portfolios held with Yuanta's ETFs", S["h"]),
        Paragraph(
            f"The research used US-listed proxies because they have 20+ years of history. To test whether the "
            f"results survive in the products a Taiwan client would actually hold, each proxy was compared "
            f"with its Taiwan-listed fund, monthly in TWD from {impl.start.iloc[0][:7]}, when all of them "
            f"exist.", S["body"]),
        Spacer(1, 3),
    ]
    funds = [("taiwan_equity", "0050", "Yuanta Taiwan Top 50", "EWT"),
             ("us_equity", "00646", "Yuanta S&amp;P 500", "SPY"),
             ("long_treasury", "00679B", "Yuanta US Treasury 20+ Year", "TLT"),
             ("gold", "00635U", "Yuanta S&amp;P GSCI Gold ER (futures)", "GLD")]
    rows = [["Fund", "Proxy", "Correlation", "Tracking error", "Return gap a year<br/>(fund − proxy)"]]
    for sleeve, code, name, proxy in funds:
        d = drift.loc[sleeve]
        rows.append([f"<b>{code}</b> {name}", proxy, f"{d.correlation:.2f}", pct(d.tracking_error),
                     pct(d.return_gap, sign=True)])
    story.append(table(rows, [70 * mm, 16 * mm, 24 * mm, 26 * mm, 38 * mm]))
    gaps = [impl.loc[f"{p}:{m}"] for p, m in frozen.items()]
    gap_lo, gap_hi = min(g.return_gap for g in gaps), max(g.return_gap for g in gaps)
    te_lo, te_hi = min(g.tracking_error for g in gaps), max(g.tracking_error for g in gaps)
    cons_cash = row(hold, "conservative", frozen["conservative"])
    fr_mid = sum(g.gap_fund_returns for g in gaps) / len(gaps)
    cost_lo, cost_hi = min(g.gap_cost_assumptions for g in gaps), max(g.gap_cost_assumptions for g in gaps)
    story += [Spacer(1, 4)] + bullets([
        f"<b>The frozen portfolios, replayed trade by trade with the Taiwan-listed funds</b> (executed the next "
        f"Taipei trading day, with spreads, commission and the 0.1% sell tax), returned "
        f"{pct(gap_lo, sign=True)} to {pct(gap_hi, sign=True)} a year against the proxies, with tracking error "
        f"of {pct(te_lo)} to {pct(te_hi)} and similar worst drawdowns. About {pct(fr_mid)} a year is the "
        f"funds' own returns (0050 ahead of EWT, 00635U behind gold); {pct(cost_lo)} to {pct(cost_hi)} is "
        f"holding costs the research charged that fund prices already contain. The research was, if "
        f"anything, conservative for a client holding these funds.",
        f"<b>The gold fund needs explaining to clients:</b> 00635U tracks a gold futures excess-return index "
        f"and lagged spot gold by {pct(-drift.loc['gold'].return_gap)} a year (roll costs, fees and no interest "
        f"on collateral), a cost a client holding it for diversification should know about.",
        f"<b>Cash at Taiwan rates</b> (the CBC deposit rate instead of US T-bills) left the conclusions intact "
        f"but lowered cash-heavy returns: the conservative choice earned {pct(cons_cash.ann_return)} a year in "
        f"the hold-out, and two short-horizon clients' goal chances fell from 78% to 73% and from 76% to 70%.",
    ])
    story.append(Paragraph("What failed, and the limitations", S["h"]))
    story += bullets([
        "<b>Momentum tilts failed every test</b> (more trading, no risk benefit). <b>Overlays helped in one "
        "crisis</b>: they cut the median balanced portfolio's 2008-09 drawdown from -20% to -15%, but changed "
        "little afterwards and only added costs in the calm hold-out.",
        "<b>My own pre-registered rule had a flaw:</b> it let an overlay package pass although it contained the "
        "failed momentum tilt. Packages should only be built from parts that pass on their own.",
        f"<b>Out of sample the portfolios protected clients but lagged a boom</b> in which 100% Taiwan equity "
        f"returned {pct(tw_hold.ann_return)} a year. Bugs found in review (an unadjusted 0050 split, a "
        f"mislabelled goal result, a Sharpe test on the wrong returns) were fixed and logged before any decision "
        f"used them.",
        "<b>Limitations:</b> one 22-year history with three real crises; the Taiwan funds give nine years; "
        "costs and taxes are approximations; goals are in nominal NTD; simulated goal chances are estimates, "
        "not guarantees.",
    ])
    story.append(KeepTogether([
        Paragraph("Links", S["h"]),
        table([["Code and documentation", f'<a href="{PROJECT}" color="#2a78d6">{PROJECT.removeprefix("https://")}</a>'],
               ["Taiwan-fund check (Colab)", f'<a href="{COLAB}06_taiwan_reality_check.ipynb" color="#2a78d6">'
                                             "notebooks/06_taiwan_reality_check.ipynb</a>"],
               ["Methods comparison (Colab)", f'<a href="{COLAB}02_allocation_methods.ipynb" color="#2a78d6">'
                                              "notebooks/02_allocation_methods.ipynb</a>"],
               ["Hold-out evaluation (Colab)", f'<a href="{COLAB}04b_holdout_evaluation.ipynb" color="#2a78d6">'
                                               "notebooks/04b_holdout_evaluation.ipynb</a>"],
               ["Full write-up · sample client report",
                f'<a href="{PROJECT}/REPORT.md" color="#2a78d6">REPORT.md</a> · '
                f'<a href="{PROJECT}/reports/sample_report_chen.pdf" color="#2a78d6">sample_report_chen.pdf</a>']],
              [52 * mm, 122 * mm], header_rows=0),
    ]))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                            bottomMargin=17 * mm, title="Goal-Based Portfolios for Taiwan Investors",
                            author="Rihards Klavins", subject="Project brief")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Saved", OUT)


if __name__ == "__main__":
    main()
