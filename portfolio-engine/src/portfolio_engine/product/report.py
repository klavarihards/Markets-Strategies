"""The one-page client report: HTML from a Jinja2 template, then PDF.

Every number on the page comes from the advice dictionary produced by the
engine; the template only formats it ("nothing in the report is typed by
hand"). Charts are drawn with matplotlib and embedded as images. The PDF is
rendered with xhtml2pdf, which is pure Python, so it also works in Colab and
in the Docker image without system libraries.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES = Path(__file__).with_name("templates")
METHOD_TEXT = {
    "equal_weight": "equal weighting (the same amount in each holding, adjusted to your profile)",
    "mean_variance": "mean-variance optimisation (the best return for the risk, from history)",
    "black_litterman": "Black-Litterman (market-implied returns, adjusted to your profile)",
    "risk_parity": "risk parity (each holding contributes the same amount of risk)",
    "hrp": "hierarchical risk parity (similar holdings grouped, less money to riskier groups)",
    "min_cvar": "minimum CVaR (the mix whose worst days have historically been least bad, within your limits)",
}
OVERLAY_TEXT = {"core": "", "vol_target": ", with volatility targeting", "regime_rule": ", with a market-stress filter",
                "regime_hmm": ", with a market-stress filter", "tilt": ", with small momentum tilts",
                "all": ", with momentum tilts, a market-stress filter and volatility targeting"}
CRISIS_TEXT = {"global_financial_crisis": "a crisis like 2008", "inflation_shock_2022": "a year like 2022, when stocks and bonds fell together",
               "covid_crash": "a crash like March 2020", "euro_debt_crisis": "a year like 2011", "taper_tantrum": "a bond sell-off like 2013"}


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def weights_chart(advice: dict[str, Any]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = sorted(advice["portfolio"]["holdings"], key=lambda x: x["weight"])
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    ax.barh([x["name"] for x in h], [100 * x["weight"] for x in h], color="#2f6f9f")
    for i, x in enumerate(h):
        ax.text(100 * x["weight"] + 0.8, i, f"{100 * x['weight']:.0f}%", va="center", fontsize=7)
    ax.set_xlim(0, max(100 * x["weight"] for x in h) * 1.25)
    ax.tick_params(labelsize=7); ax.set_xlabel("% of your money", fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    out = _png(fig); plt.close(fig)
    return out


def fan_chart(advice: dict[str, Any]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = advice["goal"]
    q = np.array(g["fan"]["values"]) / 1e6
    years = np.arange(q.shape[1]) / 12
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    ax.fill_between(years, q[0], q[4], color="#2f6f9f", alpha=0.15, label="good to bad case (90%)")
    ax.fill_between(years, q[1], q[3], color="#2f6f9f", alpha=0.3, label="middle half")
    ax.plot(years, q[2], color="#0b3a5b", linewidth=1.5, label="typical")
    ax.axhline(g["amount"] / 1e6, color="black", linestyle="--", linewidth=1, label="your goal")
    ax.set_xlabel("years from now", fontsize=7); ax.set_ylabel("NT$ million", fontsize=7)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6, loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    out = _png(fig); plt.close(fig)
    return out


def crisis_sentences(advice: dict[str, Any]) -> list[str]:
    out = []
    for k, c in advice["risk"]["crises"].items():
        rec = (f"and taken about {c['months_to_recover']:.0f} months to recover" if c["months_to_recover"] is not None
               else "and had not fully recovered by the end of the data")
        out.append(f"In {CRISIS_TEXT.get(k, k)}, this portfolio might have fallen about "
                   f"{-100 * c['max_drawdown']:.0f}% {rec}.")
    return out


def render_html(advice: dict[str, Any], client_name: str = "Hypothetical client") -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    p = advice["portfolio"]
    return env.get_template("client_report.html").render(
        client_name=client_name, weights_png=weights_chart(advice), fan_png=fan_chart(advice),
        in_ten=int(round(10 * advice["goal"]["probability"])), crises=crisis_sentences(advice),
        method_text=METHOD_TEXT[p["method"]] + OVERLAY_TEXT.get(p["overlay"], ""), **advice)


def render_pdf(advice: dict[str, Any], client_name: str = "Hypothetical client") -> bytes:
    from xhtml2pdf import pisa
    buf = io.BytesIO()
    result = pisa.CreatePDF(render_html(advice, client_name), dest=buf)
    if result.err:
        raise RuntimeError("PDF rendering failed")
    return buf.getvalue()
