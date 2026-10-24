"""Streamlit dashboard: answer six questions, see the plan, download the one-page report.

Run from the portfolio-engine folder:  streamlit run app/streamlit_app.py
Educational research project with hypothetical clients, not investment advice.
"""

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# Work from the portfolio-engine folder whatever directory the app is launched from
# (e.g. the repository root on a hosting service), and find the package without installing it.
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from portfolio_engine.product import Advisor, Answers  # noqa: E402  (after the path set-up above)
from portfolio_engine.product.report import fan_chart, render_pdf, weights_chart  # noqa: E402

st.set_page_config(page_title="Goal-based portfolio engine", layout="wide")


@st.cache_resource(show_spinner="Building the portfolios (once per start-up)...")
def advisor() -> Advisor:
    return Advisor("configs/product.yaml")


st.title("What should I hold, and how likely am I to reach my goal?")
st.caption("Educational research project with hypothetical clients. Not investment advice.")

with st.sidebar:
    st.header("Your answers")
    goal = st.number_input("Goal amount (NT$)", 100_000, 100_000_000, 3_000_000, step=100_000)
    year = st.number_input("Goal year", date.today().year + 1, date.today().year + 50, date.today().year + 10)
    savings = st.number_input("Current savings (NT$)", 0, 100_000_000, 600_000, step=50_000)
    monthly = st.number_input("Monthly contribution (NT$)", 0, 1_000_000, 15_000, step=1_000)
    income = st.selectbox("How stable is your income?", ["unstable", "stable", "very_stable"], index=1)
    reaction = st.selectbox("If your investments fell 20%, you would...", ["sell", "hold", "buy_more"], index=1)
    labels = {"taiwan_equity": "Taiwan stocks", "us_equity": "US stocks", "intl_equity": "Other markets' stocks",
              "long_treasury": "Long US Treasuries", "intermediate_bond": "Medium-term bonds", "gold": "Gold"}
    avoid = st.multiselect("Anything you must avoid?", list(labels), format_func=labels.get)
    name = st.text_input("Name on the report", "Hypothetical client")

answers = Answers(goal, int(year), savings, monthly, income, reaction, avoid)
try:
    adv = advisor().advise(answers)
except (ValueError, RuntimeError) as e:
    st.error(f"No suitable portfolio: {e}")
    st.stop()

g = adv["goal"]
c1, c2, c3 = st.columns(3)
c1.metric("Your profile", adv["profile"]["profile"].capitalize())
c2.metric("Chance of reaching the goal", f"{g['probability']:.0%}")
c3.metric("Typical outcome", f"NT${g['median']:,.0f}")
st.caption(adv["profile"]["reason"])

left, right = st.columns(2)
with left:
    st.subheader("Your portfolio")
    st.image(f"data:image/png;base64,{weights_chart(adv)}")
    st.dataframe(pd.DataFrame(adv["portfolio"]["holdings"])[["name", "weight", "fund", "why"]]
                 .style.format({"weight": "{:.0%}"}), hide_index=True)
with right:
    st.subheader("Simulated futures")
    st.image(f"data:image/png;base64,{fan_chart(adv)}")
    st.write(f"With NT${g['with_more_monthly']['extra']:,} more a month: **{g['with_more_monthly']['probability']:.0%}**. "
             f"Aiming for {g['year'] + g['two_years_later']['years']}: **{g['two_years_later']['probability']:.0%}**.")
    for k, c in adv["risk"]["crises"].items():
        st.write(f"{k.replace('_', ' ').capitalize()}: worst fall {c['max_drawdown']:.0%}.")

st.download_button("Download the one-page report (PDF)", render_pdf(adv, name), "plan.pdf", "application/pdf")
st.caption(adv["limits"])
