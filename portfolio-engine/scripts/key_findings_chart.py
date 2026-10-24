"""Draw docs/key_findings.png from the Taiwan reality check results (no numbers typed in).

Conservative profile, development period (Dec 2007 - Oct 2024), cash as a TWD
deposit: worst drawdown of every method against the client's -10% loss limit,
with each method's Sharpe ratio as the label.

Run from the project folder: ``python scripts/key_findings_chart.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUN = next(Path("results").glob("taiwan_reality_check_*_holdout"))
LABELS = {"equal_weight": "Equal weight", "mean_variance": "Mean-variance", "black_litterman": "Black-Litterman",
          "risk_parity": "Risk parity", "hrp": "Hierarchical risk parity", "min_cvar": "Min-CVaR",
          "min_cvar|all": "Min-CVaR + overlays (frozen choice)"}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6b68", "#e4e4e0", "#fcfcfb"
BLUE, GRAY, ORANGE = "#2a78d6", "#b9b9b4", "#eb6834"

df = pd.read_csv(RUN / "cash_dev.csv")
df = df[(df.profile == "conservative") & df.method.isin(LABELS)].set_index("method").loc[list(LABELS)]
dd = df.max_drawdown * 100

fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
y = range(len(df))[::-1]
colors = [BLUE if m.startswith("min_cvar") else GRAY for m in df.index]
ax.barh(list(y), dd, color=colors, height=0.6, zorder=2)
ax.axvline(-10, color=ORANGE, lw=2, ls=(0, (4, 3)), zorder=3)
ax.text(-10.3, -0.75, "client's loss limit -10%", color=INK, fontsize=9, ha="right", va="center")
for yi, (m, row) in zip(y, df.iterrows()):
    ax.text(row.max_drawdown * 100 - 0.3, yi, f"{row.max_drawdown:.1%}  (Sharpe {row.sharpe:.2f})",
            va="center", ha="right", fontsize=9, color=INK, zorder=4,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
ax.set_yticks(list(y), [LABELS[m] for m in df.index], fontsize=9.5, color=INK)
ax.set_xlim(dd.min() - 9, 0)
ax.set_ylim(-1.2, len(df) - 0.5)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_title("Conservative client: similar Sharpe ratios, very different worst losses",
             loc="left", fontsize=12, color=INK, pad=22)
ax.text(0, 1.02, f"Worst drawdown, walk-forward after costs, TWD, cash as a Taiwan deposit, "
        f"{df.start.iloc[0][:7]} to {df.end.iloc[0][:7]}", transform=ax.transAxes, fontsize=8.5, color=MUTED)
fig.tight_layout()
out = Path("docs/key_findings.png")
fig.savefig(out, facecolor=SURFACE)
print("Saved", out)
