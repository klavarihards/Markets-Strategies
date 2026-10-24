"""The six-question client questionnaire and how it sets the risk profile.

Know-your-client in two parts, as suitability rules usually require:

- **Risk capacity**: how much loss the client can *afford*. A short horizon
  or an unstable income lowers it: there is no time or income to recover.
- **Risk tolerance**: how much loss the client can *stand*. The best single
  question is what they would do after a 20% fall: sell, hold, or buy more.

The profile is the **lower** of the two. A client who loves risk but needs
the money in three years is conservative; a client with 30 years and a
stable income who would sell in a panic is not put in growth, because the
selling is what destroys the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

PROFILES = ["conservative", "balanced", "growth"]


@dataclass
class Answers:
    """A client's answers. Amounts in TWD."""

    goal_amount: float
    goal_year: int
    current_savings: float
    monthly_contribution: float
    income_stability: str          # "unstable" | "stable" | "very_stable"
    reaction_to_20pct_fall: str    # "sell" | "hold" | "buy_more"
    avoid: list[str] = field(default_factory=list)   # sleeves the client does not want to hold

    def horizon_years(self, today: date | None = None) -> float:
        today = today or date.today()
        return max(self.goal_year - today.year - (today.month - 1) / 12, 0.0)


def assess_profile(a: Answers, today: date | None = None) -> dict:
    """Risk capacity, risk tolerance and the resulting profile, with the reasons."""
    h = a.horizon_years(today)
    horizon_score = 0 if h < 5 else (1 if h < 10 else 2)
    income_score = {"unstable": 0, "stable": 1, "very_stable": 2}[a.income_stability]
    capacity_points = horizon_score + income_score
    capacity = "conservative" if capacity_points <= 1 else ("balanced" if capacity_points <= 3 else "growth")
    tolerance = {"sell": "conservative", "hold": "balanced", "buy_more": "growth"}[a.reaction_to_20pct_fall]
    profile = PROFILES[min(PROFILES.index(capacity), PROFILES.index(tolerance))]
    if h < 3:
        profile = "conservative"
    return {"profile": profile, "risk_capacity": capacity, "risk_tolerance": tolerance,
            "horizon_years": round(h, 1),
            "reason": (f"Horizon {h:.0f} years and {a.income_stability.replace('_', ' ')} income give a "
                       f"{capacity} capacity for risk; selling/holding/buying after a 20% fall "
                       f"('{a.reaction_to_20pct_fall.replace('_', ' ')}') indicates a {tolerance} tolerance. "
                       f"The profile is the more cautious of the two: {profile}.")}
