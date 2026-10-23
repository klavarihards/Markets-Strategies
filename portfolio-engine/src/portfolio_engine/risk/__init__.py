"""Risk and goal analysis: stress tests, risk decomposition and Monte Carlo goal probabilities."""

from .analysis import current_portfolios, run_risk_analysis, tables_for_saving
from .decomposition import cvar_contributions, decompose, volatility_contributions
from .goals import Client, bootstrap_paths, goal_report, simulate_goal, student_t_paths, wealth_paths
from .stress import bad_year_loss, correlation_stress_cov, historical_stress, hypothetical_stress, replay, shock_returns

__all__ = [
    "current_portfolios", "run_risk_analysis", "tables_for_saving", "cvar_contributions", "decompose",
    "volatility_contributions", "Client", "bootstrap_paths", "goal_report", "simulate_goal", "student_t_paths",
    "wealth_paths", "bad_year_loss", "correlation_stress_cov", "historical_stress", "hypothetical_stress", "replay",
    "shock_returns",
]
