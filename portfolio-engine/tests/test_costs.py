import numpy as np
import pytest

from portfolio_engine.backtest import CostModel


def _model():
    cfg = {"a": {"fund_fee": 0.006, "proxy_fee": 0.001, "withholding": 0.002,
                 "half_spread": 0.001, "commission": 0.0005, "sell_tax": 0.001, "fx_spread": 0.0005},
           "cash": {}}
    return CostModel.build(cfg, ["a", "cash"])


def test_holding_cost_charges_only_the_extra_fee():
    m = _model()
    # (0.6% - 0.1%) fee + 0.2% withholding = 0.7% a year on the 'a' sleeve
    assert m.holding[0] == pytest.approx(0.007)
    assert m.daily_holding_cost(np.array([0.5, 0.5])) == pytest.approx(0.5 * 0.007 / 252)


def test_sell_tax_only_on_sales():
    m = _model()
    buy = m.trading_cost(np.array([0.0, 1.0]), np.array([0.4, 0.6]))
    sell = m.trading_cost(np.array([0.4, 0.6]), np.array([0.0, 1.0]))
    assert buy == pytest.approx(0.4 * 0.002)               # spread + commission + fx
    assert sell == pytest.approx(0.4 * (0.002 + 0.001))    # ... plus the 0.1% tax
    assert m.trading_cost(np.array([0.4, 0.6]), np.array([0.4, 0.6])) == 0


def test_config_costs_cover_every_sleeve(cfg):
    m = CostModel.from_config(cfg, list(cfg["sleeves"]))
    assert len(m.buy) == 7
    assert (m.sell >= m.buy).all()
    assert m.buy[m.sleeves.index("cash")] == 0
