import pandas as pd
import pytest

from portfolio_engine.allocation import EqualWeight, FixedWeights, benchmark_allocators, validate_weights


def test_benchmarks_from_config(cfg, data):
    allocs = benchmark_allocators(cfg)
    assert set(allocs) == {"equal_weight", "sixty_forty", "taiwan_equity_only"}
    hist = data.returns.iloc[:100]
    ew = allocs["equal_weight"](hist)
    assert ew.sum() == pytest.approx(1) and ew.nunique() == 1 and len(ew) == 7
    sf = allocs["sixty_forty"](hist)
    eq = sum(sf[s] for s, c in data.asset_class.items() if c == "equity")
    bond = sum(sf[s] for s, c in data.asset_class.items() if c == "bond")
    assert eq == pytest.approx(0.6) and bond == pytest.approx(0.4)
    tw = allocs["taiwan_equity_only"](hist)
    assert tw["taiwan_equity"] == 1 and tw.sum() == 1


def test_fixed_weights_are_normalised():
    hist = pd.DataFrame(columns=["a", "b", "c"], dtype=float)
    w = FixedWeights({"a": 2, "b": 2})(hist)
    assert w.to_dict() == {"a": 0.5, "b": 0.5, "c": 0.0}


def test_equal_weight_subset():
    hist = pd.DataFrame(columns=["a", "b", "c", "d"], dtype=float)
    w = EqualWeight(["a", "b"])(hist)
    assert w.to_dict() == {"a": 0.5, "b": 0.5, "c": 0.0, "d": 0.0}


def test_validate_weights_rejects_bad_portfolios():
    idx = pd.Index(["a", "b"])
    with pytest.raises(ValueError):
        validate_weights(pd.Series({"a": 1.2, "b": -0.2}), idx)
    with pytest.raises(ValueError):
        validate_weights(pd.Series({"a": 0.5, "b": 0.4}), idx)
    w = validate_weights(pd.Series({"a": 1.0, "b": -1e-12}), idx)
    assert w["b"] == 0
