from portfolio_engine.config import config_hash, load_config

from conftest import ROOT


def test_phase2_inherits_phase1():
    p1 = load_config(ROOT / "configs" / "phase1_baseline.yaml")
    p2 = load_config(ROOT / "configs" / "phase2_methods.yaml")
    assert p2["name"] == "phase2_methods"
    assert p2["costs"] == p1["costs"] and p2["sleeves"] == p1["sleeves"]
    assert "extends" not in p2 and set(p2["profiles"]) == {"conservative", "balanced", "growth"}
    assert config_hash(p1) != config_hash(p2)


def test_profiles_are_internally_consistent():
    p2 = load_config(ROOT / "configs" / "phase2_methods.yaml")
    for name, p in p2["profiles"].items():
        lows = p["equity"][0] + p["bonds_cash"][0] + p["gold"][0]
        highs = p["equity"][1] + p["bonds_cash"][1] + p["gold"][1]
        assert lows <= 1 <= highs, name
        assert p["max_loss"] < 0
    assert abs(sum(p2["methods"]["black_litterman"]["market_weights"].values()) - 1) < 1e-12
