import pandas as pd

from portfolio_engine.data.quality import implausible_moves, repair_splits


def test_unadjusted_split_is_repaired():
    idx = pd.bdate_range("2014-01-01", periods=6)
    raw = pd.Series([40.0, 40.4, 40.0, 10.1, 10.2, 10.0], index=idx)     # 4-for-1 split on day 4
    fixed, repairs = repair_splits(raw)
    assert len(repairs) == 1 and repairs[0]["split"] == "4-for-1"
    assert fixed.pct_change().dropna().abs().max() < 0.03                   # no more -75% day
    assert fixed.iloc[-1] == raw.iloc[-1]                                   # recent prices untouched


def test_real_moves_are_not_repaired_but_flagged():
    idx = pd.bdate_range("2020-01-01", periods=4)
    s = pd.Series([100.0, 80.0, 81.0, 95.0], index=idx)                   # -20%: not a split ratio
    fixed, repairs = repair_splits(s)
    assert repairs == [] and fixed.equals(s)
    assert list(implausible_moves(s, 0.10).round(2)) == [-0.2, 0.17]
