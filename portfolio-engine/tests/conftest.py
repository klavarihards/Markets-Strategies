"""Shared fixtures: the real Phase 1 config, fed with synthetic data."""

from pathlib import Path

import pytest

from portfolio_engine.config import load_config
from portfolio_engine.data import load_market_data, make_synthetic_frames

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "phase1_baseline.yaml")


@pytest.fixture(scope="session")
def frames():
    return make_synthetic_frames(start="2004-11-18", end="2016-12-31", seed=7)


@pytest.fixture(scope="session")
def data(cfg, frames):
    return load_market_data(cfg, frames=frames)
