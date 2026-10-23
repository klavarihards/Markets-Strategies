"""Market data: download (Colab only), frozen snapshots, synthetic test data, and the loader."""

from .loader import MarketData, holdout_start, load_market_data, to_base_currency
from .snapshot import (
                       SnapshotIntegrityError,
                       find_snapshot,
                       read_snapshot,
                       resolve_snapshot,
                       snapshot_id,
                       write_snapshot,
)
from .synthetic import make_synthetic_frames

__all__ = [
    "MarketData", "holdout_start", "load_market_data", "to_base_currency",
    "SnapshotIntegrityError", "find_snapshot", "read_snapshot", "resolve_snapshot", "snapshot_id", "write_snapshot",
    "make_synthetic_frames",
]
