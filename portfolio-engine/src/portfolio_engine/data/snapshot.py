"""Frozen, hash-checked copies of downloaded data.

Market data providers revise history: Yahoo re-adjusts prices for dividends
and splits, FRED revises series. If the engine re-downloaded data on every
run, the same code could give different numbers next month. So data is
downloaded once, written to Parquet files, and every file's SHA-256 hash is
recorded in ``manifest.json``. Every later run reads the frozen files and
checks the hashes first. A file that changed by even one byte is rejected.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

MANIFEST = "manifest.json"


class SnapshotIntegrityError(RuntimeError):
    """A snapshot file no longer matches the hash recorded when it was saved."""


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes: its fingerprint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_snapshot(frames: dict[str, pd.DataFrame], out_dir: str | Path,
                   meta: dict[str, Any] | None = None) -> Path:
    """Save each DataFrame as ``<name>.parquet`` and write the manifest.

    The manifest records, for each file, its hash, row count and date range,
    plus free-form ``meta`` (tickers, source, download date), so anyone can
    see exactly what a run was built on.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, df in frames.items():
        path = out / f"{name}.parquet"
        df.to_parquet(path)
        files[name] = {
            "file": path.name,
            "sha256": file_sha256(path),
            "rows": int(len(df)),
            "columns": [str(c) for c in df.columns],
            "first": str(df.index.min().date()) if len(df) else None,
            "last": str(df.index.max().date()) if len(df) else None,
        }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(meta or {}),
        "files": files,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return out


def read_snapshot(snapshot_dir: str | Path,
                  verify: bool = True) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load every file listed in the manifest, refusing any whose hash changed."""
    d = Path(snapshot_dir)
    manifest = json.loads((d / MANIFEST).read_text())
    frames = {}
    for name, info in manifest["files"].items():
        path = d / info["file"]
        if verify and file_sha256(path) != info["sha256"]:
            raise SnapshotIntegrityError(
                f"{path} does not match the hash in {MANIFEST}. The data changed after the "
                "snapshot was taken; re-download into a new snapshot instead of editing files."
            )
        frames[name] = pd.read_parquet(path)
    return frames, manifest


def snapshot_id(manifest: dict[str, Any]) -> str:
    """One short id for the whole snapshot: a hash of all its file hashes."""
    joined = "".join(f"{k}:{v['sha256']}" for k, v in sorted(manifest["files"].items()))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]
