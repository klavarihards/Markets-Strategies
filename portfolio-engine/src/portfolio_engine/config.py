"""Run configuration: one YAML file per experiment.

Every number the engine produces should be traceable to the exact settings
that made it. So each run is driven by a YAML file, and the run log stores a
hash of that file's contents: if two results carry the same config hash and
the same data-snapshot hash, they must be identical.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML run configuration into a plain dictionary."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_source"] = str(path)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Short fingerprint of a configuration.

    The idea: serialise the settings in a fixed order (sorted keys) and hash
    them, so the same settings always give the same 12-character id no matter
    how the YAML was formatted. Keys starting with ``_`` (such as the file
    path) are bookkeeping, not settings, and are left out.
    """
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    text = json.dumps(clean, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def override(cfg: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """Copy of ``cfg`` with dotted keys replaced, e.g. ``override(cfg, **{"backtest.window_years": 5})``.

    Used for robustness runs: the original config is never mutated, so a
    sweep cannot silently leak a setting into the next run.
    """
    new = copy.deepcopy(cfg)
    for dotted, value in changes.items():
        node = new
        *parents, last = dotted.split(".")
        for p in parents:
            node = node[p]
        node[last] = value
    return new
