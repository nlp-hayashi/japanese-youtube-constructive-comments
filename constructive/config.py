"""
Path configuration for reproducibility scripts.

Set environment variables before running:
  DATA_DIR    — JSON/CSV inputs (default: <repo>/data)
  OUTPUT_DIR  — checkpoints, scored CSVs, metrics (default: <repo>/outputs)
"""

from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    raw = os.environ.get("DATA_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return get_repo_root() / "data"


def get_output_dir() -> Path:
    raw = os.environ.get("OUTPUT_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return get_repo_root() / "outputs"
