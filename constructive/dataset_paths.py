"""Resolve dataset JSON paths (prefer *_with_text.json after restore_comment_text.py)."""

from __future__ import annotations

from pathlib import Path

from constructive.config import get_data_dir


def dataset_json(basename: str) -> Path:
    """
    basename e.g. ``dataset_train.json``.
    If ``dataset_train_with_text.json`` exists in DATA_DIR, use it for training.
    """
    data = get_data_dir()
    stem = basename[:-5] if basename.endswith(".json") else basename
    with_text = data / f"{stem}_with_text.json"
    plain = data / basename
    if with_text.is_file():
        return with_text
    return plain
