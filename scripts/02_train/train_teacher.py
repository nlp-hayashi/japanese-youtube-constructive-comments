#!/usr/bin/env python3
"""Train the Teacher on gold HF splits (after text restore)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from constructive.config import get_output_dir
from constructive.dataset_paths import dataset_json
from constructive.training import TeacherHyperparams, train_teacher


def _out_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else get_output_dir() / p


def main() -> None:
    root = get_output_dir()
    p = argparse.ArgumentParser(description="Train Teacher on gold splits.")
    p.add_argument(
        "--run-dir",
        default="teacher_simple_best",
        help="Directory under OUTPUT_DIR for checkpoints (contains best/)",
    )
    p.add_argument("--metrics-json", default=None, help="Metrics JSON path (relative to OUTPUT_DIR ok)")
    p.add_argument("--pred-valid-csv", default=None)
    p.add_argument("--pred-test-csv", default=None)
    args = p.parse_args()

    run_dir = _out_path(args.run_dir)
    if args.metrics_json:
        metrics = _out_path(args.metrics_json)
    elif args.run_dir == "teacher_simple_best":
        metrics = root / "teacher_simple_metrics.json"
    else:
        metrics = run_dir / "metrics.json"

    if args.pred_valid_csv:
        pred_valid = _out_path(args.pred_valid_csv)
    elif args.run_dir == "teacher_simple_best":
        pred_valid = root / "teacher_simple_valid_predictions.csv"
    else:
        pred_valid = run_dir / "valid_predictions.csv"

    if args.pred_test_csv:
        pred_test = _out_path(args.pred_test_csv)
    elif args.run_dir == "teacher_simple_best":
        pred_test = root / "teacher_simple_test_predictions.csv"
    else:
        pred_test = run_dir / "test_predictions.csv"

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train_teacher(
        str(dataset_json("dataset_train.json")),
        str(dataset_json("dataset_valid.json")),
        str(dataset_json("dataset_test.json")),
        str(run_dir),
        hp=TeacherHyperparams(),
        metrics_json=str(metrics),
        pred_valid_csv=str(pred_valid),
        pred_test_csv=str(pred_test),
    )


if __name__ == "__main__":
    main()
