#!/usr/bin/env python3
"""Student self-training: gold + pseudo-labels from scored CSVs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from constructive.config import get_output_dir
from constructive.dataset_paths import dataset_json
from constructive.training import StudentHyperparams, train_student


def _out_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else get_output_dir() / p


def main() -> None:
    root = get_output_dir()
    p = argparse.ArgumentParser(description="Student self-training (gold + pseudo).")
    p.add_argument(
        "--run-dir",
        default="student_self_training",
        help="Checkpoints under OUTPUT_DIR (contains best/)",
    )
    p.add_argument(
        "--init-checkpoint",
        default="teacher_simple_best/best",
        help="Warm-start model directory (relative to OUTPUT_DIR ok)",
    )
    p.add_argument(
        "--pseudo-glob",
        default="pseudo_scores/teacher/*.csv",
        help="Glob for teacher-scored CSVs (relative to OUTPUT_DIR ok)",
    )
    p.add_argument("--report-json", default=None, help="Default: <run-dir>/train_report.json")
    args = p.parse_args()

    run_dir = _out_path(args.run_dir)
    report = _out_path(args.report_json) if args.report_json else run_dir / "train_report.json"

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train_student(
        str(dataset_json("dataset_train.json")),
        str(dataset_json("dataset_valid.json")),
        str(dataset_json("dataset_test.json")),
        str(_out_path(args.init_checkpoint)),
        str(run_dir),
        hp=StudentHyperparams(),
        pseudo_glob=str(root / args.pseudo_glob) if not Path(args.pseudo_glob).is_absolute() else args.pseudo_glob,
        report_json=str(report),
    )


if __name__ == "__main__":
    main()
