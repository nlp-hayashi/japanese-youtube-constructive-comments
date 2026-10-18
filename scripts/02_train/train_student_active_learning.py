#!/usr/bin/env python3
"""Active learning training: merge AL gold into train, optional pseudo-labels."""

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


def _resolve_al_json(name_or_path: str) -> str:
    p = Path(name_or_path)
    if p.is_file():
        return str(p.resolve())
    basename = name_or_path if name_or_path.endswith(".json") else f"dataset_active_learning_{name_or_path}.json"
    return str(dataset_json(basename))


def main() -> None:
    root = get_output_dir()
    p = argparse.ArgumentParser(description="Train after an active-learning annotation round.")
    p.add_argument(
        "--run-dir",
        default="student2_active",
        help="Checkpoints under OUTPUT_DIR (contains best/)",
    )
    p.add_argument(
        "--init-checkpoint",
        default="student_self_training/best",
        help="Warm-start checkpoint (relative to OUTPUT_DIR ok)",
    )
    p.add_argument(
        "--al-json",
        action="append",
        dest="al_jsons",
        help="AL gold JSON (basename under DATA_DIR, or path). Repeat for multiple rounds.",
    )
    p.add_argument(
        "--pseudo-glob",
        default="pseudo_scores/student1/*.csv",
        help="Scored CSV glob (relative to OUTPUT_DIR ok). Ignored if --no-pseudo.",
    )
    p.add_argument("--no-pseudo", action="store_true", help="Gold train + AL labels only")
    p.add_argument("--report-json", default=None)
    args = p.parse_args()

    if args.al_jsons:
        al_paths = [_resolve_al_json(x) for x in args.al_jsons]
    else:
        al_paths = [str(dataset_json("dataset_active_learning_al_round1.json"))]

    run_dir = _out_path(args.run_dir)
    report = _out_path(args.report_json) if args.report_json else run_dir / "train_report.json"
    pseudo = None if args.no_pseudo else (
        args.pseudo_glob if Path(args.pseudo_glob).is_absolute() else str(root / args.pseudo_glob)
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train_student(
        str(dataset_json("dataset_train.json")),
        str(dataset_json("dataset_valid.json")),
        str(dataset_json("dataset_test.json")),
        str(_out_path(args.init_checkpoint)),
        str(run_dir),
        hp=StudentHyperparams(),
        pseudo_glob=pseudo,
        al_json_paths=al_paths,
        report_json=str(report),
    )


if __name__ == "__main__":
    main()
