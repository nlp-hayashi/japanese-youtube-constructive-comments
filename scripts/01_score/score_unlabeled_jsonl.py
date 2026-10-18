#!/usr/bin/env python3
"""Score an unlabeled comment corpus (JSONL) with a trained classifier.

Writes one CSV per input file under the output directory. Training scripts
(`train_student_self_training.py`, etc.) read these CSVs; they do not run
inference on the unlabeled pool themselves.

Expected JSONL fields (override with CLI): comment, videoID, comment_id.
Output columns include videoID, comment_id, comment, p_constructive.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from constructive.config import get_output_dir

MAX_LEN = 512
POS_INDEX = 1


def _sanitize(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()


def _jsonl_iter(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path.name}:{lineno} {e}", file=sys.stderr)


def _row_from_obj(
    obj: dict[str, Any],
    *,
    text_field: str,
    video_field: str,
    comment_id_field: str,
) -> dict[str, str]:
    return {
        "videoID": _sanitize(obj.get(video_field, "")),
        "comment_id": _sanitize(obj.get(comment_id_field, "")),
        "comment": _sanitize(obj.get(text_field, "")),
    }


@torch.no_grad()
def _predict_p_pos(
    model,
    tokenizer,
    device: str,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, POS_INDEX]
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float64)


def _score_one_file(
    in_path: Path,
    out_path: Path,
    model,
    tokenizer,
    device: str,
    batch_size: int,
    text_field: str,
    video_field: str,
    comment_id_field: str,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["videoID", "comment_id", "comment", "p_constructive"]
    written = 0
    texts: list[str] = []
    rows: list[dict[str, str]] = []

    with out_path.open("w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        def flush() -> None:
            nonlocal written
            if not texts:
                return
            probs = _predict_p_pos(model, tokenizer, device, texts, batch_size)
            for row, p in zip(rows, probs):
                row["p_constructive"] = f"{float(p):.6f}"
                writer.writerow(row)
                written += 1
            texts.clear()
            rows.clear()

        for obj in _jsonl_iter(in_path):
            row = _row_from_obj(
                obj,
                text_field=text_field,
                video_field=video_field,
                comment_id_field=comment_id_field,
            )
            if not row["comment"]:
                continue
            texts.append(row["comment"])
            rows.append(row)
            if len(texts) >= batch_size:
                flush()
        flush()

    return written


def main(argv: list[str] | None = None) -> None:
    out_default = get_output_dir() / "pseudo_scores" / "teacher"
    parser = argparse.ArgumentParser(
        description="Score JSONL comment files for pseudo-label training.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of *.jsonl files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=out_default,
        help=f"Directory for *.scored.csv (default: {out_default})",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=get_output_dir() / "teacher_simple_best" / "best",
        help="HF sequence-classification checkpoint (e.g. Teacher best/)",
    )
    parser.add_argument("--glob", default="*.jsonl", help="Input file pattern")
    parser.add_argument("--recursive", action="store_true", help="Search input-dir recursively")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-field", default="comment")
    parser.add_argument("--video-field", default="videoID")
    parser.add_argument("--comment-id-field", default="comment_id")
    args = parser.parse_args(argv)

    if not args.model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_dir}")
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    files = sorted(args.input_dir.rglob(args.glob) if args.recursive else args.input_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matching {args.glob!r} under {args.input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {args.model_dir}")
    print(f"Files:  {len(files)} -> {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir)).to(device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for in_path in tqdm(files, desc="jsonl", unit="file"):
        out_path = args.output_dir / f"{in_path.stem}.scored.csv"
        n = _score_one_file(
            in_path,
            out_path,
            model,
            tokenizer,
            device,
            args.batch_size,
            args.text_field,
            args.video_field,
            args.comment_id_field,
        )
        print(f"  {out_path.name}: {n:,} rows")


if __name__ == "__main__":
    main()
