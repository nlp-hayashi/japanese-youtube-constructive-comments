# Export test-set probabilities for evaluation (inference only; not training).
# CLI: python scripts/01_score/export_test_probabilities.py

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from constructive.config import get_output_dir
from constructive.dataset_paths import dataset_json

_OUT = get_output_dir()
STUDENT3_DIR = str(_OUT / "student3_active" / "best")
TEST_JSON = str(dataset_json("dataset_test.json"))
OUT_CSV = str(_OUT / "predictions" / "student3_test_with_prob.csv")
BATCH_SIZE = 32
DEVICE = None  # None = auto (cuda if available)

MAX_LEN = 512


def logits_to_probs(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    pr = np.exp(logits)
    pr = pr / pr.sum(axis=1, keepdims=True)
    return pr[:, 0], pr[:, 1]


def _sanitize_tokenizer_config_dict(tc: dict) -> dict:
    out = dict(tc)
    out.pop("tokenizer_class", None)
    out.pop("fast_tokenizer_class", None)
    am = out.get("auto_map")
    if isinstance(am, dict):
        cleaned = {k: v for k, v in am.items() if "TokenizersBackend" not in str(v)}
        if cleaned:
            out["auto_map"] = cleaned
        else:
            out.pop("auto_map", None)
    elif am is not None and "TokenizersBackend" in str(am):
        out.pop("auto_map", None)
    return out


def _copy_tokenizer_sidecars(src_dir: str, dst_dir: str) -> None:
    for name in os.listdir(src_dir):
        sp = os.path.join(src_dir, name)
        if not os.path.isfile(sp):
            continue
        low = name.lower()
        sz = os.path.getsize(sp)
        if low.endswith(".safetensors"):
            continue
        if low.endswith(".bin") and sz > 2 * 1024 * 1024 and "token" not in low:
            continue
        if sz > 200 * 1024 * 1024:
            continue
        shutil.copy2(sp, os.path.join(dst_dir, name))


def _load_tokenizer_from_staging(model_dir: str):
    tc_path = os.path.join(model_dir, "tokenizer_config.json")
    with open(tc_path, encoding="utf-8") as f:
        tc = _sanitize_tokenizer_config_dict(json.load(f))
    tmp = tempfile.mkdtemp(prefix="export_student3_tok_")
    try:
        _copy_tokenizer_sidecars(model_dir, tmp)
        with open(os.path.join(tmp, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(tc, f, indent=2, ensure_ascii=False)
        last_err = None
        for use_fast in (True, False):
            try:
                return AutoTokenizer.from_pretrained(tmp, use_fast=use_fast, local_files_only=True)
            except (ValueError, OSError) as e:
                last_err = e
        raise last_err
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def load_tokenizer(model_dir: str):
    last_err = None
    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(model_dir, use_fast=use_fast)
        except (ValueError, OSError) as e:
            last_err = e
    err_txt = str(last_err) if last_err else ""
    if "TokenizersBackend" in err_txt or "does not exist or is not currently imported" in err_txt:
        return _load_tokenizer_from_staging(model_dir)
    raise last_err


def check_model_dir(path: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Model directory not found: {path}")
    weight_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
    ]
    if not any(os.path.isfile(os.path.join(path, w)) for w in weight_files):
        raise FileNotFoundError(
            f"No model weights in {path}\n"
            "Need model.safetensors or pytorch_model.bin in the checkpoint folder."
        )


@torch.inference_mode()
def predict_probs(
    model_dir: str,
    texts: list[str],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    check_model_dir(model_dir)
    tok = load_tokenizer(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    model.to(device)
    model.eval()

    p0_list, p1_list = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.cpu().numpy()
        p0, p1 = logits_to_probs(logits)
        p0_list.append(p0)
        p1_list.append(p1)
    return np.concatenate(p0_list), np.concatenate(p1_list)


def run_export(
    model_dir: str = STUDENT3_DIR,
    test_json: str = TEST_JSON,
    out_csv: str = OUT_CSV,
    batch_size: int = BATCH_SIZE,
    device: str | None = DEVICE,
) -> dict:
    """Call from notebooks without argparse."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isfile(test_json):
        raise FileNotFoundError(f"Missing test set: {test_json}")

    with open(test_json, encoding="utf-8") as f:
        items = json.load(f)

    texts = [str(x.get("text") or "") for x in items]
    labels = np.array([int(x["label"]) for x in items], dtype=int)

    print(f"Model:  {model_dir}")
    print(f"Test:   {test_json} ({len(items)} samples)")
    print(f"Device: {device}")

    p_no, p_yes = predict_probs(model_dir, texts, device, batch_size)

    fieldnames = [
        "videoID",
        "comment_id",
        "gold_label",
        "comment",
        "p_yes",
        "p_no",
        "model",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(items):
            w.writerow(
                {
                    "videoID": row.get("videoID", ""),
                    "comment_id": row.get("comment_id", ""),
                    "gold_label": int(labels[i]),
                    "comment": texts[i],
                    "p_yes": float(p_yes[i]),
                    "p_no": float(p_no[i]),
                    "model": os.path.basename(os.path.dirname(model_dir.rstrip("/\\")))
                    or "student3",
                }
            )

    pr_auc = float(average_precision_score(labels, p_yes))
    try:
        roc_auc = float(roc_auc_score(labels, p_yes))
    except ValueError:
        roc_auc = float("nan")

    print(f"\nWrote: {out_csv}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return {
        "out_csv": out_csv,
        "n": len(items),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "p_yes": p_yes,
        "labels": labels,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export test-set probabilities.")
    parser.add_argument("--model-dir", default=os.environ.get("STUDENT3_DIR", STUDENT3_DIR))
    parser.add_argument("--test-json", default=TEST_JSON)
    parser.add_argument("--out-csv", default=OUT_CSV)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv if argv is not None else None)

    run_export(
        model_dir=args.model_dir,
        test_json=args.test_json,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
