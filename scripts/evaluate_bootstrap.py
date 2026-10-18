#!/usr/bin/env python3
"""Optional: bootstrap 95% CIs on a labeled JSON split (paper-style reporting)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from datasets import Dataset
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from constructive.training.metrics import compute_metrics, logits_to_p1


def bootstrap_ci(labels, scores, n_boot: int, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    pr, roc, f1b = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(len(labels), size=len(labels), replace=True)
        y, s = labels[idx], scores[idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        pr.append(average_precision_score(y, s))
        try:
            roc.append(roc_auc_score(y, s))
        except ValueError:
            pass
        best = max(
            f1_score(y, (s >= th).astype(int), zero_division=0) for th in np.arange(0.05, 0.96, 0.025)
        )
        f1b.append(best)

    def band(arr):
        if not arr:
            return {"mean": float("nan"), "lo95": float("nan"), "hi95": float("nan")}
        a = np.asarray(arr)
        return {
            "mean": round(float(a.mean()), 4),
            "lo95": round(float(np.percentile(a, 2.5)), 4),
            "hi95": round(float(np.percentile(a, 97.5)), 4),
        }

    return {"pr_auc": band(pr), "roc_auc": band(roc), "f1_best": band(f1b)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--json", required=True, help="Labeled split with text field")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.json, encoding="utf-8") as f:
        items = json.load(f)
    rows = [
        {"text": x["text"], "labels": int(x["label"])}
        for x in items
        if x.get("text") and x.get("label") in (0, 1)
    ]
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    ds = Dataset.from_list(rows).map(
        lambda b: tok(b["text"], truncation=True, max_length=512),
        batched=True,
        load_from_cache_file=False,
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=2)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/yc_bootstrap", per_device_eval_batch_size=32, report_to="none"),
        compute_metrics=compute_metrics,
    )
    pred = trainer.predict(ds)
    labels = pred.label_ids.astype(int)
    p1 = logits_to_p1(pred.predictions)
    print(json.dumps(bootstrap_ci(labels, p1, args.n_bootstrap, args.seed), indent=2))


if __name__ == "__main__":
    main()
