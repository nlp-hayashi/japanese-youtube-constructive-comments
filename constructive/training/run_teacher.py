from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
)

from constructive.training.config import TeacherHyperparams
from constructive.training.data import load_labeled_json
from constructive.training.metrics import compute_metrics, export_predictions_csv
from constructive.training.setup import set_deterministic_seed
from constructive.training.trainers import WeightedCETrainer


def train_teacher(
    train_json: str,
    valid_json: str,
    test_json: str,
    out_dir: str,
    *,
    hp: TeacherHyperparams | None = None,
    metrics_json: str | None = None,
    pred_valid_csv: str | None = None,
    pred_test_csv: str | None = None,
) -> dict:
    hp = hp or TeacherHyperparams()
    set_deterministic_seed(hp.seed)
    os.makedirs(out_dir, exist_ok=True)

    train_rows = load_labeled_json(train_json)
    valid_rows = load_labeled_json(valid_json)
    test_rows = load_labeled_json(test_json)
    print(f"Train / valid / test: {len(train_rows)} / {len(valid_rows)} / {len(test_rows)}")

    n_pos = sum(r["labels"] == 1 for r in train_rows)
    n_neg = len(train_rows) - n_pos
    total = n_pos + n_neg
    w_pos = total / (2 * n_pos) if n_pos else 1.0
    w_neg = total / (2 * n_neg) if n_neg else 1.0
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)

    meta_valid = pd.DataFrame(valid_rows)[["videoID", "comment_id", "text"]]
    meta_test = pd.DataFrame(test_rows)[["videoID", "comment_id", "text"]]

    ds = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(valid_rows),
            "test": Dataset.from_list(test_rows),
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(hp.model_name, use_fast=True)
    ds_tok = ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=hp.max_len),
        batched=True,
        load_from_cache_file=False,
    )

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=hp.lr,
        weight_decay=hp.weight_decay,
        per_device_train_batch_size=hp.batch_train,
        per_device_eval_batch_size=hp.batch_eval,
        num_train_epochs=hp.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        logging_steps=hp.logging_steps,
        report_to="none",
        fp16=hp.fp16,
        seed=hp.seed,
        data_seed=hp.seed,
        dataloader_num_workers=0,
    )
    trainer = WeightedCETrainer(
        class_weights=class_weights,
        model=AutoModelForSequenceClassification.from_pretrained(hp.model_name, num_labels=2),
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=hp.patience)],
    )
    trainer.train()

    best_dir = str(Path(out_dir) / "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Saved checkpoint: {best_dir}")

    valid_metrics = trainer.evaluate(ds_tok["validation"])
    test_metrics = trainer.evaluate(ds_tok["test"])
    print("Valid:", valid_metrics)
    print("Test:", test_metrics)

    if pred_valid_csv:
        export_predictions_csv(trainer.predict(ds_tok["validation"]), meta_valid, pred_valid_csv)
    if pred_test_csv:
        export_predictions_csv(trainer.predict(ds_tok["test"]), meta_test, pred_test_csv)

    summary = {
        "valid": valid_metrics,
        "test": test_metrics,
        "train_counts": {"pos": n_pos, "neg": n_neg},
        "class_weights": {"w_neg": float(w_neg), "w_pos": float(w_pos)},
    }
    if metrics_json:
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
