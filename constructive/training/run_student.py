from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

from constructive.training.config import DEFAULT_PSEUDO_BINS, StudentHyperparams
from constructive.training.data import (
    blocked_video_ids,
    gold_comment_ids,
    load_labeled_json,
    merge_train_with_al,
)
from constructive.training.metrics import compute_metrics
from constructive.training.pseudo import class_weights_from_mix, collect_pseudo_rows
from constructive.training.setup import set_deterministic_seed
from constructive.training.trainers import SoftLabelCollator, SoftLabelTrainer


def _tokenize(ds: Dataset, tokenizer, max_len: int) -> Dataset:
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_len),
        batched=True,
        load_from_cache_file=False,
    )


def train_student(
    train_json: str,
    valid_json: str,
    test_json: str,
    init_checkpoint: str,
    out_dir: str,
    *,
    hp: StudentHyperparams | None = None,
    pseudo_glob: str | None = None,
    pseudo_bins: list[dict] | None = None,
    al_json_paths: list[str] | None = None,
    report_json: str | None = None,
) -> dict:
    hp = hp or StudentHyperparams()
    pseudo_bins = pseudo_bins or DEFAULT_PSEUDO_BINS
    set_deterministic_seed(hp.seed)
    os.makedirs(out_dir, exist_ok=True)

    base_train = load_labeled_json(train_json)
    al_parts = [load_labeled_json(p) for p in (al_json_paths or []) if os.path.isfile(p)]
    gold_train = merge_train_with_al(base_train, al_parts) if al_parts else base_train
    gold_valid = load_labeled_json(valid_json)
    gold_test = load_labeled_json(test_json)

    blocked = blocked_video_ids(valid_json, test_json)
    exclude_ids = gold_comment_ids(gold_train)

    pseudo_rows: list[dict] = []
    bin_report: list[dict] = []
    file_stats: list[dict] = []
    if pseudo_glob:
        pseudo_rows, bin_report, file_stats = collect_pseudo_rows(
            pseudo_glob,
            bins=pseudo_bins,
            blocked_videos=blocked,
            exclude_comment_ids=exclude_ids,
            seed=hp.seed,
            per_video_cap=hp.per_video_cap_per_bin,
        )

    class_weights, w_neg, w_pos = class_weights_from_mix(gold_train, pseudo_rows)
    train_rows = gold_train + pseudo_rows
    print(f"Train rows: gold={len(gold_train)} pseudo={len(pseudo_rows)} total={len(train_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(init_checkpoint, use_fast=True)
    ds_train = _tokenize(Dataset.from_list(train_rows), tokenizer, hp.max_len)
    ds_valid = _tokenize(Dataset.from_list(gold_valid), tokenizer, hp.max_len)
    ds_test = _tokenize(Dataset.from_list(gold_test), tokenizer, hp.max_len)

    ds_train.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "soft_label", "sample_weight"],
    )
    for ds in (ds_valid, ds_test):
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    steps_per_epoch = max(1, (len(ds_train) + hp.batch_train - 1) // hp.batch_train)
    min_steps = max(1, int(hp.min_epochs_before_best * steps_per_epoch))
    warmup_steps = max(1, int(steps_per_epoch * hp.epochs * hp.warmup_ratio))

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=hp.lr,
        warmup_steps=warmup_steps,
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
    trainer = SoftLabelTrainer(
        class_weights=class_weights,
        min_steps_before_best=min_steps,
        model=AutoModelForSequenceClassification.from_pretrained(init_checkpoint, num_labels=2),
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        data_collator=SoftLabelCollator(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=hp.patience)],
    )
    trainer.train()

    best_dir = str(Path(out_dir) / "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Saved checkpoint: {best_dir}")

    valid_metrics = trainer.evaluate(ds_valid)
    test_metrics = trainer.evaluate(ds_test)
    print("Valid:", valid_metrics)
    print("Test:", test_metrics)

    summary = {
        "init_checkpoint": init_checkpoint,
        "pseudo_glob": pseudo_glob,
        "al_json_paths": al_json_paths or [],
        "gold_train": len(gold_train),
        "pseudo_train": len(pseudo_rows),
        "class_weights": {"w_neg": w_neg, "w_pos": w_pos},
        "bin_report": bin_report,
        "pseudo_file_stats": file_stats,
        "valid": valid_metrics,
        "test": test_metrics,
    }
    if report_json:
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
