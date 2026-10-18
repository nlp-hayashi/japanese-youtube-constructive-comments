from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def logits_to_p1(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return (probs / probs.sum(axis=1, keepdims=True))[:, 1]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = np.asarray(labels).astype(int)
    p1 = logits_to_p1(logits)
    pred = (p1 >= 0.5).astype(int)

    acc = accuracy_score(labels, pred)
    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    pr_auc = average_precision_score(labels, p1)
    try:
        roc_auc = roc_auc_score(labels, p1)
    except ValueError:
        roc_auc = float("nan")

    best_f1, best_th = 0.0, 0.5
    for th in np.arange(0.05, 0.96, 0.025):
        f1_th = f1_score(labels, (p1 >= th).astype(int), zero_division=0)
        if f1_th > best_f1:
            best_f1, best_th = f1_th, float(th)

    return {
        "acc": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1_best": best_f1,
        "f1_best_threshold": best_th,
    }


def export_predictions_csv(pred, meta: pd.DataFrame, path: str) -> None:
    p1 = logits_to_p1(np.asarray(pred.predictions))
    df = meta.copy()
    df["label"] = np.asarray(pred.label_ids)
    df["pred"] = (p1 >= 0.5).astype(int)
    df["p_constructive"] = p1
    df["text"] = df["text"].astype(str).str.replace("\r", " ").str.replace("\n", " ")
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, escapechar="\\")
