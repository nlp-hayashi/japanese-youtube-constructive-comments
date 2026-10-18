#!/usr/bin/env python3
"""
Test-set metrics for binary constructive classification.

For each prediction CSV: PR-AUC, ROC-AUC, macro-F1 at threshold 0.5.

Optional: --student / --gpt-0shot / --gpt-fewshot for paired bootstrap (PR-AUC)
and DeLong (ROC-AUC) on the same 660 test comments.

CSV must include labels and probabilities (columns: gold_label or label;
score, p_yes, or p_constructive).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from constructive.config import get_output_dir

THRESHOLD = 0.5
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42


def _label_and_prob_cols(df: pd.DataFrame) -> tuple[str, str]:
    if "gold_label" in df.columns:
        label_col = "gold_label"
    elif "label" in df.columns:
        label_col = "label"
    else:
        raise ValueError("Need label column: gold_label or label")
    if "score" in df.columns:
        p_col = "score"
    elif "p_yes" in df.columns:
        p_col = "p_yes"
    elif "p_constructive" in df.columns:
        p_col = "p_constructive"
    else:
        raise ValueError("Need probability column: score, p_yes, or p_constructive")
    return label_col, p_col


def metrics_for_csv(path: Path, threshold: float = THRESHOLD) -> dict:
    df = pd.read_csv(path)
    label_col, p_col = _label_and_prob_cols(df)
    y = pd.to_numeric(df[label_col], errors="coerce")
    p = pd.to_numeric(df[p_col], errors="coerce")
    mask = y.notna() & p.notna()
    y = y[mask].astype(int).values
    p = p[mask].astype(float).values
    if len(y) == 0:
        raise ValueError(f"No valid rows in {path}")
    y_pred = (p >= threshold).astype(int)
    return {
        "file": str(path),
        "n": int(len(y)),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "threshold": threshold,
    }


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1)
        i = k
    t2 = np.empty(n, dtype=np.float64)
    t2[j] = t + 1
    return t2


def _delong_roc_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    m = int(y_true.sum())
    preds = np.vstack([np.asarray(pred_a), np.asarray(pred_b)])[:, order]
    n = preds.shape[1] - m
    k = preds.shape[0]
    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = _compute_midrank(preds[r, :m])
        ty[r, :] = _compute_midrank(preds[r, m:])
        tz[r, :] = _compute_midrank(preds[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    diff = aucs[0] - aucs[1]
    var = max(delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1], 1e-12)
    z = diff / np.sqrt(var)
    return {
        "delta_a_minus_b": float(diff),
        "z": float(z),
        "p_two_sided": float(2 * stats.norm.sf(abs(z))),
    }


def _bootstrap_pr_diff(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    n = len(y_true)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = average_precision_score(y_true[idx], pred_a[idx]) - average_precision_score(
            y_true[idx], pred_b[idx]
        )
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_one = float(np.mean(deltas <= 0))
    return {
        "delta_a_minus_b": float(np.mean(deltas)),
        "ci_95_low": float(lo),
        "ci_95_high": float(hi),
        "p_two_sided": float(min(1.0, 2 * min(p_one, 1 - p_one))),
        "n_bootstrap": n_boot,
    }


def pairwise_stats(student_csv: Path, gpt0_csv: Path, gpt_few_csv: Path, n_boot: int) -> dict:
    s3 = pd.read_csv(student_csv)
    g0 = pd.read_csv(gpt0_csv)
    gf = pd.read_csv(gpt_few_csv)
    key = "comment_id"
    if key not in s3.columns:
        raise ValueError("student CSV needs comment_id for alignment")
    m = s3[[key, "gold_label" if "gold_label" in s3.columns else "label"]].rename(
        columns={"gold_label" if "gold_label" in s3.columns else "label": "y"}
    )
    _, p_col_s = _label_and_prob_cols(s3)
    m["p_student"] = s3[p_col_s].values
    _, p0 = _label_and_prob_cols(g0)
    m = m.merge(g0[[key, p0]].rename(columns={p0: "p_gpt0"}), on=key, how="inner")
    _, pf = _label_and_prob_cols(gf)
    m = m.merge(gf[[key, pf]].rename(columns={pf: "p_gpt_few"}), on=key, how="inner")
    y = m["y"].astype(int).values
    pairs = [
        ("Student", "GPT-4o_0shot", "p_student", "p_gpt0"),
        ("Student", "GPT-4o_fewshot", "p_student", "p_gpt_few"),
    ]
    out = []
    for name_a, name_b, ca, cb in pairs:
        pa, pb = m[ca].values, m[cb].values
        out.append(
            {
                "model_a": name_a,
                "model_b": name_b,
                "pr_auc_bootstrap": _bootstrap_pr_diff(y, pa, pb, n_boot),
                "roc_auc_delong": _delong_roc_test(y, pa, pb),
            }
        )
    return {"n": len(m), "pairwise": out}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate test prediction CSVs.")
    parser.add_argument(
        "csvs",
        nargs="*",
        type=Path,
        help="Prediction CSV paths (default: outputs/*_test_predictions.csv if any)",
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--student", type=Path, help="Student test prob CSV for GPT comparison")
    parser.add_argument("--gpt-0shot", type=Path, dest="gpt0")
    parser.add_argument("--gpt-fewshot", type=Path, dest="gpt_few")
    parser.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_N)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write full results JSON (default: outputs/eval_test_results.json)",
    )
    args = parser.parse_args()

    out_dir = get_output_dir()
    paths = list(args.csvs)
    if not paths:
        paths = sorted(out_dir.glob("*_test_predictions.csv"))
        paths += sorted(out_dir.glob("predictions/*_test_prob*.csv"))

    results: dict = {"per_model": [], "pairwise": None}
    if paths:
        for p in paths:
            if not p.is_file():
                print(f"Skip missing: {p}", file=sys.stderr)
                continue
            row = metrics_for_csv(p, args.threshold)
            row["model"] = p.stem.replace("_test_predictions", "").replace("_test_with_prob", "")
            results["per_model"].append(row)
            print(
                f"{row['model']:24s}  n={row['n']:4d}  "
                f"PR={row['pr_auc']:.4f}  ROC={row['roc_auc']:.4f}  "
                f"macro-F1@{args.threshold}={row['macro_f1']:.4f}"
            )
    else:
        print("No CSVs given and none found under OUTPUT_DIR.", file=sys.stderr)

    if args.student and args.gpt0 and args.gpt_few:
        print("\n--- Pairwise vs GPT (same test comments) ---")
        pw = pairwise_stats(args.student, args.gpt0, args.gpt_few, args.bootstrap_n)
        results["pairwise"] = pw
        for c in pw["pairwise"]:
            pr = c["pr_auc_bootstrap"]
            roc = c["roc_auc_delong"]
            print(
                f"{c['model_a']} vs {c['model_b']}:  "
                f"PR delta={pr['delta_a_minus_b']:+.4f} p={pr['p_two_sided']:.4f}  "
                f"ROC delta={roc['delta_a_minus_b']:+.4f} p={roc['p_two_sided']:.4f}"
            )

    out_json = args.out_json or (out_dir / "eval_test_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
