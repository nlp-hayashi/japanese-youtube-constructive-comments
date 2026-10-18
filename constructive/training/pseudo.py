from __future__ import annotations

import glob
import os
import random
from typing import Any

import pandas as pd

from constructive.training.config import DEFAULT_PSEUDO_BINS
from constructive.training.data import sanitize_text


def collect_pseudo_rows(
    scored_glob: str,
    *,
    bins: list[dict] | None = None,
    blocked_videos: set,
    exclude_comment_ids: set,
    seed: int,
    per_video_cap: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (pseudo_rows, bin_report, per_file_stats)."""
    bins = bins or DEFAULT_PSEUDO_BINS
    collectors: list[list[dict]] = [[] for _ in bins]
    file_stats: list[dict] = []

    for fp in sorted(glob.glob(scored_glob)):
        df = pd.read_csv(fp)
        if "text" not in df.columns and "comment" in df.columns:
            df = df.rename(columns={"comment": "text"})
        if not {"videoID", "p_constructive", "text"}.issubset(df.columns):
            continue

        df = df[df["videoID"].notna() & ~df["videoID"].isin(blocked_videos)].copy()
        df["p_constructive"] = pd.to_numeric(df["p_constructive"], errors="coerce")
        df = df[df["p_constructive"].notna()].copy()
        df["text"] = df["text"].astype(str).map(lambda x: sanitize_text(x))
        df = df[df["text"].str.len() > 0]

        f_stat: dict[str, Any] = {"file": os.path.basename(fp), "total_rows": len(df)}
        for i, bdef in enumerate(bins):
            mask = (df["p_constructive"] >= bdef["lo"]) & (df["p_constructive"] < bdef["hi"])
            bin_df = df[mask]
            if len(bin_df) > 0:
                bin_df = bin_df.groupby("videoID").head(per_video_cap)
            for _, r in bin_df.iterrows():
                p = float(r["p_constructive"])
                collectors[i].append(
                    {
                        "videoID": r["videoID"],
                        "comment_id": r.get("comment_id"),
                        "text": r["text"],
                        "labels": int(round(p)),
                        "soft_label": p,
                        "sample_weight": float(bdef["weight"]),
                        "source": f"pseudo:{os.path.basename(fp)}",
                        "p_teacher": p,
                    }
                )
            f_stat[f"bin_{bdef['lo']:.2f}_{bdef['hi']:.3f}"] = len(bin_df)
        file_stats.append(f_stat)

    pseudo_rows: list[dict] = []
    bin_report: list[dict] = []
    for i, bdef in enumerate(bins):
        candidates = collectors[i]
        quota = bdef["quota"]
        if len(candidates) > quota:
            reverse = bdef["label_type"] == "POS"
            candidates = sorted(candidates, key=lambda r: r["p_teacher"], reverse=reverse)[:quota]
        selected = candidates
        pseudo_rows.extend(selected)
        bin_report.append(
            {
                "type": bdef["label_type"],
                "range": f"[{bdef['lo']}, {bdef['hi']})",
                "candidates": len(collectors[i]),
                "selected": len(selected),
                "quota": quota,
                "weight": bdef["weight"],
                "unique_videos": len({r["videoID"] for r in selected}),
            }
        )

    pseudo_rows = [r for r in pseudo_rows if r.get("comment_id") not in exclude_comment_ids]
    random.seed(seed)
    random.shuffle(pseudo_rows)
    return pseudo_rows, bin_report, file_stats


def class_weights_from_mix(gold_train: list[dict], pseudo_rows: list[dict]):
    import torch

    pos = sum(r["labels"] == 1 for r in gold_train) + sum(
        r["sample_weight"] for r in pseudo_rows if r["labels"] == 1
    )
    neg = sum(r["labels"] == 0 for r in gold_train) + sum(
        r["sample_weight"] for r in pseudo_rows if r["labels"] == 0
    )
    total = pos + neg
    w_pos = total / (2 * pos) if pos > 0 else 1.0
    w_neg = total / (2 * neg) if neg > 0 else 1.0
    return torch.tensor([w_neg, w_pos], dtype=torch.float32), float(w_neg), float(w_pos)
