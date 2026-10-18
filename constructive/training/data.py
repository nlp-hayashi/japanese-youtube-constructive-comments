from __future__ import annotations

import json
from typing import Iterable


def sanitize_text(text: str | None) -> str:
    if not text:
        return ""
    return str(text).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()


def load_labeled_json(path: str) -> list[dict]:
    """Rows for soft-label training: videoID, comment_id, text, labels, soft_label, …"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows: list[dict] = []
    for item in data:
        text = sanitize_text(item.get("text"))
        if not text:
            continue
        vid = item.get("videoID")
        if vid is None:
            continue
        lbl = item.get("label")
        if lbl not in (0, 1):
            continue
        label = int(lbl)
        rows.append(
            {
                "videoID": vid,
                "comment_id": item.get("comment_id"),
                "text": text,
                "labels": label,
                "soft_label": float(label),
                "sample_weight": 1.0,
                "source": "gold",
            }
        )
    return rows


def merge_train_with_al(base: list[dict], al_parts: Iterable[list[dict]]) -> list[dict]:
    """On duplicate comment_id, later (AL) rows win."""
    by_id: dict = {}
    for r in base:
        cid = r.get("comment_id")
        if cid is not None:
            by_id[cid] = r
    for part in al_parts:
        for r in part:
            cid = r.get("comment_id")
            if cid is not None:
                by_id[cid] = r
    return list(by_id.values())


def blocked_video_ids(*json_paths: str) -> set:
    out: set = set()
    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            for item in json.load(f):
                vid = item.get("videoID")
                if vid:
                    out.add(vid)
    return out


def gold_comment_ids(rows: list[dict]) -> set:
    return {r["comment_id"] for r in rows if r.get("comment_id") is not None}
