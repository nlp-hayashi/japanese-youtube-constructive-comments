#!/usr/bin/env python3
# Restore comment text from comment_id (YouTube Data API v3). HF ships ids only.
#
#   export YOUTUBE_API_KEY=...
#   python scripts/restore_comment_text.py --data-dir ./data --all-splits
#   python scripts/restore_comment_text.py -i in.json -o out.json

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from constructive.config import get_data_dir, get_output_dir

API = "https://www.googleapis.com/youtube/v3/comments"
SPLITS = [
    "dataset_train.json",
    "dataset_valid.json",
    "dataset_test.json",
    "dataset_active_learning_al_round1.json",
    "dataset_active_learning_al_round2.json",
]


def fetch_batch(api_key: str, ids: list[str]) -> dict[str, str]:
    q = urllib.parse.urlencode({"part": "snippet", "id": ",".join(ids), "key": api_key})
    with urllib.request.urlopen(f"{API}?{q}", timeout=60) as resp:
        data = json.loads(resp.read())
    out = {}
    for item in data.get("items", []):
        sn = item.get("snippet") or {}
        out[item["id"]] = sn.get("textDisplay") or sn.get("textOriginal") or ""
    return out


def fill_cache(api_key: str, ids: list[str], cache: dict[str, str], cache_file: Path) -> None:
    for i in range(0, len(ids), 50):
        batch = [x for x in ids[i : i + 50] if x not in cache]
        if not batch:
            continue
        cache.update(fetch_batch(api_key, batch))
        cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        time.sleep(0.15)


def process(in_path: Path, out_path: Path, api_key: str, cache: dict[str, str], cache_file: Path) -> None:
    rows = json.loads(in_path.read_text(encoding="utf-8"))
    ids = list(dict.fromkeys(r["comment_id"] for r in rows if r.get("comment_id")))
    fill_cache(api_key, ids, cache, cache_file)
    for r in rows:
        r["text"] = cache.get(r["comment_id"], "")
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in rows if r["text"])
    print(f"{out_path}  {ok}/{len(rows)} with text")


def main() -> None:
    p = argparse.ArgumentParser(description="Restore comment text via YouTube API.")
    p.add_argument("--data-dir", type=Path, default=get_data_dir())
    p.add_argument("--all-splits", action="store_true")
    p.add_argument("-i", "--input", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--api-key", default=os.environ.get("YOUTUBE_API_KEY", ""))
    args = p.parse_args()

    if not args.api_key:
        sys.exit("Set YOUTUBE_API_KEY or --api-key")

    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "comment_text_cache.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.is_file() else {}

    if args.all_splits:
        for name in SPLITS:
            inp = args.data_dir / name
            if inp.is_file():
                process(inp, inp.with_name(inp.stem + "_with_text.json"), args.api_key, cache, cache_file)
    elif args.input and args.output:
        process(args.input, args.output, args.api_key, cache, cache_file)
    else:
        sys.exit("Use --all-splits or both -i and -o")


if __name__ == "__main__":
    main()
