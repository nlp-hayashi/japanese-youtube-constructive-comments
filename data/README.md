# Dataset (Hugging Face)

Human labels for constructive vs non-constructive comments on Japanese YouTube Shorts.

**Hugging Face:** [nlp-hayashi/japanese-youtube-constructive-comments](https://huggingface.co/datasets/nlp-hayashi/japanese-youtube-constructive-comments)  
**Paper:** TBD

## Record fields

| Field | Description |
|--------|-------------|
| `videoID` | YouTube video id. |
| `comment_id` | YouTube comment id (use with YouTube Data API v3 to recover text locally). |
| `annotations` | Per-annotator judgments (see below). |
| `label` | Final gold label: `0` non-constructive, `1` constructive. |

Each element of `annotations`:

| Field | Description |
|--------|-------------|
| `annotator_id` | Opaque id (e.g. `A1`, `A2`); no personal names in the public release. |
| `premise` | Criterion 1 (`0` / `1`). |
| `additional_info` | Criterion 2 (`0` / `1`). |

**Final label:** `label = 1` iff `premise == 1` and `additional_info == 1`; otherwise `0`.

## Files

| File | Role |
|------|------|
| `dataset_train.json` | Train split |
| `dataset_valid.json` | Validation split |
| `dataset_test.json` | Test split |
| `dataset_active_learning_al_round1.json` | AL round 1 |
| `dataset_active_learning_al_round2.json` | AL round 2 |

Public JSON has no comment `text` (restore via `scripts/restore_comment_text.py` and your API key; see main [README](../README.md)). Pseudo-label scores and the unlabeled crawl pool are not published on Hugging Face.

## License

[Creative Commons Attribution 4.0 (CC BY 4.0)](LICENSE). Repository source code is [MIT](../LICENSE).
