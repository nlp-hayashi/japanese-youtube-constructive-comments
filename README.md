# Constructive YouTube Shorts Comments

Code for binary classification of **constructive** vs **non-constructive** comments on Japanese YouTube Shorts.

**Paper:** TBD  
**Data (Hugging Face):** [nlp-hayashi/japanese-youtube-constructive-comments](https://huggingface.co/datasets/nlp-hayashi/japanese-youtube-constructive-comments) — see also [`data/README.md`](data/README.md)

## Citation

```bibtex
% TODO
```

```
constructive/     shared Python code (training, paths, metrics)
scripts/          run these from repo root (see below)
data/             HF splits (download separately)
demo/             browser extension + local API
```

Run all commands from the repo root.

## Setup

```bash
cd /path/to/this/repo
conda create -n constructive python=3.10 -y
conda activate constructive
pip install -r requirements.txt
export DATA_DIR=$(pwd)/data
export OUTPUT_DIR=$(pwd)/outputs
```

## 1. Download labels

```bash
huggingface-cli download nlp-hayashi/japanese-youtube-constructive-comments --repo-type dataset --local-dir ./data
```

See `data/README.md` for file names.

## 2. Restore comment text

```bash
export YOUTUBE_API_KEY=your_key
python scripts/restore_comment_text.py --data-dir ./data --all-splits
```

Training reads `*_with_text.json` when present.

## 3. Train Teacher

```bash
python scripts/02_train/train_teacher.py
```

## 4. Score unlabeled pool

Your own JSONL shards; not on Hugging Face.

```bash
python scripts/01_score/score_unlabeled_jsonl.py \
  --input-dir /path/to/jsonl_corpus \
  --model-dir outputs/teacher_simple_best/best
```

## 5. Train Student (self-training)

```bash
python scripts/02_train/train_student_self_training.py
```

## 6. Active learning

```bash
python scripts/02_train/train_student_active_learning.py \
  --run-dir student2_active \
  --init-checkpoint student_self_training/best \
  --al-json dataset_active_learning_al_round1.json \
  --pseudo-glob pseudo_scores/student1/*.csv
```

Repeat with the next student checkpoint, additional `--al-json` files, and matching pseudo scores for later rounds.

## 7. Evaluate

```bash
python scripts/01_score/export_test_probabilities.py \
  --test-json data/dataset_test_with_text.json \
  --model-dir outputs/student3_active/best

python scripts/evaluate_test.py
```

## Reproducibility

| Stage | What you need | On Hugging Face |
|--------|----------------|-----------------|
| Teacher | Gold splits + restored text | Labels |
| Student self-training | Unlabeled pool + teacher scores | — |
| Active learning | Student checkpoint + AL JSON + restored text | AL labels |

The crawled Shorts pool and pseudo-label CSVs are not redistributed.

## Demo

`demo/constructive_extension/`

## License

Source code: [MIT](LICENSE). Dataset: [CC BY 4.0](data/LICENSE).
