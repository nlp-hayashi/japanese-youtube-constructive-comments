from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PSEUDO_BINS: list[dict] = [
    {"lo": 0.00, "hi": 0.05, "quota": 2000, "weight": 1.0, "label_type": "NEG"},
    {"lo": 0.98, "hi": 1.001, "quota": 500, "weight": 1.0, "label_type": "POS"},
]


@dataclass(frozen=True)
class TeacherHyperparams:
    model_name: str = "sbintuitions/modernbert-ja-130m"
    max_len: int = 512
    seed: int = 42
    lr: float = 2e-5
    weight_decay: float = 0.05
    epochs: int = 10
    batch_train: int = 16
    batch_eval: int = 32
    logging_steps: int = 10
    fp16: bool = True
    patience: int = 3


@dataclass(frozen=True)
class StudentHyperparams:
    max_len: int = 512
    seed: int = 42
    lr: float = 2e-6
    weight_decay: float = 0.05
    warmup_ratio: float = 0.06
    epochs: int = 10
    batch_train: int = 16
    batch_eval: int = 32
    logging_steps: int = 20
    fp16: bool = True
    patience: int = 5
    per_video_cap_per_bin: int = 20
    min_epochs_before_best: int = 0
