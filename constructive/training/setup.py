from __future__ import annotations

import torch
from transformers import set_seed


def set_deterministic_seed(seed: int) -> None:
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
