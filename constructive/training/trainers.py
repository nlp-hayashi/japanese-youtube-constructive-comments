from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DataCollatorWithPadding, Trainer


class SoftLabelCollator:
    def __init__(self, tokenizer):
        self._pad = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        extras = {}
        for key in ("soft_label", "sample_weight"):
            if key in features[0]:
                extras[key] = torch.tensor([float(f.pop(key)) for f in features], dtype=torch.float32)
            else:
                for f in features:
                    f.pop(key, None)
        batch = self._pad(features)
        batch.update(extras)
        return batch


class WeightedCETrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        cw = self.class_weights.to(outputs.logits.device)
        loss = nn.CrossEntropyLoss(weight=cw)(outputs.logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class SoftLabelTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, min_steps_before_best: int | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.min_steps_before_best = min_steps_before_best

    def _determine_best_metric(self, metrics, trial=None):
        if self.min_steps_before_best is not None and self.state.global_step < self.min_steps_before_best:
            return False
        return super()._determine_best_metric(metrics, trial)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        soft_label = inputs.get("soft_label")
        sample_w = inputs.get("sample_weight")
        model_inputs = {k: v for k, v in inputs.items() if k not in ("labels", "soft_label", "sample_weight")}
        outputs = model(**model_inputs)
        logits = outputs.logits
        cw = self.class_weights.to(logits.device)

        if soft_label is not None:
            log_probs = F.log_softmax(logits, dim=-1)
            targets = torch.stack([1.0 - soft_label, soft_label], dim=-1)
            loss_vec = -(targets * log_probs).sum(dim=-1)
            w = targets[:, 0] * cw[0] + targets[:, 1] * cw[1]
            loss_vec = loss_vec * w
        else:
            loss_vec = nn.CrossEntropyLoss(weight=cw, reduction="none")(
                logits.view(-1, 2), labels.view(-1)
            )

        if sample_w is not None:
            loss = (loss_vec * sample_w.to(logits.device).view(-1)).mean()
        else:
            loss = loss_vec.mean()
        return (loss, outputs) if return_outputs else loss
