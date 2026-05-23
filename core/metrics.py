"""Metrics for Sketchformer reconstruction validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from core.losses import masked_mean


@dataclass
class ReconstructionMetrics:
    """Common validation metrics for continuous stroke reconstruction."""

    xy_mse: torch.Tensor
    xy_l1: torch.Tensor
    pen_accuracy: torch.Tensor
    valid_tokens: torch.Tensor

    def as_log_dict(self, prefix: str = "val") -> dict[str, torch.Tensor]:
        return {
            f"{prefix}/xy_mse": self.xy_mse.detach(),
            f"{prefix}/xy_l1": self.xy_l1.detach(),
            f"{prefix}/pen_accuracy": self.pen_accuracy.detach(),
            f"{prefix}/valid_tokens": self.valid_tokens.detach(),
        }


@torch.no_grad()
def reconstruction_metrics(output: Any, batch: Mapping[str, torch.Tensor]) -> ReconstructionMetrics:
    """Compute mask-aware reconstruction metrics."""

    if output.reconstruction is None:
        raise ValueError("Model output does not include reconstruction predictions")

    targets = batch["targets"]
    valid_mask = batch.get("valid_mask")
    if valid_mask is None:
        valid_mask = torch.ones(targets.shape[:2], dtype=torch.bool, device=targets.device)
    else:
        valid_mask = valid_mask.to(device=targets.device, dtype=torch.bool)

    target_xy = targets[..., :2]
    pred_xy = output.reconstruction.xy
    xy_mse = masked_mean(F.mse_loss(pred_xy, target_xy, reduction="none"), valid_mask)
    xy_l1 = masked_mean(F.l1_loss(pred_xy, target_xy, reduction="none"), valid_mask)

    pen_target = targets[..., 2].round().long().clamp(
        min=0,
        max=output.reconstruction.pen_logits.shape[-1] - 1,
    )
    pen_pred = torch.argmax(output.reconstruction.pen_logits, dim=-1)
    correct = (pen_pred == pen_target).to(dtype=torch.float32)
    pen_accuracy = masked_mean(correct, valid_mask)
    valid_tokens = valid_mask.sum().to(dtype=torch.float32)

    return ReconstructionMetrics(
        xy_mse=xy_mse,
        xy_l1=xy_l1,
        pen_accuracy=pen_accuracy,
        valid_tokens=valid_tokens,
    )
