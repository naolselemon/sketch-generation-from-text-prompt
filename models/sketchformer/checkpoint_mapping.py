"""Checkpoint helpers for the native PyTorch Sketchformer path.

TensorFlow-to-PyTorch conversion will be implemented after the architecture and
loss contract are stable. This module already contains small loading utilities
used by future conversion scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Summary of a partial checkpoint load."""

    missing_keys: list[str]
    unexpected_keys: list[str]


def load_torch_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = False,
) -> CheckpointLoadReport:
    """Load a PyTorch state dict and return missing/unexpected key details."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict: dict[str, Any]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError("Checkpoint must be a state-dict-like mapping")

    incompatible = model.load_state_dict(state_dict, strict=strict)
    return CheckpointLoadReport(
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
    )


def resize_learned_position_embedding(
    source_weight: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Resize learned position embeddings by linear interpolation."""

    if source_weight.ndim != 2:
        raise ValueError("Expected positional embedding weight with shape (length, dim)")
    if source_weight.shape[0] == target_length:
        return source_weight

    resized = torch.nn.functional.interpolate(
        source_weight.T.unsqueeze(0),
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    return resized.squeeze(0).T.contiguous()
