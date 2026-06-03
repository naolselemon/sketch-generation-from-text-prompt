"""Reconstruction reporting helpers for native Sketchformer evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.tokenizer import decode_tokens


@dataclass(frozen=True)
class ReconstructionExample:
    """One target/prediction pair prepared for qualitative inspection."""

    target: np.ndarray
    prediction: np.ndarray
    length: int
    source_file: str
    source_index: int
    label: int | None = None


def prediction_to_stroke3(output: Any) -> torch.Tensor:
    """Convert a model output object into predicted ``(dx, dy, pen)`` strokes."""

    if output.reconstruction is None:
        raise ValueError("Model output does not include reconstruction predictions")
    if getattr(output.reconstruction, "token_logits", None) is not None:
        raise ValueError("Token predictions require a codebook; use collect_reconstruction_examples")

    xy = output.reconstruction.xy
    pen = torch.argmax(output.reconstruction.pen_logits, dim=-1).to(dtype=xy.dtype)
    return torch.cat([xy, pen.unsqueeze(-1)], dim=-1)


def _batch_lengths(batch: Mapping[str, Any], fallback_mask: torch.Tensor | None) -> list[int]:
    lengths = batch.get("lengths")
    if torch.is_tensor(lengths):
        return [int(value) for value in lengths.detach().cpu().tolist()]
    if lengths is not None:
        return [int(value) for value in lengths]
    if fallback_mask is not None:
        return [int(value) for value in fallback_mask.detach().cpu().sum(dim=1).tolist()]
    return []


def collect_reconstruction_examples(
    output: Any,
    batch: Mapping[str, Any],
    *,
    max_examples: int,
    codebook: np.ndarray | None = None,
) -> list[ReconstructionExample]:
    """Collect a small CPU copy of target and predicted stroke3 sequences."""

    if max_examples <= 0:
        return []

    token_logits = (
        None
        if output.reconstruction is None
        else getattr(output.reconstruction, "token_logits", None)
    )
    if token_logits is not None and codebook is None:
        raise ValueError("A tok-dict codebook is required to plot token reconstructions")

    predictions = (
        torch.argmax(token_logits, dim=-1).detach().cpu().numpy()
        if token_logits is not None
        else prediction_to_stroke3(output).detach().cpu().numpy()
    )
    targets = batch["targets"].detach().cpu().numpy()
    valid_mask = batch.get("valid_mask")
    lengths = _batch_lengths(batch, valid_mask)
    if not lengths:
        lengths = [targets.shape[1]] * targets.shape[0]

    source_files = batch.get("source_files") or [""] * targets.shape[0]
    source_indices = batch.get("source_indices")
    labels = batch.get("labels")
    if torch.is_tensor(source_indices):
        source_indices = source_indices.detach().cpu().tolist()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().tolist()

    examples: list[ReconstructionExample] = []
    for row in range(min(max_examples, targets.shape[0])):
        length = min(int(lengths[row]), targets.shape[1])
        label = int(labels[row]) if labels is not None else None
        source_index = int(source_indices[row]) if source_indices is not None else row
        if token_logits is not None:
            assert codebook is not None
            target = decode_tokens(
                np.asarray(targets[row, :length], dtype=np.int64),
                codebook,
            )
            prediction = decode_tokens(
                np.asarray(predictions[row, :length], dtype=np.int64),
                codebook,
            )
        else:
            target = np.asarray(targets[row, :length], dtype=np.float32)
            prediction = np.asarray(predictions[row, :length], dtype=np.float32)
        examples.append(
            ReconstructionExample(
                target=target,
                prediction=prediction,
                length=length,
                source_file=str(source_files[row]),
                source_index=source_index,
                label=label,
            )
        )
    return examples


def tensor_logs_to_floats(logs: Mapping[str, torch.Tensor | float | int]) -> dict[str, float]:
    """Convert scalar tensor logs into JSON-safe floats."""

    result: dict[str, float] = {}
    for key, value in logs.items():
        if torch.is_tensor(value):
            result[key] = float(value.detach().cpu())
        else:
            result[key] = float(value)
    return result


def write_metrics_report(
    output_path: str | Path,
    logs: Mapping[str, torch.Tensor | float | int],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a compact JSON report for an evaluation run."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": tensor_logs_to_floats(logs),
        "metadata": dict(metadata or {}),
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
