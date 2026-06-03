"""Batch collation for variable-length stroke3 and tok-dict samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dataloaders.masks import build_sequence_masks


def _round_up(value: int, multiple: int | None) -> int:
    if not multiple or multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


@dataclass
class Stroke3Collator:
    """Pad stroke3 samples and build masks for attention/losses."""

    max_length: int | None = None
    pad_value: float = 0.0
    pad_to_multiple_of: int | None = 8
    causal_attention: bool = False
    build_attention_mask: bool = True

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")

        raw_lengths = [int(sample["length"]) for sample in samples]
        lengths = [
            min(length, self.max_length) if self.max_length is not None else length
            for length in raw_lengths
        ]
        batch_max = max(lengths)
        if self.max_length is not None:
            batch_max = min(batch_max, self.max_length)
        padded_length = _round_up(batch_max, self.pad_to_multiple_of)

        strokes = torch.full(
            (len(samples), padded_length, 3),
            fill_value=float(self.pad_value),
            dtype=torch.float32,
        )
        labels = torch.empty(len(samples), dtype=torch.long)
        source_indices = torch.empty(len(samples), dtype=torch.long)
        source_files: list[str] = []

        for row, sample in enumerate(samples):
            sequence = np.asarray(sample["stroke3"], dtype=np.float32)
            length = lengths[row]
            strokes[row, :length] = torch.from_numpy(sequence[:length])
            labels[row] = int(sample["label"])
            source_indices[row] = int(sample["source_index"])
            source_files.append(str(sample["source_file"]))

        lengths_tensor = torch.as_tensor(lengths, dtype=torch.long)
        masks = build_sequence_masks(
            lengths_tensor,
            max_length=padded_length,
            causal=self.causal_attention,
            device=strokes.device,
        )
        if not self.build_attention_mask:
            masks["sdpa_mask"] = None

        return {
            "strokes": strokes,
            "targets": strokes.clone(),
            "lengths": lengths_tensor,
            "raw_lengths": torch.as_tensor(raw_lengths, dtype=torch.long),
            "labels": labels,
            "source_indices": source_indices,
            "source_files": source_files,
            **masks,
        }


@dataclass
class TokenSequenceCollator:
    """Pad tok-dict samples and build masks for attention/losses."""

    max_length: int | None = None
    pad_token_id: int = 1002
    pad_to_multiple_of: int | None = 8
    causal_attention: bool = False
    build_attention_mask: bool = True

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")

        raw_lengths = [int(sample["length"]) for sample in samples]
        lengths = [
            min(length, self.max_length) if self.max_length is not None else length
            for length in raw_lengths
        ]
        batch_max = max(lengths)
        if self.max_length is not None:
            batch_max = min(batch_max, self.max_length)
        padded_length = _round_up(batch_max, self.pad_to_multiple_of)

        tokens = torch.full(
            (len(samples), padded_length),
            fill_value=int(self.pad_token_id),
            dtype=torch.long,
        )
        labels = torch.empty(len(samples), dtype=torch.long)
        source_indices = torch.empty(len(samples), dtype=torch.long)
        source_files: list[str] = []

        for row, sample in enumerate(samples):
            sequence = np.asarray(sample["tokens"], dtype=np.int64)
            length = lengths[row]
            tokens[row, :length] = torch.from_numpy(sequence[:length]).long()
            labels[row] = int(sample["label"])
            source_indices[row] = int(sample["source_index"])
            source_files.append(str(sample["source_file"]))

        lengths_tensor = torch.as_tensor(lengths, dtype=torch.long)
        masks = build_sequence_masks(
            lengths_tensor,
            max_length=padded_length,
            causal=self.causal_attention,
            device=tokens.device,
        )
        if not self.build_attention_mask:
            masks["sdpa_mask"] = None

        return {
            "tokens": tokens,
            "targets": tokens.clone(),
            "lengths": lengths_tensor,
            "raw_lengths": torch.as_tensor(raw_lengths, dtype=torch.long),
            "labels": labels,
            "source_indices": source_indices,
            "source_files": source_files,
            "pad_token_id": int(self.pad_token_id),
            **masks,
        }
