"""Positional encodings for long stroke sequences."""

from __future__ import annotations

import torch
from torch import nn


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding for variable-length stroke sequences."""

    def __init__(self, max_length: int, d_model: int) -> None:
        super().__init__()
        self.max_length = int(max_length)
        self.embedding = nn.Embedding(self.max_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input shape (batch, sequence, channels)")
        sequence_length = x.shape[1]
        if sequence_length > self.max_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds positional capacity {self.max_length}"
            )
        positions = torch.arange(sequence_length, device=x.device)
        return x + self.embedding(positions).unsqueeze(0)
