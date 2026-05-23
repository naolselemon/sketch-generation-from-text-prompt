"""Input embeddings for continuous stroke3 data."""

from __future__ import annotations

import torch
from torch import nn

from models.sketchformer.config import SketchformerConfig
from models.sketchformer.positional_encoding import LearnedPositionalEncoding


class Stroke3Embedding(nn.Module):
    """Embed ``[dx, dy, pen_state]`` sequences into transformer hidden states."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.xy_projection = nn.Linear(2, config.d_model)
        self.pen_embedding = nn.Embedding(config.pen_classes, config.d_model)
        self.position = LearnedPositionalEncoding(
            config.positional_encoding.max_length,
            config.d_model,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, strokes: torch.Tensor) -> torch.Tensor:
        if strokes.ndim != 3 or strokes.shape[-1] != 3:
            raise ValueError("Expected strokes with shape (batch, sequence, 3)")

        xy = strokes[..., :2]
        pen_state = strokes[..., 2].round().long().clamp(min=0, max=self.pen_embedding.num_embeddings - 1)

        x = self.xy_projection(xy) + self.pen_embedding(pen_state)
        x = self.position(x)
        return self.dropout(x)
