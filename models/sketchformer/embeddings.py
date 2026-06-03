"""Input embeddings for stroke3 and token-dictionary data."""

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


class TokenEmbedding(nn.Module):
    """Embed discrete tok-dict sequences into transformer hidden states."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        token_config = config.token_dictionary
        self.token_embedding = nn.Embedding(
            token_config.vocab_size,
            config.d_model,
            padding_idx=token_config.pad_token_id,
        )
        self.position = LearnedPositionalEncoding(
            config.positional_encoding.max_length,
            config.d_model,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("Expected tokens with shape (batch, sequence)")

        x = self.token_embedding(tokens.long())
        x = self.position(x)
        return self.dropout(x)


class DecoderQueryEmbedding(nn.Module):
    """Learned decoder queries that avoid leaking target token identities."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.position = nn.Embedding(
            config.positional_encoding.max_length,
            config.d_model,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=device)
        x = self.position(positions).unsqueeze(0).expand(batch_size, -1, -1)
        return self.dropout(x)
