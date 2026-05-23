"""Transformer encoder blocks using PyTorch scaled dot-product attention."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.sketchformer.config import SketchformerConfig


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"Unsupported activation: {name}")


class SDPAAttention(nn.Module):
    """Multi-head attention backed by ``scaled_dot_product_attention``."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = query.shape[0]

        q = self._split_heads(self.q_proj(query), batch_size)
        k = self._split_heads(self.k_proj(key), batch_size)
        v = self._split_heads(self.v_proj(value), batch_size)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, query.shape[1], self.d_model)
        return self.out_proj(attended)

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, x.shape[1], self.num_heads, self.head_dim)
        return x.transpose(1, 2)


class FeedForward(nn.Module):
    """Position-wise transformer feed-forward block."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.norm_first = config.norm_first
        self.self_attn = SDPAAttention(config.d_model, config.num_heads, config.dropout)
        self.ffn = FeedForward(
            config.d_model,
            config.dim_feedforward,
            config.dropout,
            config.activation,
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_first:
            norm_x = self.norm1(x)
            x = x + self.dropout1(self.self_attn(norm_x, norm_x, norm_x, attention_mask))
            x = x + self.dropout2(self.ffn(self.norm2(x)))
            return x

        attn = self.self_attn(x, x, x, attention_mask)
        x = self.norm1(x + self.dropout1(attn))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class StrokeEncoder(nn.Module):
    """Stack of SDPA transformer encoder blocks."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.layers = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_encoder_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)
        return self.final_norm(x)


class AttentionPool(nn.Module):
    """Mask-aware attention pooling from sequence states to one latent vector."""

    def __init__(self, d_model: int, latent_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)
        self.projection = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.projection(pooled)
