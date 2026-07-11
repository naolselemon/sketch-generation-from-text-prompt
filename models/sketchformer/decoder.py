"""Transformer decoder blocks for Sketchformer reconstruction."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.sketchformer.config import SketchformerConfig
from models.sketchformer.encoder import FeedForward, SDPAAttention


class LatentExpander(nn.Module):
    """Expand a pooled latent vector into decoder memory states."""

    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        max_length: int,
        *,
        mode: str = "projected_position",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.max_length = int(max_length)
        self.output_dim = d_model if mode == "projected_position" else latent_dim
        if mode == "projected_position":
            self.latent_projection = nn.Linear(latent_dim, d_model)
            self.position = nn.Embedding(max_length, d_model)
        elif mode == "tf_dense":
            self.expand_layer = nn.Linear(1, max_length)
        else:
            raise ValueError("mode must be one of: projected_position, tf_dense")

    def forward(self, latent: torch.Tensor, sequence_length: int) -> torch.Tensor:
        if sequence_length > self.max_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds latent expander capacity {self.max_length}"
            )
        if self.mode == "projected_position":
            positions = torch.arange(sequence_length, device=latent.device)
            memory = self.latent_projection(latent).unsqueeze(1)
            return memory + self.position(positions).unsqueeze(0)

        expanded = self.expand_layer(latent.unsqueeze(2)).transpose(1, 2)
        return expanded[:, :sequence_length, :]


class DecoderBlock(nn.Module):
    """Pre-norm transformer decoder block with self and cross attention."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.norm_first = config.norm_first
        memory_dim = config.pool_output_dim if config.latent_expander_mode == "tf_dense" else config.d_model
        self.self_attn = SDPAAttention(config.d_model, config.num_heads, config.dropout)
        self.cross_attn = SDPAAttention(
            config.d_model,
            config.num_heads,
            config.dropout,
            key_dim=memory_dim,
            value_dim=memory_dim,
        )
        self.ffn = FeedForward(
            config.d_model,
            config.dim_feedforward,
            config.dropout,
            config.activation,
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_first:
            norm_x = self.norm1(x)
            x = x + self.dropout1(
                self.self_attn(norm_x, norm_x, norm_x, self_attention_mask)
            )
            x = x + self.dropout2(
                self.cross_attn(self.norm2(x), memory, memory, cross_attention_mask)
            )
            x = x + self.dropout3(self.ffn(self.norm3(x)))
            return x

        self_attn = self.self_attn(x, x, x, self_attention_mask)
        x = self.norm1(x + self.dropout1(self_attn))
        cross_attn = self.cross_attn(x, memory, memory, cross_attention_mask)
        x = self.norm2(x + self.dropout2(cross_attn))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class StrokeDecoder(nn.Module):
    """Stack of SDPA transformer decoder blocks."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_decoder_layers)]
        )
        self.final_norm = (
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
            if config.use_final_norm
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(
                    layer,
                    x,
                    memory,
                    self_attention_mask,
                    cross_attention_mask,
                    use_reentrant=False,
                )
            else:
                x = layer(x, memory, self_attention_mask, cross_attention_mask)
        return self.final_norm(x)
