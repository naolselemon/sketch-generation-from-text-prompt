"""Native PyTorch Sketchformer-style autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from models.sketchformer.config import SketchformerConfig
from models.sketchformer.decoder import LatentExpander, StrokeDecoder
from models.sketchformer.embeddings import (
    DecoderQueryEmbedding,
    Stroke3Embedding,
    TokenEmbedding,
)
from models.sketchformer.encoder import AttentionPool, StrokeEncoder
from models.sketchformer.heads import (
    ClassificationHead,
    ContinuousReconstructionHead,
    ReconstructionOutput,
    TokenReconstructionHead,
    TokenReconstructionOutput,
)


@dataclass
class SketchformerOutput:
    """Forward-pass output consumed by losses, metrics, and visualization."""

    embedding: torch.Tensor
    encoded: torch.Tensor
    decoded: torch.Tensor
    reconstruction: ReconstructionOutput | TokenReconstructionOutput | None
    class_logits: torch.Tensor | None


class SketchformerModel(nn.Module):
    """Long-sequence-capable Sketchformer-style model implemented in PyTorch."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        if self._uses_token_input:
            self.input_embedding = TokenEmbedding(config)
            self.target_embedding = DecoderQueryEmbedding(config)
        else:
            self.input_embedding = Stroke3Embedding(config)
            self.target_embedding = Stroke3Embedding(config)
        self.encoder = StrokeEncoder(config)
        self.pool = AttentionPool(config.d_model, config.latent_dim)
        self.latent_expander = LatentExpander(
            config.latent_dim,
            config.d_model,
            config.max_seq_len,
        )
        self.decoder = StrokeDecoder(config)

        self.reconstruction_head = self._build_reconstruction_head(config)
        self.classification_head = (
            ClassificationHead(config) if config.classification.enabled else None
        )

    @property
    def _uses_token_input(self) -> bool:
        return self.config.input_mode in {"tok_dict", "token", "tokens"}

    @staticmethod
    def _build_reconstruction_head(
        config: SketchformerConfig,
    ) -> ContinuousReconstructionHead | TokenReconstructionHead | None:
        if not config.reconstruction.enabled:
            return None
        if config.reconstruction.target in {"tok_dict", "token", "tokens"}:
            return TokenReconstructionHead(config)
        return ContinuousReconstructionHead(config)

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "SketchformerModel":
        return cls(SketchformerConfig.from_mapping(config))

    def forward(
        self,
        strokes: torch.Tensor | dict[str, Any],
        *,
        targets: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> SketchformerOutput:
        if isinstance(strokes, dict):
            batch = strokes
            targets = batch.get("targets", targets)
            valid_mask = batch.get("valid_mask", valid_mask)
            attention_mask = batch.get("sdpa_mask", attention_mask)
            strokes = batch["tokens"] if self._uses_token_input else batch["strokes"]

        if targets is None:
            targets = strokes
        if valid_mask is None:
            valid_mask = torch.ones(
                strokes.shape[:2],
                dtype=torch.bool,
                device=strokes.device,
            )

        encoded = self.encode(strokes, attention_mask=attention_mask)
        embedding = self.pool(encoded, valid_mask=valid_mask)
        decoded = self.decode(
            embedding,
            targets,
            self_attention_mask=attention_mask,
            valid_mask=valid_mask,
        )

        reconstruction = (
            self.reconstruction_head(decoded)
            if self.reconstruction_head is not None
            else None
        )
        class_logits = (
            self.classification_head(embedding)
            if self.classification_head is not None
            else None
        )

        return SketchformerOutput(
            embedding=embedding,
            encoded=encoded,
            decoded=decoded,
            reconstruction=reconstruction,
            class_logits=class_logits,
        )

    def encode(
        self,
        strokes: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded_input = self.input_embedding(strokes)
        return self.encoder(encoded_input, attention_mask=attention_mask)

    def decode(
        self,
        embedding: torch.Tensor,
        targets: torch.Tensor,
        *,
        self_attention_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._uses_token_input:
            target_input = self.target_embedding(
                targets.shape[0],
                targets.shape[1],
                device=embedding.device,
            )
        else:
            target_input = self.target_embedding(targets)
        memory = self.latent_expander(embedding, target_input.shape[1])
        cross_attention_mask = (
            None
            if self.config.blind_decoder_mask
            else self._cross_attention_mask(valid_mask, target_input.shape[1])
        )
        return self.decoder(
            target_input,
            memory,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
        )

    @staticmethod
    def _cross_attention_mask(
        valid_mask: torch.Tensor | None,
        sequence_length: int,
    ) -> torch.Tensor | None:
        if valid_mask is None:
            return None
        batch_size = valid_mask.shape[0]
        return valid_mask.unsqueeze(1).expand(batch_size, sequence_length, sequence_length).unsqueeze(1)
