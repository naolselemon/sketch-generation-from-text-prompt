from __future__ import annotations

import unittest

import torch

from dataloaders.masks import build_sequence_masks
from models.sketchformer.config import (
    PositionalEncodingConfig,
    ReconstructionHeadConfig,
    SketchformerConfig,
)
from models.sketchformer.model import SketchformerModel


class SketchformerForwardTest(unittest.TestCase):
    def test_tiny_model_forward_returns_reconstruction_shapes(self) -> None:
        config = SketchformerConfig(
            max_seq_len=16,
            d_model=16,
            latent_dim=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            dim_feedforward=32,
            dropout=0.0,
            gradient_checkpointing=False,
            positional_encoding=PositionalEncodingConfig(max_length=16),
            reconstruction=ReconstructionHeadConfig(
                enabled=True,
                xy_distribution="deterministic",
                num_mixtures=1,
            ),
        )
        model = SketchformerModel(config)
        model.eval()

        strokes = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.1, 0.2, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.1, 0.1, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        masks = build_sequence_masks([3, 2], max_length=4)
        with torch.no_grad():
            output = model({"strokes": strokes, "targets": strokes.clone(), **masks})

        self.assertEqual(output.embedding.shape, (2, 8))
        self.assertIsNotNone(output.reconstruction)
        assert output.reconstruction is not None
        self.assertEqual(output.reconstruction.xy.shape, (2, 4, 2))
        self.assertEqual(output.reconstruction.pen_logits.shape, (2, 4, 3))
        self.assertIsNone(output.class_logits)


if __name__ == "__main__":
    unittest.main()
