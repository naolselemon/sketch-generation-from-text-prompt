from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from dataloaders.masks import build_sequence_masks
from models.sketchformer.config import (
    PositionalEncodingConfig,
    ReconstructionHeadConfig,
    SketchformerConfig,
    TokenDictionaryConfig,
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

    def test_tiny_tok_dict_model_forward_returns_token_logits(self) -> None:
        config = SketchformerConfig(
            name="sketchformer_tok_dict",
            input_mode="tok_dict",
            max_seq_len=16,
            token_dictionary=TokenDictionaryConfig(
                codebook_size=4,
                motion_token_offset=1,
                pad_token_id=0,
                sep_token_id=5,
                sos_token_id=6,
                eos_token_id=7,
                vocab_size=8,
            ),
            d_model=16,
            latent_dim=16,
            pool_hidden_dim=8,
            pooling_mode="tf_self_attn_v1",
            latent_expander_mode="tf_dense",
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            dim_feedforward=32,
            dropout=0.0,
            activation="relu",
            norm_first=False,
            use_final_norm=False,
            gradient_checkpointing=False,
            positional_encoding=PositionalEncodingConfig(type="sinusoidal", max_length=16),
            decoder_autoregressive=True,
            reconstruction=ReconstructionHeadConfig(
                enabled=True,
                target="tok_dict",
            ),
        )
        model = SketchformerModel(config)
        model.eval()

        tokens = torch.tensor(
            [
                [6, 1, 5, 7, 0, 0],
                [6, 2, 5, 3, 7, 0],
            ],
            dtype=torch.long,
        )
        masks = build_sequence_masks([4, 5], max_length=6)
        with patch.object(
            model.latent_expander,
            "forward",
            wraps=model.latent_expander.forward,
        ) as expand_forward, torch.no_grad():
            output = model({"tokens": tokens, "targets": tokens.clone(), **masks})

        self.assertEqual(output.embedding.shape, (2, 16))
        self.assertIsNotNone(output.reconstruction)
        assert output.reconstruction is not None
        self.assertEqual(output.reconstruction.token_logits.shape, (2, 5, 8))
        self.assertEqual(expand_forward.call_args.args[1], 16)
        self.assertEqual(output.loss_targets.shape, (2, 5))
        self.assertIsNone(output.class_logits)


if __name__ == "__main__":
    unittest.main()
