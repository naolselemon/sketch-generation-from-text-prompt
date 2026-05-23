from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from models.sketchformer.checkpoint_mapping import (
    load_torch_checkpoint,
    resize_learned_position_embedding,
)
from models.sketchformer.pretrained import inspect_tensorflow_checkpoint


class CheckpointMappingTest(unittest.TestCase):
    def test_resize_learned_position_embedding_changes_length_only(self) -> None:
        source = torch.arange(12, dtype=torch.float32).view(3, 4)

        resized = resize_learned_position_embedding(source, target_length=6)

        self.assertEqual(resized.shape, (6, 4))
        self.assertEqual(resized.dtype, source.dtype)

    def test_load_torch_checkpoint_reports_missing_and_unexpected_keys(self) -> None:
        model = nn.Linear(2, 2)
        checkpoint = {
            "state_dict": {
                "weight": torch.ones_like(model.weight),
                "unexpected.weight": torch.zeros(1),
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save(checkpoint, path)

            report = load_torch_checkpoint(model, path, strict=False)

        self.assertIn("bias", report.missing_keys)
        self.assertIn("unexpected.weight", report.unexpected_keys)

    def test_tensorflow_checkpoint_inspection_accepts_index_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "ckpt-1"
            prefix.with_suffix(".index").write_text("", encoding="utf-8")
            (Path(tmp) / "ckpt-1.data-00000-of-00001").write_text("", encoding="utf-8")

            checkpoint = inspect_tensorflow_checkpoint(prefix.with_suffix(".index"))

            self.assertEqual(checkpoint.prefix, prefix)
            self.assertTrue(checkpoint.exists)
            self.assertEqual(len(checkpoint.data_files), 1)


if __name__ == "__main__":
    unittest.main()
