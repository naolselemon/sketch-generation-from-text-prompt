from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from core.losses import SketchformerLoss, gaussian_mixture_nll, masked_mean


class SketchformerLossTest(unittest.TestCase):
    def test_masked_mean_ignores_padding_positions(self) -> None:
        values = torch.tensor([[1.0, 3.0, 100.0], [5.0, 100.0, 100.0]])
        mask = torch.tensor([[True, True, False], [True, False, False]])

        self.assertEqual(masked_mean(values, mask).item(), 3.0)

    def test_deterministic_reconstruction_loss_is_finite(self) -> None:
        targets = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        output = SimpleNamespace(
            reconstruction=SimpleNamespace(
                xy=torch.tensor([[[0.0, 0.0], [0.5, 1.5], [8.0, 8.0]]]),
                pen_logits=torch.tensor(
                    [[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]]
                ),
                mixture_logits=None,
            ),
            class_logits=None,
        )
        batch = {
            "targets": targets,
            "valid_mask": torch.tensor([[True, True, False]]),
        }

        loss = SketchformerLoss(
            {"reconstruction": 1.0, "pen_state": 1.0, "classification": 0.0}
        )(output, batch)

        self.assertTrue(torch.isfinite(loss.total))
        self.assertTrue(loss.reconstruction.item() > 0.0)
        self.assertTrue(loss.pen_state.item() < 0.1)
        self.assertEqual(loss.classification.item(), 0.0)

    def test_gaussian_mixture_nll_returns_per_step_values(self) -> None:
        reconstruction = SimpleNamespace(
            mixture_logits=torch.zeros((1, 2, 1)),
            mu=torch.zeros((1, 2, 1, 2)),
            log_sigma=torch.zeros((1, 2, 1, 2)),
            rho=torch.zeros((1, 2, 1)),
        )
        target_xy = torch.zeros((1, 2, 2))

        nll = gaussian_mixture_nll(reconstruction, target_xy)

        self.assertEqual(nll.shape, (1, 2))
        self.assertTrue(torch.isfinite(nll).all())


if __name__ == "__main__":
    unittest.main()
