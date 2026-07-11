from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from metrics.sketchformer.reconstruction import ReconstructionExample
from metrics.sketchformer.visualisation import save_reconstruction_pair, stroke3_to_points


class SketchformerVisualisationTest(unittest.TestCase):
    def test_stroke3_to_points_accepts_decoded_stroke5(self) -> None:
        stroke5 = np.asarray(
            [
                [1.0, 2.0, 1.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        points = stroke3_to_points(stroke5)

        np.testing.assert_allclose(
            points,
            np.asarray(
                [
                    [1.0, 2.0],
                    [4.0, 6.0],
                    [4.0, 6.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_save_reconstruction_pair_accepts_decoded_stroke5(self) -> None:
        stroke5 = np.asarray(
            [
                [1.0, 2.0, 1.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        example = ReconstructionExample(
            target=stroke5,
            prediction=stroke5,
            length=len(stroke5),
            source_file="sample.npz",
            source_index=0,
            label=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pair.png"
            saved = save_reconstruction_pair(example, output_path)

            self.assertEqual(saved, output_path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
