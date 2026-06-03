from __future__ import annotations

import unittest

import numpy as np

from prep_data.prepare_sketchformer_tokens import truncate_tokens


class PrepareSketchformerTokensTest(unittest.TestCase):
    def test_truncate_tokens_preserves_eos_at_max_length(self) -> None:
        tokens = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)

        truncated = truncate_tokens(tokens, max_length=4, eos_token_id=99)

        self.assertEqual(truncated.tolist(), [0, 1, 2, 99])


if __name__ == "__main__":
    unittest.main()
