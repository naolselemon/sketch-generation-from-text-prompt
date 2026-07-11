from __future__ import annotations

import unittest

import numpy as np

from prep_data.prepare_sketchformer_tokens import truncate_tokens
from utils.tokenizer import decode_tokens, encode_stroke5


class PrepareSketchformerTokensTest(unittest.TestCase):
    def test_truncate_tokens_preserves_eos_at_max_length(self) -> None:
        tokens = np.asarray([1002, 1, 2, 3, 4, 1003], dtype=np.int64)

        truncated = truncate_tokens(
            tokens,
            max_length=4,
            sep_token_id=1001,
            eos_token_id=1003,
        )

        self.assertEqual(truncated.tolist(), [1002, 1, 1001, 1003])

    def test_tokenizer_uses_original_tensorflow_token_layout(self) -> None:
        codebook = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        stroke5 = np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        tokens = encode_stroke5(stroke5, codebook)
        decoded = decode_tokens(tokens, codebook)

        self.assertEqual(tokens.tolist(), [4, 1, 2, 3, 5])
        self.assertEqual(decoded[-1].tolist(), [0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(decoded[1].tolist(), [1.0, 0.0, 0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
