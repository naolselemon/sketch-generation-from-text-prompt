"""Sketch token encoding helpers.

Encodes a stroke-5 array into a sequence of discrete token indices using a
K-means codebook built by ``prep_data.sketch_token.create_token_dict``.

Special tokens match the released TensorFlow Sketchformer tokenizer:
    0       → padding
    1..K    → codebook motion tokens
    K + 1   → stroke separator / pen lift
    K + 2   → start of sketch
    K + 3   → end of sketch
"""

from __future__ import annotations

import numpy as np


def encode_stroke5(
    stroke5: np.ndarray,
    codebook: np.ndarray,
    *,
    motion_token_offset: int = 1,
    sep_token_id: int | None = None,
    sos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> np.ndarray:
    """Map each stroke-5 row to a discrete token index.

    Pen-lift points (p2=1) emit two tokens: a shifted codebook motion token
    preserving the (dx, dy) displacement, followed by SEP.
    """
    K = len(codebook)
    sep = K + 1 if sep_token_id is None else int(sep_token_id)
    sos = K + 2 if sos_token_id is None else int(sos_token_id)
    eos = K + 3 if eos_token_id is None else int(eos_token_id)
    tokens: list[int] = [sos]

    for row in stroke5:
        p3 = row[4]
        p2 = row[3]

        if p3 == 1.0:
            tokens.append(eos)
            break
        else:
            # Quantize (dx, dy) for ALL points, including pen-lift.
            delta = codebook - row[:2]
            motion_token = int(np.argmin((delta * delta).sum(axis=1))) + int(motion_token_offset)
            tokens.append(motion_token)

            if p2 == 1.0:
                tokens.append(sep)

    if tokens[-1] != eos:
        tokens.append(eos)

    return np.array(tokens, dtype=np.int32)


def decode_tokens(
    tokens: np.ndarray,
    codebook: np.ndarray,
    *,
    motion_token_offset: int = 1,
    pad_token_id: int = 0,
    sep_token_id: int | None = None,
    sos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> np.ndarray:
    """Approximate inverse of ``encode_stroke5``.

    Handles the TensorFlow-compatible token IDs produced by ``encode_stroke5``.
    """
    K = len(codebook)
    sep = K + 1 if sep_token_id is None else int(sep_token_id)
    sos = K + 2 if sos_token_id is None else int(sos_token_id)
    eos = K + 3 if eos_token_id is None else int(eos_token_id)
    motion_start = int(motion_token_offset)
    motion_end = motion_start + K
    rows: list[list[float]] = []

    for tok in tokens:
        tok = int(tok)
        if tok in {int(pad_token_id), sos}:
            continue
        if tok == eos:
            rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
            break
        if tok == sep:
            if rows:
                rows[-1][2] = 0.0
                rows[-1][3] = 1.0
        elif motion_start <= tok < motion_end:
            dx, dy = codebook[tok - motion_start]
            rows.append([float(dx), float(dy), 1.0, 0.0, 0.0])

    if not rows or rows[-1][4] != 1.0:
        rows.append([0.0, 0.0, 0.0, 0.0, 1.0])

    return np.array(rows, dtype=np.float32)
