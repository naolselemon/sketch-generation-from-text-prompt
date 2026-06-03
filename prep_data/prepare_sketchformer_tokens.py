"""Prepare tok-dict chunks for native Sketchformer fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prep_data.prepare_sketchformer import build_splits, chunk_sketch_data
from prep_data.sketch_token.create_token_dict import load_codebook_from_dir
from utils.paths import DEFAULT_SKETCH_TOKEN_DIR, DEFAULT_STROKE5_DIR, PROCESSED_DATA_DIR
from utils.tokenizer import encode_stroke5


def load_stroke5_file(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if len(data.files) != 1:
        raise ValueError(f"Expected one array in {path}, found {data.files}")
    return np.asarray(data[data.files[0]], dtype=np.float32)


def truncate_tokens(
    tokens: np.ndarray,
    *,
    max_length: int | None,
    eos_token_id: int,
) -> np.ndarray:
    sequence = np.asarray(tokens, dtype=np.int64)
    if max_length is None or len(sequence) <= max_length:
        return sequence
    if max_length <= 0:
        raise ValueError("--max-length must be positive")

    truncated = np.array(sequence[:max_length], copy=True, dtype=np.int64)
    truncated[-1] = int(eos_token_id)
    return truncated


def save_chunk(path: Path, token_sequences: list[np.ndarray], labels: list[int]) -> None:
    np.savez_compressed(
        path,
        x=np.asarray(token_sequences, dtype=object),
        y=np.asarray(labels, dtype=np.int32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare anime stroke5 data as tok-dict Sketchformer chunks."
    )
    parser.add_argument("--source-dir", default=DEFAULT_STROKE5_DIR)
    parser.add_argument(
        "--token-dict-dir",
        default=DEFAULT_SKETCH_TOKEN_DIR,
        help="Directory containing codebook.npy and metadata.json.",
    )
    parser.add_argument(
        "--target-dir",
        default=PROCESSED_DATA_DIR / "sketchformer-ready-data" / "tok-dict",
    )
    parser.add_argument("--n-chunks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--n-classes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_chunks <= 0:
        raise ValueError("--n-chunks must be positive")

    source_path = Path(args.source_dir)
    target_path = Path(args.target_dir)
    token_dict_dir = Path(args.token_dict_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    codebook, codebook_metadata = load_codebook_from_dir(token_dict_dir)
    codebook_size = int(len(codebook))
    sep_token_id = codebook_size
    eos_token_id = codebook_size + 1
    pad_token_id = codebook_size + 2
    vocab_size = codebook_size + 3

    file_list = sorted(source_path.glob("*.npz"))
    if not file_list:
        raise FileNotFoundError(f"No .npz files found in {source_path}")

    token_sequences = [
        truncate_tokens(
            encode_stroke5(load_stroke5_file(path), codebook),
            max_length=args.max_length,
            eos_token_id=eos_token_id,
        )
        for path in file_list
    ]

    train_tokens, valid_tokens, test_tokens = build_splits(
        token_sequences,
        seed=args.seed,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
    )

    train_labels = [0] * len(train_tokens)
    valid_labels = [0] * len(valid_tokens)
    test_labels = [0] * len(test_tokens)

    for chunk_idx, (chunk_tokens, chunk_labels) in enumerate(
        chunk_sketch_data(train_tokens, train_labels, args.n_chunks)
    ):
        save_chunk(target_path / f"train_{chunk_idx:03}.npz", chunk_tokens, chunk_labels)

    save_chunk(target_path / "valid.npz", valid_tokens, valid_labels)
    save_chunk(target_path / "test.npz", test_tokens, test_labels)

    np.savez(
        target_path / "meta.npz",
        format="tok_dict",
        codebook_size=codebook_size,
        sep_token_id=sep_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
        codebook_metadata=np.asarray(codebook_metadata, dtype=object),
        class_names=np.asarray(["anime"], dtype=object),
        n_classes=int(args.n_classes),
        n_samples_train=len(train_tokens),
        n_samples_valid=len(valid_tokens),
        n_samples_test=len(test_tokens),
    )
    print(f"Saved tok-dict Sketchformer data to {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
