"""Sketch token dictionary builder.

Builds a discrete motion vocabulary by clustering the (Δx, Δy) displacement
pairs from a corpus of stroke-5 arrays using MiniBatchKMeans.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from utils.paths import DEFAULT_SKETCH_TOKEN_DIR, DEFAULT_STROKE5_DIR


# Core builder

def build_codebook(
    stroke5_arrays: list[np.ndarray],
    K: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster (Δx, Δy) pairs from stroke-5 arrays into K centroids."""
    drawing_deltas: list[np.ndarray] = []

    for s5 in stroke5_arrays:
        mask = s5[:, 2] == 1.0   # p1 == 1  →  pen drawing
        subset = s5[mask, :2]
        if len(subset) > 0:
            drawing_deltas.append(subset)

    if not drawing_deltas:
        raise ValueError(
            "No drawing-stroke samples found in the provided stroke-5 arrays. "
            "Cannot build a codebook from empty data."
        )

    deltas = np.concatenate(drawing_deltas, axis=0)

    # Reduce K if we have fewer samples than requested clusters.
    K_actual = min(K, len(deltas))
    if K_actual < K:
        print(
            f"[sketch_token] Warning: only {len(deltas)} samples available — "
            f"reducing K from {K} to {K_actual}."
        )

    kmeans = MiniBatchKMeans(
        n_clusters=K_actual,
        random_state=random_state,
        n_init=3,
        batch_size=max(1024, K_actual),
        verbose=0,
    )
    kmeans.fit(deltas)

    return kmeans.cluster_centers_.astype(np.float32)


# Persistence helpers

def save_codebook(
    codebook: np.ndarray,
    output_dir: Path,
    K: int,
    n_samples: int,
) -> tuple[Path, Path]:
    """Persist the codebook array and a companion metadata JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path  = output_dir / "codebook.npy"
    meta_path = output_dir / "metadata.json"

    np.save(npy_path, codebook)

    metadata = {
        "K": K,
        "n_samples": n_samples,
        "codebook_shape": list(codebook.shape),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    return npy_path, meta_path


def load_codebook_from_dir(sketch_token_dir: Path) -> tuple[np.ndarray, dict]:
    """Load a saved codebook and its metadata from *sketch_token_dir*.

    Returns
    -------
    (codebook, metadata) tuple.
    """
    sketch_token_dir = Path(sketch_token_dir)
    codebook = np.load(sketch_token_dir / "codebook.npy")
    with open(sketch_token_dir / "metadata.json") as fh:
        metadata = json.load(fh)
    return codebook, metadata


def load_stroke5_arrays(source_dir: Path) -> list[np.ndarray]:
    """Load all stroke-5 arrays from a directory of one-array ``.npz`` files."""

    arrays: list[np.ndarray] = []
    for path in sorted(Path(source_dir).glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        if len(data.files) != 1:
            raise ValueError(f"Expected one array in {path}, found {data.files}")
        array = np.asarray(data[data.files[0]], dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 5:
            raise ValueError(f"Expected stroke5 shape (N, 5) in {path}, got {array.shape}")
        arrays.append(array)
    if not arrays:
        raise FileNotFoundError(f"No .npz stroke5 files found in {source_dir}")
    return arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sketch tok-dict codebook.")
    parser.add_argument("--source-dir", default=DEFAULT_STROKE5_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_SKETCH_TOKEN_DIR)
    parser.add_argument("--K", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stroke5_arrays = load_stroke5_arrays(Path(args.source_dir))
    codebook = build_codebook(stroke5_arrays, K=args.K, random_state=args.seed)
    npy_path, meta_path = save_codebook(
        codebook,
        Path(args.output_dir),
        K=int(args.K),
        n_samples=len(stroke5_arrays),
    )
    print(f"Saved codebook: {npy_path}")
    print(f"Saved metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
