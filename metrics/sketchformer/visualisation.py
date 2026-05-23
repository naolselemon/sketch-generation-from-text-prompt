"""Qualitative reconstruction plots for native Sketchformer evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from metrics.sketchformer.reconstruction import ReconstructionExample

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stroke3_to_points(stroke3: np.ndarray) -> np.ndarray:
    """Convert relative stroke3 deltas into absolute xy points."""

    array = np.asarray(stroke3, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected stroke3 array with shape (N, 3), got {array.shape}")
    return np.cumsum(array[:, :2], axis=0)


def _plot_stroke3(ax: plt.Axes, stroke3: np.ndarray, title: str) -> None:
    points = stroke3_to_points(stroke3)
    pen_lift = np.asarray(stroke3[:, 2] >= 0.5, dtype=bool)

    if len(points) == 0:
        ax.text(0.5, 0.5, "empty", ha="center", va="center")
    else:
        start = 0
        for index, lifted in enumerate(pen_lift):
            if lifted:
                _plot_segment(ax, points[start : index + 1])
                start = index + 1
        _plot_segment(ax, points[start:])

        ax.scatter(points[0, 0], points[0, 1], s=10, color="#2f80ed", zorder=3)
        ax.scatter(points[-1, 0], points[-1, 1], s=10, color="#eb5757", zorder=3)

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.axis("off")


def _plot_segment(ax: plt.Axes, points: np.ndarray) -> None:
    if len(points) == 1:
        ax.scatter(points[0, 0], points[0, 1], s=4, color="#333333", alpha=0.7)
    elif len(points) > 1:
        ax.plot(points[:, 0], points[:, 1], linewidth=0.8, color="#222222", alpha=0.85)


def save_reconstruction_pair(
    example: ReconstructionExample,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Save target and predicted sketches side by side."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    if title:
        fig.suptitle(title, fontsize=10)
    _plot_stroke3(axes[0], example.target, "target")
    _plot_stroke3(axes[1], example.prediction, "prediction")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_reconstruction_examples(
    examples: list[ReconstructionExample],
    output_dir: str | Path,
    *,
    prefix: str = "reconstruction",
) -> list[Path]:
    """Save a numbered plot for each reconstruction example."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for index, example in enumerate(examples, start=1):
        title = (
            f"source_index={example.source_index} "
            f"length={example.length} label={example.label}"
        )
        output_path = directory / f"{prefix}_{index:03d}.png"
        saved.append(save_reconstruction_pair(example, output_path, title=title))
    return saved
