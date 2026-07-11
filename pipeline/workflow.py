"""Core workflows for legacy and faithful long-sequence preprocessing."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from pipeline.kinematics import generate_kinematics
from pipeline.ordering import (
    order_continuity_greedy,
    order_directional_bias,
    order_greedy_nearest_neighbor,
    order_tsp,
)
from pipeline.stroke5 import stroke5_to_canvas_strokes, strokes_to_stroke5, to_stroke5
from pipeline.vectorization import (
    DEFAULT_MAX_GEOMETRY_ERROR,
    DEFAULT_RDP_EPSILON,
    centerline_metrics,
    rasterize_strokes,
    read_grayscale_image,
    source_centerline,
    vectorize_image,
    vectorize_image_with_stats,
)
from prep_data.sketch_token.create_token_dict import build_codebook, save_codebook
from utils.io import save_stroke5, save_token_sequence
from utils.tokenizer import (
    ErrorFeedbackQuantizer,
    decode_tokens,
    encode_stroke5,
    quantization_metrics,
)

_NORMALIZATION_EXTENTS = (2.5, 2.0, 1.5, 1.0)
_TARGET_TOKEN_GEOMETRY_F1 = 0.95

_ORDER_FN_MAP = {
    "continuity": order_continuity_greedy,
    "directional": order_directional_bias,
    "greedy": order_greedy_nearest_neighbor,
    "tsp": order_tsp,
}


def run_pipeline(
    sketches_dir: Path,
    stroke5_dir: Path,
    sketch_token_dir: Path,
    n_sketches: int,
    ordering: str,
    rdp_epsilon: float = DEFAULT_RDP_EPSILON,
    codebook_K: int = 1000,
    seed: int = 42,
    *,
    vectorizer: str = "contour",
    threshold_profile: str = "hysteresis",
    max_token_length: int | None = None,
    max_geometry_error: float = DEFAULT_MAX_GEOMETRY_ERROR,
    token_dict_dir: Path | None = None,
    manifest_path: Path | None = None,
    fail_on_overlength: bool = False,
    extractor_name: str | None = None,
) -> None:
    """Run V1 preprocessing or the checkpoint-compatible V2 centerline path."""

    all_sketches = sorted(Path(sketches_dir).rglob("*.png"))
    if not all_sketches:
        raise FileNotFoundError(
            f"No sketches found in {sketches_dir}. Run the filter-sketches command first."
        )
    if ordering not in _ORDER_FN_MAP:
        valid = ", ".join(sorted(_ORDER_FN_MAP))
        raise ValueError(f"Unknown ordering {ordering!r}. Expected one of: {valid}.")
    if vectorizer not in {"contour", "centerline"}:
        raise ValueError("vectorizer must be one of: contour, centerline")

    count = min(int(n_sketches), len(all_sketches))
    samples = random.Random(seed).sample(all_sketches, count)
    if vectorizer == "centerline":
        _run_faithful_pipeline(
            samples=samples,
            sketches_dir=Path(sketches_dir),
            stroke5_dir=Path(stroke5_dir),
            token_dict_dir=Path(token_dict_dir or sketch_token_dir),
            order_fn=_ORDER_FN_MAP[ordering],
            ordering=ordering,
            rdp_epsilon=float(rdp_epsilon),
            threshold_profile=threshold_profile,
            max_token_length=int(max_token_length or 4096),
            max_geometry_error=float(max_geometry_error),
            manifest_path=Path(manifest_path) if manifest_path else None,
            fail_on_overlength=fail_on_overlength,
            extractor_name=extractor_name or Path(sketches_dir).name,
        )
        return

    _run_legacy_pipeline(
        samples=samples,
        stroke5_dir=Path(stroke5_dir),
        sketch_token_dir=Path(sketch_token_dir),
        order_fn=_ORDER_FN_MAP[ordering],
        ordering=ordering,
        rdp_epsilon=float(rdp_epsilon),
        codebook_K=int(codebook_K),
    )


def _run_faithful_pipeline(
    *,
    samples: list[Path],
    sketches_dir: Path,
    stroke5_dir: Path,
    token_dict_dir: Path,
    order_fn,
    ordering: str,
    rdp_epsilon: float,
    threshold_profile: str,
    max_token_length: int,
    max_geometry_error: float,
    manifest_path: Path | None,
    fail_on_overlength: bool,
    extractor_name: str,
) -> None:
    if max_token_length < 4:
        raise ValueError("max_token_length must be at least 4")
    if max_geometry_error < rdp_epsilon:
        raise ValueError("max_geometry_error must be >= rdp_epsilon")

    codebook_path = token_dict_dir / "codebook.npy"
    if not codebook_path.exists():
        raise FileNotFoundError(
            f"Released token dictionary not found: {codebook_path}. "
            "Run tts-create-sketch-token-dict first."
        )
    codebook = np.asarray(np.load(codebook_path), dtype=np.float32)
    if codebook.ndim != 2 or codebook.shape[1] != 2:
        raise ValueError(f"Expected codebook shape (K, 2), got {codebook.shape}")
    quantizer = ErrorFeedbackQuantizer(codebook)
    codebook_sha256 = hashlib.sha256(codebook_path.read_bytes()).hexdigest()

    token_dir = stroke5_dir.parent / "tokens-v2"
    resolved_manifest = manifest_path or stroke5_dir.parent / "v2_manifest.jsonl"
    records: list[dict[str, Any]] = []
    accepted = overlength = failed = 0

    print(
        f"[pipeline-v2] sketches={len(samples)} vectorizer=centerline "
        f"ordering={ordering} max_tokens={max_token_length}"
    )
    print(f"[pipeline-v2] token_dictionary={codebook_path}")

    for image_path in tqdm(samples, desc="Faithful V2", unit="sketch"):
        relative = image_path.relative_to(sketches_dir)
        base_record: dict[str, Any] = {
            "schema_version": 2,
            "sample_id": relative.with_suffix("").as_posix(),
            "source_path": str(image_path),
            "source_relative_path": relative.as_posix(),
            "extractor": extractor_name,
            "vectorizer": "centerline",
            "threshold_profile": threshold_profile,
            "ordering": ordering,
            "max_token_length": max_token_length,
            "token_dictionary_path": str(codebook_path),
            "token_dictionary_size": len(codebook),
            "token_dictionary_sha256": codebook_sha256,
            "quantizer": "error_feedback_pair_search_v2",
        }
        try:
            result = fit_faithful_sequence(
                image_path=image_path,
                codebook=codebook,
                quantizer=quantizer,
                order_fn=order_fn,
                initial_epsilon=rdp_epsilon,
                max_epsilon=max_geometry_error,
                threshold_profile=threshold_profile,
                max_token_length=max_token_length,
            )
            base_record.update(result["record"])
            if not result["accepted"]:
                overlength += 1
                records.append(base_record)
                continue

            stroke5 = result["stroke5"]
            tokens = result["tokens"]
            output_relative = relative.with_suffix(".npz")
            stroke_path = stroke5_dir / output_relative
            token_path = token_dir / output_relative
            save_stroke5(stroke5, stroke_path)
            save_token_sequence(tokens, token_path)
            base_record.update(
                {
                    "status": "accepted",
                    "stroke5_path": str(stroke_path),
                    "tokens_path": str(token_path),
                }
            )
            records.append(base_record)
            accepted += 1
        except Exception as exc:
            failed += 1
            base_record.update({"status": "error", "rejection_reason": str(exc)})
            records.append(base_record)
            tqdm.write(f"[pipeline-v2] error {image_path.name}: {exc}")

    _write_manifest(resolved_manifest, records)
    print(
        f"[pipeline-v2] accepted={accepted} overlength={overlength} errors={failed} "
        f"manifest={resolved_manifest}"
    )
    if fail_on_overlength and overlength:
        raise RuntimeError(
            f"{overlength} sketches exceeded {max_token_length} tokens within the "
            f"{max_geometry_error:g}px geometry limit; see {resolved_manifest}"
        )
    if accepted == 0:
        raise RuntimeError(f"No sketches were accepted; see {resolved_manifest}")


def fit_faithful_sequence(
    *,
    image_path: Path,
    codebook: np.ndarray,
    quantizer: ErrorFeedbackQuantizer,
    order_fn,
    initial_epsilon: float,
    max_epsilon: float,
    threshold_profile: str,
    max_token_length: int,
) -> dict[str, Any]:
    image = read_grayscale_image(image_path)
    reference = source_centerline(image, threshold_profile=threshold_profile)
    epsilon_values = _epsilon_schedule(initial_epsilon, max_epsilon)
    best_feasible: dict[str, Any] | None = None
    best_overlength: dict[str, Any] | None = None

    for epsilon in epsilon_values:
        strokes, stats = vectorize_image_with_stats(
            image_path,
            epsilon=epsilon,
            method="centerline",
            threshold_profile=threshold_profile,
        )
        if not strokes:
            raise ValueError("centerline vectorization produced no strokes")
        ordered = order_fn(strokes)
        rendered = rasterize_strokes(ordered, image.shape)
        vector_geometry = centerline_metrics(reference, rendered, tolerance_px=2.0)
        candidates = [
            _encode_geometry_candidate(
                ordered=ordered,
                image_shape=image.shape,
                reference=reference,
                codebook=codebook,
                quantizer=quantizer,
                normalization_extent=extent,
                epsilon=epsilon,
                stats=stats,
                vector_geometry=vector_geometry,
            )
            for extent in _NORMALIZATION_EXTENTS
        ]
        feasible = [item for item in candidates if len(item["tokens"]) <= max_token_length]
        if feasible:
            selected = max(
                feasible,
                key=lambda item: (
                    float(item["record"]["geometry_f1_2px"]),
                    -float(item["record"]["geometry_chamfer_px"]),
                    -len(item["tokens"]),
                ),
            )
            if (
                best_feasible is None
                or float(selected["record"]["geometry_f1_2px"])
                > float(best_feasible["record"]["geometry_f1_2px"])
            ):
                best_feasible = selected
            if float(selected["record"]["geometry_f1_2px"]) >= _TARGET_TOKEN_GEOMETRY_F1:
                selected["accepted"] = True
                return selected

        shortest = min(
            candidates,
            key=lambda item: (
                len(item["tokens"]),
                -float(item["record"]["geometry_f1_2px"]),
            ),
        )
        if best_overlength is None or len(shortest["tokens"]) < len(best_overlength["tokens"]):
            best_overlength = shortest

    if best_feasible is not None:
        best_feasible["accepted"] = True
        best_feasible["record"]["quality_warning"] = "below_target_token_geometry_f1"
        return best_feasible

    assert best_overlength is not None
    best_overlength["accepted"] = False
    best_overlength["record"].update(
        {
            "status": "rejected",
            "rejection_reason": "overlength_after_max_geometry_error",
        }
    )
    return best_overlength


def _encode_geometry_candidate(
    *,
    ordered: list[list[tuple[int, int]]],
    image_shape: tuple[int, int],
    reference: np.ndarray,
    codebook: np.ndarray,
    quantizer: ErrorFeedbackQuantizer,
    normalization_extent: float,
    epsilon: float,
    stats,
    vector_geometry,
) -> dict[str, Any]:
    stroke5, transform = strokes_to_stroke5(
        ordered,
        canvas_shape=image_shape,
        normalization_extent=normalization_extent,
    )
    tokens = quantizer.encode(stroke5)
    decoded_strokes = stroke5_to_canvas_strokes(decode_tokens(tokens, codebook), transform)
    decoded_render = rasterize_strokes(decoded_strokes, image_shape)
    token_geometry = centerline_metrics(reference, decoded_render, tolerance_px=2.0)
    quantization = quantization_metrics(stroke5, tokens, codebook)
    return {
        "stroke5": stroke5,
        "tokens": tokens,
        "record": {
            "rdp_epsilon": float(epsilon),
            "normalization_extent": float(normalization_extent),
            "token_length": int(len(tokens)),
            "raw_stroke_count": stats.raw_stroke_count,
            "raw_point_count": stats.raw_point_count,
            "pre_order_stroke_count": stats.simplified_stroke_count,
            "pre_order_point_count": stats.simplified_point_count,
            "stroke_count": len(ordered),
            "point_count": sum(len(stroke) for stroke in ordered),
            "geometry_f1_2px": token_geometry.f1,
            "geometry_chamfer_px": token_geometry.symmetric_chamfer,
            "vector_geometry_f1_2px": vector_geometry.f1,
            "vector_geometry_chamfer_px": vector_geometry.symmetric_chamfer,
            "quantization_mean_error": quantization.mean_point_error,
            "quantization_endpoint_error": quantization.endpoint_error,
            "transform": asdict(transform),
        },
    }


def _epsilon_schedule(initial: float, maximum: float) -> list[float]:
    values = [float(initial)]
    current = max(initial, 0.25)
    while current < maximum:
        current = min(maximum, current + 0.25)
        if current > values[-1]:
            values.append(float(current))
    return values


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _run_legacy_pipeline(
    *,
    samples: list[Path],
    stroke5_dir: Path,
    sketch_token_dir: Path,
    order_fn,
    ordering: str,
    rdp_epsilon: float,
    codebook_K: int,
) -> None:
    """Retain the original contour/kinematics workflow for reproducibility."""

    print(
        f"[pipeline-v1] sketches={len(samples)} ordering={ordering} "
        f"RDP={rdp_epsilon:g} K={codebook_K}"
    )
    successful: list[tuple[np.ndarray, Path]] = []
    skipped = 0
    for image_path in tqdm(samples, desc="Legacy V1", unit="sketch"):
        try:
            strokes = vectorize_image(image_path, epsilon=rdp_epsilon, method="contour")
            ordered = order_fn(strokes)
            timed = generate_kinematics(ordered)
            if not timed:
                skipped += 1
                continue
            stroke5 = to_stroke5(timed)
            successful.append((stroke5, image_path))
            save_stroke5(stroke5, stroke5_dir / f"{image_path.stem}.npz")
        except Exception as exc:
            tqdm.write(f"[pipeline-v1] skip {image_path.name}: {exc}")
            skipped += 1

    if not successful:
        raise RuntimeError("No stroke-5 data was generated")
    arrays = [stroke5 for stroke5, _ in successful]
    drawing_points = int(sum(int((stroke5[:, 2] == 1.0).sum()) for stroke5 in arrays))
    codebook = build_codebook(arrays, K=codebook_K)
    save_codebook(
        codebook,
        sketch_token_dir,
        K=len(codebook),
        n_samples=drawing_points,
    )
    token_dir = stroke5_dir.parent / "tokens"
    for stroke5, image_path in successful:
        tokens = encode_stroke5(stroke5, codebook)
        save_token_sequence(tokens, token_dir / f"{image_path.stem}.npz")
    print(f"[pipeline-v1] accepted={len(successful)} skipped={skipped}")
