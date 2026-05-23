"""Evaluate the native in-repo Sketchformer model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()

import torch

from builders import build_loss, build_model, maybe_compile_model
from core import average_logs, load_checkpoint, move_to_device
from core.metrics import reconstruction_metrics
from dataloaders import StrokeSequenceDataModule
from metrics.sketchformer.reconstruction import (
    collect_reconstruction_examples,
    write_metrics_report,
)
from scripts.sketchformer.config import (
    compose_training_config,
    format_logs,
    limited,
    parse_batch_limit,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-batches", type=parse_batch_limit, default=1.0)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--plots-output-dir", default=None)
    parser.add_argument("--num-plots", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = compose_training_config(args.config, experiment=args.experiment)
    if args.data_root:
        config["data"]["dataset"]["root"] = args.data_root

    device = resolve_device(args.device)
    datamodule = StrokeSequenceDataModule(config["data"], project_root=PROJECT_ROOT)
    datamodule.setup("test" if args.split == "test" else "fit")
    loader = datamodule.test_dataloader() if args.split == "test" else datamodule.val_dataloader()

    raw_model = build_model(config["model"])
    if args.checkpoint:
        load_checkpoint(PROJECT_ROOT / args.checkpoint, raw_model, strict=False)
    else:
        print("[warning] evaluating randomly initialized model; pass --checkpoint for trained weights")
    model = maybe_compile_model(raw_model.to(device), config["model"])

    loss_fn = build_loss(config["optimizer"]).to(device)
    model.eval()

    logs: list[dict[str, torch.Tensor]] = []
    examples = []
    with torch.no_grad():
        for batch in limited(loader, args.limit_batches):
            batch = move_to_device(batch, device)
            output = model(batch)
            loss_output = loss_fn(output, batch)
            metric_output = reconstruction_metrics(output, batch)
            step_logs = loss_output.as_log_dict(prefix=args.split)
            step_logs.update(metric_output.as_log_dict(prefix=args.split))
            logs.append(step_logs)

            remaining_examples = args.num_plots - len(examples)
            if args.plots_output_dir and remaining_examples > 0:
                examples.extend(
                    collect_reconstruction_examples(
                        output,
                        batch,
                        max_examples=remaining_examples,
                    )
                )

    summary = average_logs(logs)
    print(format_logs(summary))

    if args.metrics_output:
        metrics_path = PROJECT_ROOT / args.metrics_output
        write_metrics_report(
            metrics_path,
            summary,
            metadata={
                "experiment": args.experiment,
                "split": args.split,
                "checkpoint": args.checkpoint,
                "data_root": config["data"]["dataset"]["root"],
                "device": str(device),
                "limit_batches": args.limit_batches,
            },
        )
        print(f"[metrics] wrote {metrics_path}")

    if args.plots_output_dir and examples:
        from metrics.sketchformer.visualisation import save_reconstruction_examples

        plot_dir = PROJECT_ROOT / args.plots_output_dir
        saved = save_reconstruction_examples(examples, plot_dir)
        print(f"[plots] wrote {len(saved)} reconstruction plots to {plot_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
