"""Train the native in-repo Sketchformer model."""

from __future__ import annotations

import argparse
import math
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

from builders import build_loss, build_model_from_config, build_optimizer, build_scheduler
from builders.config_utils import get_nested
from core import CheckpointCallback, average_logs, move_to_device, set_seed
from core.metrics import reconstruction_metrics
from dataloaders import StrokeSequenceDataModule
from scripts.sketchformer_config import (
    batch_limit,
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
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=parse_batch_limit, default=None)
    parser.add_argument("--limit-val-batches", type=parse_batch_limit, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.data_root:
        config["data"]["dataset"]["root"] = args.data_root
    if args.output_dir:
        config.setdefault("experiment", {}).setdefault("run", {})["output_dir"] = args.output_dir
    if args.resume:
        config.setdefault("experiment", {}).setdefault("run", {})["resume_from_checkpoint"] = args.resume
    if args.max_epochs is not None:
        config["trainer"]["training"]["max_epochs"] = args.max_epochs
    if args.limit_train_batches is not None:
        config["trainer"]["training"]["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        config["trainer"]["training"]["limit_val_batches"] = args.limit_val_batches


def _checkpoint_dir(config: dict[str, Any]) -> Path:
    output_dir = get_nested(config, "experiment.run.output_dir")
    if output_dir:
        return PROJECT_ROOT / output_dir

    output_root = get_nested(config, "paths.output_root", "weights/finetuned")
    run_name = get_nested(config, "experiment.name", "sketchformer-run")
    return PROJECT_ROOT / output_root / run_name


def _total_optimizer_steps(config: dict[str, Any], train_loader: Any) -> int:
    max_epochs = int(get_nested(config, "trainer.training.max_epochs", 1))
    limit_train = get_nested(config, "trainer.training.limit_train_batches", 1.0)
    accumulate = int(get_nested(config, "trainer.training.accumulate_grad_batches", 1))
    batches = batch_limit(train_loader, limit_train)
    return max(1, math.ceil(batches / max(1, accumulate)) * max_epochs)


def _validate(model, valid_loader, loss_fn, config: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    logs = []
    limit_val = get_nested(config, "trainer.training.limit_val_batches", 1.0)
    with torch.no_grad():
        for batch in limited(valid_loader, limit_val):
            batch = move_to_device(batch, device)
            output = model(batch)
            loss_output = loss_fn(output, batch)
            metric_output = reconstruction_metrics(output, batch)
            step_logs = loss_output.as_log_dict(prefix="val")
            step_logs.update(metric_output.as_log_dict(prefix="val"))
            logs.append(step_logs)
    return average_logs(logs)


def main() -> int:
    args = parse_args()
    config = compose_training_config(args.config, experiment=args.experiment)
    _apply_cli_overrides(config, args)

    seed = int(get_nested(config, "project.seed", 42))
    deterministic = bool(get_nested(config, "trainer.runtime.deterministic", False))
    set_seed(seed, deterministic=deterministic)

    device = resolve_device(args.device)
    checkpoint_dir = _checkpoint_dir(config)

    if args.dry_run:
        print(f"experiment={get_nested(config, 'experiment.name')}")
        print(f"data_root={get_nested(config, 'data.dataset.root')}")
        print(f"model={get_nested(config, 'model.name')}")
        print(f"device={device}")
        print(f"checkpoint_dir={checkpoint_dir}")
        print(f"max_epochs={get_nested(config, 'trainer.training.max_epochs')}")
        return 0

    datamodule = StrokeSequenceDataModule(config["data"], project_root=PROJECT_ROOT, seed=seed)
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.val_dataloader()

    model = build_model_from_config(config["model"]).to(device)
    loss_fn = build_loss(config["optimizer"]).to(device)
    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_scheduler(
        optimizer,
        config["optimizer"],
        total_steps=_total_optimizer_steps(config, train_loader),
    )

    resume_path = get_nested(config, "experiment.run.resume_from_checkpoint")
    if resume_path:
        from core import load_checkpoint

        load_checkpoint(PROJECT_ROOT / resume_path, model, optimizer=optimizer, scheduler=scheduler, strict=False)

    checkpoint_callback = CheckpointCallback(
        checkpoint_dir,
        monitor=str(get_nested(config, "trainer.checkpointing.monitor", "val/reconstruction_loss")),
        mode=str(get_nested(config, "trainer.checkpointing.mode", "min")),
        save_last=bool(get_nested(config, "trainer.checkpointing.save_last", True)),
    )

    max_epochs = int(get_nested(config, "trainer.training.max_epochs", 1))
    log_every = int(get_nested(config, "trainer.training.log_every_n_steps", 10))
    accumulate = max(1, int(get_nested(config, "trainer.training.accumulate_grad_batches", 1)))
    grad_clip = get_nested(config, "optimizer.gradient.clip_norm", None)
    limit_train = get_nested(config, "trainer.training.limit_train_batches", 1.0)
    global_step = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logs = []
        num_batches = batch_limit(train_loader, limit_train)

        for batch_index, batch in enumerate(limited(train_loader, limit_train)):
            batch = move_to_device(batch, device)
            output = model(batch)
            loss_output = loss_fn(output, batch)
            (loss_output.total / accumulate).backward()
            train_logs.append(loss_output.as_log_dict(prefix="train"))

            should_step = (batch_index + 1) % accumulate == 0 or (batch_index + 1) == num_batches
            if should_step:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
                if scheduler.scheduler is not None:
                    scheduler.scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if (batch_index + 1) % log_every == 0:
                print(f"epoch={epoch + 1} batch={batch_index + 1} {format_logs(train_logs[-1])}")

        train_epoch_logs = average_logs(train_logs)
        val_logs = _validate(model, valid_loader, loss_fn, config, device)
        checkpoint_callback.on_validation_end(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            step=global_step,
            metrics={key: float(value.detach().cpu()) for key, value in val_logs.items()},
        )

        print(f"epoch={epoch + 1} train {format_logs(train_epoch_logs)}")
        print(f"epoch={epoch + 1} valid {format_logs(val_logs)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
