"""Train the native in-repo Sketchformer model."""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
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

from builders import (
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
    maybe_compile_model,
)
from builders.config_utils import get_nested
from core import CheckpointCallback, average_logs, move_to_device, set_seed
from core.metrics import reconstruction_metrics
from dataloaders import StrokeSequenceDataModule
from scripts.sketchformer.config import (
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
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=parse_batch_limit, default=None)
    parser.add_argument("--limit-val-batches", type=parse_batch_limit, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.data_root:
        config["data"]["dataset"]["root"] = args.data_root
    if args.output_dir:
        config.setdefault("experiment", {}).setdefault("run", {})[
            "output_dir"
        ] = args.output_dir
    if args.pretrained:
        pretrained = config.setdefault("experiment", {}).setdefault("pretrained", {})
        pretrained["use_converted_sketchformer_weights"] = True
        pretrained["path"] = args.pretrained
    if args.resume:
        config.setdefault("experiment", {}).setdefault("run", {})[
            "resume_from_checkpoint"
        ] = args.resume
    if args.precision:
        config.setdefault("trainer", {}).setdefault("runtime", {})["precision"] = args.precision
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


@dataclass(frozen=True)
class PrecisionRuntime:
    """Autocast and scaler settings resolved from trainer.runtime.precision."""

    requested: str
    effective: str
    autocast_dtype: torch.dtype | None
    scaler: Any


def _configure_torch_runtime(config: dict[str, Any], device: torch.device) -> None:
    """Apply device-level runtime settings that matter on RTX-class GPUs."""

    if device.type != "cuda":
        return

    benchmark = bool(get_nested(config, "trainer.runtime.benchmark", True))
    allow_tf32 = bool(get_nested(config, "trainer.runtime.allow_tf32", True))
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _resolve_precision_runtime(
    config: dict[str, Any],
    device: torch.device,
) -> PrecisionRuntime:
    requested = str(get_nested(config, "trainer.runtime.precision", "32-true")).lower()
    effective = requested
    autocast_dtype: torch.dtype | None = None

    if requested in {"32", "32-true", "fp32", "float32"}:
        effective = "32-true"
    elif requested in {"16", "16-mixed", "fp16", "float16"}:
        if device.type != "cuda":
            print(f"[precision] {requested} requested on {device.type}; using 32-true")
            effective = "32-true"
        else:
            effective = "16-mixed"
            autocast_dtype = torch.float16
    elif requested in {"bf16", "bf16-mixed", "bfloat16"}:
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            print("[precision] bf16 is not supported on this CUDA device; using 16-mixed")
            effective = "16-mixed"
            autocast_dtype = torch.float16
        elif device.type in {"cuda", "cpu"}:
            effective = "bf16-mixed"
            autocast_dtype = torch.bfloat16
        else:
            print(f"[precision] bf16 requested on {device.type}; using 32-true")
            effective = "32-true"
    else:
        raise ValueError(
            "trainer.runtime.precision must be one of 32-true, 16-mixed, or bf16-mixed"
        )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and effective == "16-mixed")
    )
    return PrecisionRuntime(
        requested=requested,
        effective=effective,
        autocast_dtype=autocast_dtype,
        scaler=scaler,
    )


def _autocast_context(device: torch.device, precision: PrecisionRuntime):
    if precision.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=precision.autocast_dtype)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load_state_dict_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path)))

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a mapping: {path}")
    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _load_pretrained_if_configured(model: torch.nn.Module, config: dict[str, Any]) -> None:
    use_pretrained = bool(
        get_nested(
            config,
            "experiment.pretrained.use_converted_sketchformer_weights",
            False,
        )
    ) or bool(get_nested(config, "model.checkpoint.load_converted_weights", False))
    if not use_pretrained:
        return

    path = get_nested(
        config,
        "experiment.pretrained.path",
        get_nested(config, "model.checkpoint.converted_weights_path"),
    )
    if not path:
        raise ValueError("Pretrained loading is enabled, but no pretrained path is configured")

    checkpoint_path = _resolve_project_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Configured pretrained weights do not exist: "
            f"{checkpoint_path}. Convert TensorFlow weights first or disable pretrained loading."
        )

    strict = bool(
        get_nested(
            config,
            "experiment.pretrained.strict",
            get_nested(config, "model.checkpoint.strict", False),
        )
    )
    incompatible = model.load_state_dict(_load_state_dict_file(checkpoint_path), strict=strict)
    print(
        "[pretrained] loaded={} missing_keys={} unexpected_keys={}".format(
            checkpoint_path,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
    )


def _validate(
    model,
    valid_loader,
    loss_fn,
    config: dict[str, Any],
    device: torch.device,
    precision: PrecisionRuntime,
) -> dict[str, torch.Tensor]:
    model.eval()
    logs = []
    limit_val = get_nested(config, "trainer.training.limit_val_batches", 1.0)
    with torch.no_grad():
        for batch in limited(valid_loader, limit_val):
            batch = move_to_device(batch, device)
            with _autocast_context(device, precision):
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
    _configure_torch_runtime(config, device)
    precision = _resolve_precision_runtime(config, device)
    checkpoint_dir = _checkpoint_dir(config)

    if args.dry_run:
        print(f"experiment={get_nested(config, 'experiment.name')}")
        print(f"data_root={get_nested(config, 'data.dataset.root')}")
        print(f"model={get_nested(config, 'model.name')}")
        print(f"device={device}")
        print(f"precision={precision.effective}")
        print(f"checkpoint_dir={checkpoint_dir}")
        print(f"max_epochs={get_nested(config, 'trainer.training.max_epochs')}")
        return 0

    datamodule = StrokeSequenceDataModule(
        config["data"],
        project_root=PROJECT_ROOT,
        seed=seed,
    )
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.val_dataloader()

    raw_model = build_model(config["model"])
    _load_pretrained_if_configured(raw_model, config)
    raw_model = raw_model.to(device)
    model = maybe_compile_model(raw_model, config["model"])
    loss_fn = build_loss(config["optimizer"]).to(device)
    optimizer = build_optimizer(raw_model, config["optimizer"])
    scheduler = build_scheduler(
        optimizer,
        config["optimizer"],
        total_steps=_total_optimizer_steps(config, train_loader),
    )

    resume_path = get_nested(config, "experiment.run.resume_from_checkpoint")
    if resume_path:
        from core import load_checkpoint

        load_checkpoint(
            PROJECT_ROOT / resume_path,
            raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            strict=False,
        )

    checkpoint_callback = CheckpointCallback(
        checkpoint_dir,
        monitor=str(
            get_nested(config, "trainer.checkpointing.monitor", "val/token_loss")
        ),
        mode=str(get_nested(config, "trainer.checkpointing.mode", "min")),
        save_last=bool(get_nested(config, "trainer.checkpointing.save_last", True)),
    )

    max_epochs = int(get_nested(config, "trainer.training.max_epochs", 1))
    log_every = int(get_nested(config, "trainer.training.log_every_n_steps", 10))
    accumulate = max(
        1,
        int(get_nested(config, "trainer.training.accumulate_grad_batches", 1)),
    )
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
            with _autocast_context(device, precision):
                output = model(batch)
                loss_output = loss_fn(output, batch)
                scaled_loss = loss_output.total / accumulate

            precision.scaler.scale(scaled_loss).backward()
            train_logs.append(loss_output.as_log_dict(prefix="train"))

            should_step = (
                (batch_index + 1) % accumulate == 0
                or (batch_index + 1) == num_batches
            )
            if should_step:
                if grad_clip is not None:
                    precision.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(grad_clip))
                precision.scaler.step(optimizer)
                precision.scaler.update()
                if scheduler.scheduler is not None:
                    scheduler.scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if (batch_index + 1) % log_every == 0:
                print(
                    f"epoch={epoch + 1} batch={batch_index + 1} "
                    f"{format_logs(train_logs[-1])}"
                )

        train_epoch_logs = average_logs(train_logs)
        val_logs = _validate(model, valid_loader, loss_fn, config, device, precision)
        checkpoint_callback.on_validation_end(
            raw_model,
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
