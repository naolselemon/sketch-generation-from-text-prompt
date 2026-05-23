"""Prepare checkpoint conversion for native Sketchformer weights.

The TensorFlow-to-PyTorch variable mapping is intentionally left explicit. This
script provides a safe CLI scaffold and dry-run validation before the actual
mapping table is implemented.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()

from models.sketchformer.pretrained import inspect_tensorflow_checkpoint
from scripts.sketchformer.config import compose_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-format", choices=["tensorflow", "torch"], default="tensorflow")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = compose_training_config(args.config, experiment=args.experiment)
    source = PROJECT_ROOT / args.source
    output = PROJECT_ROOT / args.output
    checkpoint = inspect_tensorflow_checkpoint(source)

    print(f"model={config['model']['name']}")
    print(f"source={source}")
    print(f"output={output}")
    print(f"source_format={args.source_format}")
    if args.source_format == "tensorflow":
        print(f"tensorflow_index={checkpoint.index_file}")
        print(f"tensorflow_data_shards={len(checkpoint.data_files)}")
        print(f"tensorflow_checkpoint_complete={checkpoint.exists}")

    if args.dry_run:
        return 0

    if args.source_format == "tensorflow":
        if not checkpoint.exists:
            raise SystemExit(f"Incomplete TensorFlow checkpoint source: {checkpoint.prefix}")
        raise SystemExit(
            "TensorFlow Sketchformer checkpoint conversion needs an explicit "
            "variable mapping table. Use --dry-run for now."
        )

    raise SystemExit(
        "Torch-to-native conversion is not needed yet; use scripts/sketchformer/export.py "
        "for PyTorch checkpoints."
    )


if __name__ == "__main__":
    raise SystemExit(main())
