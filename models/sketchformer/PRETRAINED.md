# Pretrained Sketchformer Assets

This document describes pretrained Sketchformer assets used by the native
PyTorch fine-tuning path.

The original CVPR continuous checkpoint is a TensorFlow checkpoint family, not
a single portable weight file. A usable pretrained source includes:

- `config.json`
- `weights/checkpoint`
- one or more `weights/*.index` files
- matching `weights/*.data-*` shard files
- optional evaluation plots under `plots/`

## Why This Exists Here

Native fine-tuning needs a reliable handoff from the original TensorFlow
checkpoint layout into the in-repo PyTorch model. Before implementing the
variable-by-variable conversion table, we need a stable way to answer:

- Is the pretrained directory present?
- Which checkpoints are available?
- Is the recommended checkpoint complete?
- Which architecture settings came from the original run?
- Are previous evaluation plots available for qualitative comparison?

## Inspect The Assets

```bash
python scripts/sketchformer/inspect_pretrained.py
```

The default root is:

```text
weights/pretrained/sketch-transformer-tf2-cvpr_tform_cont
```

Print JSON instead of a text report:

```bash
python scripts/sketchformer/inspect_pretrained.py --json
```

## Conversion Boundary

This inspection utility does not convert TensorFlow weights yet. Conversion
belongs to:

```text
scripts/sketchformer/convert_checkpoint.py
models/sketchformer/checkpoint_mapping.py
```

This native model utility only validates the source assets those conversion
tools will read.
