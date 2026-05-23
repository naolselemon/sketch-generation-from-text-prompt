# Sketchformer Codebase Fine-Tuning Integration

This folder contains the adapter for fine-tuning with the original
Sketchformer codebase.

Text-to-Sketch remains responsible for generating and preparing stroke data.
After the handoff point, this integration launches the local git-ignored
`sketchformer/` checkout for training, checkpoint loading, loss computation,
and evaluation.

The adapter does not reimplement Sketchformer's model, dataloader, masks,
losses, or checkpoint structure.

## Contents

| Path | Purpose |
|---|---|
| `launcher.py` | Builds Docker commands for original Sketchformer training and evaluation. |
| `docker/` | CPU TensorFlow 2.1 image used to run the legacy code. |

## Common Commands

Build the image:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo build-image
```

Prepare legacy-compatible stroke3 chunks:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo prepare-data \
  --source-dir data/processed/stroke5 \
  --target-dir data/processed/sketchformer-ready-data/stroke3 \
  --n-chunks 10 \
  --n-classes 345
```

Evaluate pretrained reconstruction:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo evaluate-reconstruction
```

Fine-tune:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo finetune-continuous
```
