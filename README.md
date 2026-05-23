# Text-to-Sketch / Anime Sketchformer

This repository prepares anime portrait sketches as vector stroke sequences and
experiments with scaling the 2020 Sketchformer idea to more complex anime-style
line art.

The project currently has two parallel tracks:

1. **Original Sketchformer integration**: prepare anime stroke data and launch a
   local checkout of the original TensorFlow Sketchformer codebase for
   checkpoint-compatible experiments.
2. **Native PyTorch rebuild**: a clean in-repo Sketchformer-style model,
   dataloader, losses, checkpointing, and training loop designed for modern
   CUDA training on a single RTX 3090-class GPU.

The local development machine can be CPU-only. Full native training is expected
to run on the GPU server.

## Current State

| Area | Status |
|---|---|
| Anime image preprocessing | Implemented. Downloads portraits, extracts line art, filters sketches, vectorizes contours, orders strokes, and writes stroke-5 files. |
| Sketchformer-ready data | Implemented. Converts stroke-5 sketches into chunked stroke3 `.npz` files with train/valid/test splits. |
| Original Sketchformer handoff | Implemented as a Docker command launcher. The bundled image is CPU-oriented; GPU use requires a custom compatible image. |
| Native PyTorch Sketchformer | Implemented for continuous stroke3 reconstruction with SDPA attention, gradient checkpointing, length-bucketed loading, Gaussian-mixture reconstruction, pen-state loss, evaluation, and export. |
| Native fine-tuning from converted TF weights | Supported only after a converted PyTorch or Safetensors checkpoint exists. The TensorFlow-to-PyTorch mapping is scaffolded but not implemented yet. |
| Text prompt conditioning | Not implemented in the current codebase. The present focus is anime sketch sequence modeling. |

## Pipeline Overview

```text
Danbooru2019 portraits
  -> anime line-art sketches
  -> filtered sketch images
  -> vector contours
  -> ordered drawing paths
  -> stroke-5 arrays
  -> Sketchformer-style stroke3 chunks
  -> original Sketchformer or native PyTorch training
```

## Repository Layout

```text
.
├── configs/
│   ├── data/                    Dataset paths, stroke3 format, batching
│   ├── experiment/              Smoke test and anime fine-tuning presets
│   ├── model/                   Native Sketchformer architecture config
│   ├── optimizer/               Optimizer, scheduler, and loss weights
│   └── trainer/                 Precision, checkpointing, runtime settings
│
├── prep_data/                   Download, extraction, filtering, stroke3 prep
├── pipeline/                    Vectorization, ordering, timing, stroke-5 export
├── metrics/                     Preprocessing and reconstruction evaluation
├── utils/                       Shared IO, paths, and tokenization helpers
│
├── models/sketchformer/         Native PyTorch Sketchformer-style model
├── dataloaders/                 Stroke3 dataset, masks, collation, loaders
├── core/                        Losses, metrics, checkpointing, train helpers
├── builders/                    Model, optimizer, scheduler, loss factories
├── scripts/sketchformer/        Native train, evaluate, export, inspect CLIs
│
├── integrations/
│   └── original_sketchformer/   Launcher and Docker files for legacy TF code
├── scripts/integrations/        CLI wrappers for integration workflows
│
├── data/                        Local generated data, git-ignored
├── weights/                     Local pretrained and fine-tuned weights
├── dependencies/                Optional local third-party checkouts
├── sketchformer/                Optional original Sketchformer checkout
└── tests/                       Unit and smoke tests
```

| Area | Main Paths | Purpose |
|---|---|---|
| Data preparation | `prep_data/`, `pipeline/`, `scripts/prepare_data/` | Build clean anime sketch data from images and export stroke-5/stroke3 files. |
| Native training | `models/sketchformer/`, `dataloaders/`, `core/`, `builders/`, `scripts/sketchformer/` | Rebuilt PyTorch Sketchformer-style training path for long anime stroke sequences. |
| Configuration | `configs/` | Compose reusable data, model, optimizer, trainer, and experiment settings. |
| Legacy integration | `integrations/original_sketchformer/`, `scripts/integrations/`, `sketchformer/` | Run the original TensorFlow Sketchformer checkout for compatibility experiments. |
| Outputs | `data/`, `weights/`, `logs/`, `runs/` | Local datasets, checkpoints, logs, and training artifacts. These are not meant for source control. |

## Environment

Use Python 3.10 or 3.11 for the training environment. The repo may be inspected
on CPU, but native training needs PyTorch with CUDA on the server.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For the RTX 3090 server, install the PyTorch CUDA wheel that matches the server
driver before installing the remaining requirements. Example shape:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

The data downloader also requires the system `rsync` binary.

```bash
sudo apt install rsync
```

## Data Preparation

Download portraits:

```bash
tts-download-data --num-images 5000
```

Extract line-art sketches with the default ControlNet anime line-art detector:

```bash
tts-extract-sketches --extractor lineart-anime --max-images 5000
```

Optional Anime2Sketch extraction is supported through a separate local checkout:

```bash
tts-extract-sketches \
  --extractor anime2sketch \
  --anime2sketch-dir dependencies/Anime2Sketch \
  --anime2sketch-python dependencies/Anime2Sketch/.venv/bin/python \
  --anime2sketch-model improved \
  --anime2sketch-gpu-ids "" \
  --max-images 5000
```

Filter noisy sketches:

```bash
tts-filter-sketches --max-points 10000
```

Vectorize, order, time, and export stroke-5 sketches:

```bash
tts-run-pipeline
```

Convert stroke-5 files into Sketchformer-style stroke3 chunks:

```bash
tts-prepare-sketchformer \
  --source-dir data/processed/stroke5 \
  --target-dir data/processed/sketchformer-ready-data/stroke3 \
  --n-chunks 10
```

Expected stroke3 output:

```text
data/processed/sketchformer-ready-data/stroke3/
├── train_000.npz
├── train_001.npz
├── ...
├── valid.npz
├── test.npz
└── meta.npz
```

## Native PyTorch Training

The native path is the preferred direction for RTX 3090 training. It uses:

- stroke3 variable-length batches with SDPA-compatible masks
- length-bucketed sampling to reduce padding
- gradient checkpointing for long sequences
- CUDA mixed precision with `16-mixed` by default
- TF32 matmul enabled by default on CUDA
- full SDPA padding-mask construction disabled by default for 2048-token anime
  runs, which helps PyTorch stay on Flash or memory-efficient attention kernels
- Gaussian-mixture xy reconstruction plus pen-state cross entropy

CPU dry run:

```bash
tts-train-sketchformer --experiment smoke_test --dry-run
```

RTX 3090 training:

```bash
tts-train-sketchformer \
  --experiment anime_continuous_finetune \
  --device cuda \
  --precision 16-mixed
```

Resume a native checkpoint:

```bash
tts-train-sketchformer \
  --experiment anime_continuous_finetune \
  --device cuda \
  --resume weights/finetuned/sketchformer-continuous-anime/last.pt
```

Load a converted native checkpoint when available:

```bash
tts-train-sketchformer \
  --experiment anime_continuous_finetune \
  --device cuda \
  --pretrained weights/pretrained/sketchformer_continuous.safetensors
```

Evaluate:

```bash
tts-evaluate-sketchformer \
  --checkpoint weights/finetuned/sketchformer-continuous-anime/best.pt \
  --split valid \
  --device cuda \
  --metrics-output data/processed/evaluations/native_valid_metrics.json \
  --plots-output-dir data/processed/evaluations/native_reconstructions
```

Export weights:

```bash
tts-export-sketchformer \
  --checkpoint weights/finetuned/sketchformer-continuous-anime/best.pt \
  --output weights/finetuned/sketchformer-continuous-anime/model.safetensors
```

## Original Sketchformer Integration

The legacy path is useful for validating data compatibility against the 2020
codebase and its TensorFlow checkpoints.

Build the bundled CPU image:

```bash
tts-sketchformer-codebase-finetune --sudo build-image
```

Build the RTX 3090-oriented GPU image:

```bash
tts-sketchformer-codebase-finetune --sudo build-gpu-image
```

Prepare legacy-compatible data:

```bash
tts-sketchformer-codebase-finetune --sudo prepare-data \
  --source-dir data/processed/stroke5 \
  --target-dir data/processed/sketchformer-ready-data/stroke3 \
  --n-chunks 10 \
  --n-classes 345
```

Fine-tune with the original checkout:

```bash
tts-sketchformer-codebase-finetune --sudo finetune-continuous \
  --dataset data/processed/sketchformer-ready-data/stroke3 \
  --output-dir weights/finetuned \
  --run-id anime-continuous-finetune \
  --resume weights/pretrained/sketch-transformer-tf2-cvpr_tform_cont/weights/ckpt-12
```

For GPU experiments, provide a custom image compatible with the server CUDA
stack, or use the bundled GPU image, and expose Docker GPUs explicitly:

```bash
tts-sketchformer-codebase-finetune \
  --sudo \
  --image sketchformer-tf2-gpu \
  --gpus all \
  --dry-run \
  finetune-continuous
```

The released continuous TensorFlow checkpoint is sequence-length dependent and
was trained with `max_seq_len=200`. Keep that value when resuming the original
checkpoint. Larger values in the legacy path are for from-scratch compatibility
experiments and will use the old TensorFlow attention implementation, not the
optimized native PyTorch path.

## Configuration

Important configs:

| File | Purpose |
|---|---|
| `configs/train.yaml` | Root composed training config. |
| `configs/model/sketchformer_continuous.yaml` | Native model architecture and reconstruction head. |
| `configs/data/anime_stroke3.yaml` | Stroke3 dataset, sequence length, batching, and augmentation. |
| `configs/trainer/single_gpu.yaml` | Single-GPU runtime, precision, checkpointing, and logging settings. |
| `configs/experiment/smoke_test.yaml` | Tiny CPU-friendly dry-run/smoke settings. |
| `configs/experiment/anime_continuous_finetune.yaml` | RTX 3090-oriented native training experiment. |

The anime experiment currently trains or resumes the native PyTorch model. To
use original TensorFlow Sketchformer weights in the native path, first produce a
converted PyTorch/Safetensors checkpoint; the conversion CLI is currently a safe
inspection scaffold.

## Formats

Stroke-5:

```text
[dx, dy, p1, p2, p3]
```

Stroke3:

```text
[dx, dy, pen_state]
```

The native model consumes continuous stroke3 data. Pen states are expected to be
integer-like values in `[0, 2]`.

## Verification

Run the tests with either command:

```bash
python -m unittest discover -s tests -v
pytest -q
```

On a CPU-only development machine, use `--dry-run` and the smoke experiment to
validate config composition without launching full training.

## Known Gaps

- TensorFlow-to-PyTorch Sketchformer checkpoint conversion still needs the
  explicit variable mapping table.
- The native model is reconstruction-first; text prompt conditioning is not
  wired into the architecture yet.
- The legacy Docker image is CPU-oriented and intentionally conservative.
