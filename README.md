# Text-to-Sketch

Text-to-Sketch is a preprocessing project for turning anime portrait images
into vector sketch data that can be used for Sketchformer-style experiments.

The current pipeline is data preparation only. It downloads anime portrait
images, extracts line-art sketches, filters noisy sketches, vectorizes them,
orders the strokes, exports stroke-5 arrays, builds a sketch-token codebook,
and prepares Sketchformer-style stroke3 chunks.

```text
Danbooru2019 Portraits
  -> line-art sketches
  -> filtered sketches
  -> vector strokes
  -> ordered drawing paths
  -> human-like timing / kinematics
  -> stroke-5 arrays
  -> sketch-token codebook and token sequences
  -> Sketchformer-ready stroke3 chunks
```

## Current Capabilities

- Download a user-selected number of Danbooru2019 Portraits images over `rsync`.
- Skip already-installed images and resume incomplete downloads.
- Extract binary anime line-art sketches with ControlNet's `LineartAnimeDetector`.
- Optionally extract sketches with a local Anime2Sketch checkout for comparison.
- Filter sketches by foreground point count before vectorization.
- Vectorize sketches into OpenCV contour strokes with RDP simplification.
- Order strokes with directional, greedy nearest-neighbor, or TSP-style ordering.
- Add simple Sigma-Lognormal hand-motion timing.
- Export stroke-5 `.npz` files.
- Build a K-means sketch-token codebook and token sequences.
- Convert stroke-5 arrays into Sketchformer-style stroke3 train/valid/test chunks.
- Generate basic evaluation reports and visualizations.

In-repo model training code is not implemented yet. Near-term fine-tuning is
handled through the Sketchformer codebase fine-tuning handoff to the original
`sketchformer/` checkout.

## Project Layout

The layout includes active preprocessing modules plus empty training-oriented
folders. In the near term, the external `sketchformer/` checkout is used to
validate fine-tuning feasibility with the prepared portrait-sketch data. In the
long term, folders such as `models/`, `dataloaders/`, `builders/`, `core/`,
`experiments/`, and `weights/` are reserved for building our own modernized
Sketchformer-compatible codebase inside this repo instead of depending on the
original Sketchformer repository.

```text
text-to-sketch/
├── pipeline/
│   ├── lineart.py
│   ├── vectorization.py
│   ├── ordering.py
│   ├── kinematics.py
│   ├── stroke5.py
│   ├── workflow.py
│   └── run_pipeline.py
│
├── prep_data/
│   ├── download_data.py
│   ├── extract_sketches.py
│   ├── filter_sketches.py
│   ├── prepare_sketchformer.py
│   └── sketch_token/
│       └── create_token_dict.py
│
├── metrics/
│   ├── compare_rdp_epsilon.py
│   ├── evaluate_encoder.py
│   ├── evaluate_ordering.py
│   └── visualisation.py
│
├── scripts/
│   ├── prepare_data/
│   ├── metrics/
│   └── run_pipeline.py
│
├── utils/
│   ├── io.py
│   ├── paths.py
│   └── tokenizer.py
│
├── builders/
│   ├── __init__.py
│   └── layers/
│       └── __init__.py
├── core/
│   └── __init__.py
├── dataloaders/
│   └── __init__.py
├── models/
│   └── __init__.py
├── experiments/
│   └── __init__.py
├── dependencies/
│   ├── README.md
│   └── Anime2Sketch/          # optional local checkout, git-ignored
├── integrations/
│   └── original_sketchformer/
│       ├── launcher.py
│       └── docker/
│
├── data/
├── weights/
│   ├── README.md
│   ├── pretrained/
│   └── finetuned/
├── sketchformer/
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

| Path | Purpose |
|---|---|
| `prep_data/download_data.py` | Downloads Danbooru2019 Portraits images. |
| `prep_data/extract_sketches.py` | Converts raw portrait images to binary line-art sketches. |
| `prep_data/filter_sketches.py` | Filters sketches by point count. |
| `pipeline/` | Vectorization, stroke ordering, kinematics, and stroke-5 export. |
| `prep_data/sketch_token/` | K-means sketch-token codebook generation. |
| `prep_data/prepare_sketchformer.py` | Converts stroke-5 files into Sketchformer-style stroke3 chunks. |
| `metrics/` | Evaluation and visualization scripts. |
| `utils/paths.py` | Shared default paths. |
| `builders/` | Reserved for model-building helpers, custom layers, losses, and schedulers. |
| `core/` | Reserved for future training loops, validation loops, checkpointing, and orchestration. |
| `dataloaders/` | Reserved for Sketchformer-compatible dataset loaders. |
| `models/` | Reserved for in-repo Sketchformer or Sketchformer-inspired model implementations. |
| `experiments/` | Reserved for fine-tuning configs, ablations, and experiment entry points. |
| `dependencies/` | Environment notes and integration docs for training dependencies. |
| `dependencies/Anime2Sketch/` | Optional local Anime2Sketch checkout, virtualenv, and weights. This is git-ignored. |
| `integrations/original_sketchformer/` | Adapter that launches the original Sketchformer checkout for fine-tuning and evaluation. |
| `data/` | Local generated data. This is git-ignored. |
| `weights/` | Placeholder for pretrained and fine-tuned weights. |
| `sketchformer/` | External Sketchformer checkout for near-term fine-tuning feasibility validation. |

## Requirements

Use Python 3.10 or newer.

Python dependencies are installed from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The dataset downloader also requires the system `rsync` binary. On
Debian/Ubuntu:

```bash
sudo apt install rsync
```

Install the project in editable mode to make the `tts-*` shortcuts available:

```bash
pip install -e .
```

Anime2Sketch is optional and is intentionally kept outside git. When using
`--extractor anime2sketch`, keep a local Mukosame/Anime2Sketch checkout, its
own virtualenv, and its pretrained weights under `dependencies/Anime2Sketch/`.
The main pipeline calls that checkout through `--anime2sketch-python`, so the
project environment and Anime2Sketch environment can stay separate.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Create local environment config:

```bash
cp .env.example .env
```

The default `.env.example` points the pipeline at Danbooru2019 Portraits:

```env
DOWNLOAD_TARGET_DIR=data/raw/portraits
DANBOORU2019_PORTRAITS_RSYNC_URL=rsync://176.9.41.242:873/biggan/portraits/

INPUT_DIR=data/raw/portraits
OUTPUT_DIR=data/processed/sketches
SKETCH_EXTRACTOR=lineart-anime

DETECT_RES=512
IMAGE_RES=512
MAX_IMAGES=15

# Optional Anime2Sketch extractor settings
# Anime2Sketch is an ignored local dependency, so clone/install it manually:
#   dependencies/Anime2Sketch/
#
# Expected local weight files:
#   dependencies/Anime2Sketch/weights/netG.pth
#   dependencies/Anime2Sketch/weights/improved.bin
#
# To use it, copy this file to .env and change:
#   SKETCH_EXTRACTOR=anime2sketch
ANIME2SKETCH_DIR=dependencies/Anime2Sketch
ANIME2SKETCH_PYTHON=dependencies/Anime2Sketch/.venv/bin/python

# Choices: default uses weights/netG.pth, improved uses weights/improved.bin.
ANIME2SKETCH_MODEL=improved
ANIME2SKETCH_GPU_IDS=
ANIME2SKETCH_CLAHE_CLIP=-1
```

## Data Locations

| Stage | Default Path |
|---|---|
| Raw portrait images | `data/raw/portraits/` |
| Download manifest | `data/raw/portraits/.danbooru2019-portraits-files.txt` |
| Extracted sketches | `data/processed/sketches/<extractor_name>/` |
| ControlNet sketches | `data/processed/sketches/lineart_anime/` |
| Anime2Sketch sketches | `data/processed/sketches/anime2sketch/` |
| Anime2Sketch checkout | `dependencies/Anime2Sketch/` local and git-ignored |
| Anime2Sketch weights | `dependencies/Anime2Sketch/weights/netG.pth` and `dependencies/Anime2Sketch/weights/improved.bin` |
| Filtered sketches | `data/processed/sketches_filtered/` |
| Filter report | `data/processed/sketch_point_filter_report.csv` |
| Stroke-5 arrays | `data/processed/stroke5/` |
| Sketch-token codebook | `data/processed/sketch_token/codebook.npy` |
| Token sequences | `data/processed/tokens/` |
| Sketchformer-ready chunks | `data/processed/sketchformer-ready-data/stroke3/` |
| Evaluation outputs | `data/processed/evaluations/` |

## Run The Pipeline

You can use either the script paths or the installed `tts-*` shortcuts.

### 1. Download Portrait Images

Choose how many raw images should exist locally:

```bash
python scripts/prepare_data/download_data.py --num-images 5000
```

Equivalent shortcut:

```bash
tts-download-data --num-images 5000
```

The downloader lists the remote portrait images, checks local files by path and
file size, skips complete images, and downloads only the missing remainder.

Preview without downloading:

```bash
python scripts/prepare_data/download_data.py --num-images 5000 --dry-run
```

Limit bandwidth:

```bash
python scripts/prepare_data/download_data.py --num-images 5000 --bwlimit 5m
```

Override the destination or rsync source:

```bash
python scripts/prepare_data/download_data.py \
  --num-images 5000 \
  --target-dir data/raw/portraits \
  --rsync-url rsync://176.9.41.242:873/biggan/portraits/
```

### 2. Extract Line-Art Sketches

The extraction stage supports two sketch extractors:

| Extractor | Status | Output Folder |
|---|---|---|
| `lineart-anime` | Default ControlNet `LineartAnimeDetector` baseline. | `data/processed/sketches/lineart_anime/` |
| `anime2sketch` | Optional local Anime2Sketch checkout. | `data/processed/sketches/anime2sketch/` |

The input portraits are treated as one flat image pool. `--max-images` limits
the number of new sketches created in the current run. Existing output files
are skipped before the limit is applied, so interrupted runs can be resumed
without regenerating sketches.

For example, if 100 raw portraits exist, 40 Anime2Sketch outputs already exist,
and you run `--max-images 20`, the command creates 20 more sketches from the
remaining 60 missing outputs.

#### ControlNet Baseline

ControlNet remains the default:

```bash
python scripts/prepare_data/extract_sketches.py
```

Equivalent shortcut:

```bash
tts-extract-sketches
```

By default, outputs are namespaced by extractor:

```text
data/processed/sketches/lineart_anime/
```

Run with explicit settings:

```bash
python scripts/prepare_data/extract_sketches.py \
  --input-dir data/raw/portraits \
  --output-dir data/processed/sketches \
  --extractor lineart-anime \
  --detect-resolution 512 \
  --image-resolution 512 \
  --max-images 15
```

Use `--max-images 0` to process every pending image.

#### Anime2Sketch

Anime2Sketch is an optional external extractor. It is ignored by git, so each
machine that wants to use it must prepare its own local checkout and weights.

Clone and install it into a separate virtualenv:

```bash
mkdir -p dependencies
git clone https://github.com/Mukosame/Anime2Sketch.git dependencies/Anime2Sketch
cd dependencies/Anime2Sketch
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cd ../..
```

Place the model files in the checkout:

```text
dependencies/Anime2Sketch/weights/netG.pth
dependencies/Anime2Sketch/weights/improved.bin
```

The `default` model uses `netG.pth`. The `improved` model uses `improved.bin`
and is the better starting point for darker or lower-contrast portraits.

Run Anime2Sketch through the main project environment, while pointing to the
Anime2Sketch environment:

```bash
python scripts/prepare_data/extract_sketches.py \
  --input-dir data/raw/portraits \
  --output-dir data/processed/sketches \
  --extractor anime2sketch \
  --anime2sketch-dir dependencies/Anime2Sketch \
  --anime2sketch-python dependencies/Anime2Sketch/.venv/bin/python \
  --anime2sketch-model improved \
  --anime2sketch-gpu-ids "" \
  --image-resolution 512 \
  --max-images 20
```

For the default Anime2Sketch weights, change only the model flag:

```bash
--anime2sketch-model default
```

On CPU-only machines, keep `--anime2sketch-gpu-ids ""`.

Anime2Sketch outputs are written separately:

```text
data/processed/sketches/anime2sketch/
```

To compare extractors fairly, run both extractors over the same portrait set and
inspect their separate output folders:

```bash
python scripts/prepare_data/extract_sketches.py \
  --extractor lineart-anime \
  --max-images 20

python scripts/prepare_data/extract_sketches.py \
  --extractor anime2sketch \
  --anime2sketch-dir dependencies/Anime2Sketch \
  --anime2sketch-python dependencies/Anime2Sketch/.venv/bin/python \
  --anime2sketch-model improved \
  --anime2sketch-gpu-ids "" \
  --max-images 20
```

Use `--flat-output` only when you intentionally want to write directly into
`--output-dir` without the extractor subfolder.

### 3. Filter Noisy Sketches

```bash
python scripts/prepare_data/filter_sketches_by_points.py
```

Equivalent shortcut:

```bash
tts-filter-sketches
```

Default behavior:

- Reads sketches from `data/processed/sketches/`.
- Recursively includes extractor-specific subfolders such as `lineart_anime/`
  and `anime2sketch/`.
- Keeps sketches at or below the user-selected `--max-points` threshold.
- Copies kept sketches to `data/processed/sketches_filtered/`.
- Writes `data/processed/sketch_point_filter_report.csv`.

To filter only one extractor, point `--input-dir` and `--output-dir` at that
extractor's folder:

```bash
python scripts/prepare_data/filter_sketches_by_points.py \
  --input-dir data/processed/sketches/anime2sketch \
  --output-dir data/processed/sketches_filtered/anime2sketch \
  --max-points 10000
```

Useful options:

```bash
python scripts/prepare_data/filter_sketches_by_points.py \
  --max-points 10000 \
  --count original \
  --limit 100
```

### 4. Vectorize, Order, Time, And Tokenize

```bash
python scripts/run_pipeline.py
```

Equivalent shortcut:

```bash
tts-run-pipeline
```

This interactive command samples filtered sketches, asks how many to process,
asks which ordering method to use, then writes:

- stroke-5 arrays under `data/processed/stroke5/`
- sketch-token codebook under `data/processed/sketch_token/`
- token sequences under `data/processed/tokens/`

### 5. Prepare Sketchformer-Style Stroke3 Data

```bash
python scripts/prepare_data/prepare_anime_data.py
```

Equivalent shortcut:

```bash
tts-prepare-sketchformer
```

This converts stroke-5 arrays into chunked stroke3 data:

```text
data/processed/sketchformer-ready-data/stroke3/
├── train_000.npz
├── train_001.npz
├── ...
├── valid.npz
├── test.npz
└── meta.npz
```

## Sketchformer Codebase Fine-Tuning

Near-term fine-tuning follows a strict handoff to the original Sketchformer
codebase:

```text
Text-to-Sketch pipeline
  -> cleaned sketches
  -> stroke5 files
  -> Sketchformer-compatible stroke3 chunks
  -> handoff

Original sketchformer/ checkout
  -> dataloader
  -> model architecture
  -> losses and masks
  -> pretrained checkpoint loading
  -> fine-tuning and evaluation
```

The Text-to-Sketch code does not reimplement Sketchformer's model, dataloader,
losses, masks, or checkpoint structure in this workflow. It only prepares the
data and launches the original Sketchformer code through
`integrations/original_sketchformer/` in a legacy TensorFlow 2.1 Docker
environment.

Build the CPU image:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo build-image
```

Write legacy NumPy-compatible stroke3 chunks from existing stroke5 data:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo prepare-data \
  --source-dir data/processed/stroke5 \
  --target-dir data/processed/sketchformer-ready-data/stroke3 \
  --n-chunks 10 \
  --n-classes 345
```

For a tiny smoke test, add `--min-valid-size 18` so the original reconstruction
plotter has enough validation samples for its fixed grid.

Evaluate the pretrained continuous checkpoint:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo evaluate-reconstruction \
  --dataset data/processed/sketchformer-ready-data/stroke3 \
  --output-dir weights/pretrained \
  --model-id cvpr_tform_cont \
  --resume weights/pretrained/sketch-transformer-tf2-cvpr_tform_cont/weights/ckpt-12
```

Fine-tune the continuous checkpoint:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo finetune-continuous \
  --dataset data/processed/sketchformer-ready-data/stroke3 \
  --output-dir weights/finetuned \
  --run-id anime-continuous-finetune \
  --resume weights/pretrained/sketch-transformer-tf2-cvpr_tform_cont/weights/ckpt-12
```

The default fine-tuning command keeps Sketchformer's classifier layer present
for checkpoint compatibility but sets `class_weight=0.0`, so unlabeled anime
data is optimized through reconstruction rather than dummy classification.

Use `--dry-run` before any launcher command to print the Docker command without
executing it:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo --dry-run finetune-continuous
```

Detailed notes live in
`study/fine-tune-strategy/sketchformer_codebase_finetuning.md`.

## Evaluation Commands

Compare stroke ordering strategies:

```bash
python scripts/metrics/evaluate_ordering.py --samples 20
```

Shortcut:

```bash
tts-evaluate-ordering --samples 20
```

Evaluate sketch-token encoding and decoding:

```bash
python scripts/metrics/evaluate_encoder.py
```

Shortcut:

```bash
tts-evaluate-encoder
```

Compare RDP simplification on dense sketches:

```bash
python scripts/metrics/compare_rdp_epsilon.py
```

Shortcut:

```bash
tts-compare-rdp
```

## Formats

### Stroke-5

Stroke-5 is the main vector format produced by this preprocessing pipeline.
Each `.npz` file contains a `stroke5` array with shape `(N + 1, 5)`.

```text
[dx, dy, p1, p2, p3]
```

| Column | Meaning |
|---|---|
| `dx` | X movement from the previous point. |
| `dy` | Y movement from the previous point. |
| `p1` | Pen is drawing. |
| `p2` | Pen lifts after this point. |
| `p3` | End-of-sketch marker. |

Example:

```python
from utils.io import load_stroke5

s5 = load_stroke5("data/processed/stroke5/example.npz")
```

### Stroke3

Stroke3 is the Sketchformer-style training format:

```text
[dx, dy, pen_state]
```

`prep_data/prepare_sketchformer.py` converts stroke-5 files into stroke3
train/valid/test chunks.

### Sketch Tokens

Sketch tokens are integer IDs produced by quantizing `[dx, dy]` movements with
a K-means codebook.

```python
from utils.io import load_codebook, load_stroke5
from utils.tokenizer import decode_tokens, encode_stroke5

stroke5 = load_stroke5("data/processed/stroke5/example.npz")
codebook = load_codebook("data/processed/sketch_token/codebook.npy")

tokens = encode_stroke5(stroke5, codebook)
reconstructed = decode_tokens(tokens, codebook)
```

## Command Reference

| Task | Script | Shortcut |
|---|---|---|
| Download portraits | `python scripts/prepare_data/download_data.py --num-images 5000` | `tts-download-data --num-images 5000` |
| Extract ControlNet line-art | `python scripts/prepare_data/extract_sketches.py --extractor lineart-anime --max-images 20` | `tts-extract-sketches --extractor lineart-anime --max-images 20` |
| Extract Anime2Sketch line-art | `python scripts/prepare_data/extract_sketches.py --extractor anime2sketch --anime2sketch-dir dependencies/Anime2Sketch --anime2sketch-python dependencies/Anime2Sketch/.venv/bin/python --anime2sketch-model improved --anime2sketch-gpu-ids "" --max-images 20` | `tts-extract-sketches --extractor anime2sketch --anime2sketch-dir dependencies/Anime2Sketch --anime2sketch-python dependencies/Anime2Sketch/.venv/bin/python --anime2sketch-model improved --anime2sketch-gpu-ids "" --max-images 20` |
| Filter sketches | `python scripts/prepare_data/filter_sketches_by_points.py` | `tts-filter-sketches` |
| Run main pipeline | `python scripts/run_pipeline.py` | `tts-run-pipeline` |
| Prepare stroke3 data | `python scripts/prepare_data/prepare_anime_data.py` | `tts-prepare-sketchformer` |
| Sketchformer codebase fine-tuning | `python scripts/integrations/sketchformer_codebase_finetune.py --sudo --dry-run finetune-continuous` | `tts-sketchformer-codebase-finetune --sudo --dry-run finetune-continuous` |
| Evaluate ordering | `python scripts/metrics/evaluate_ordering.py --samples 20` | `tts-evaluate-ordering --samples 20` |
| Evaluate encoder | `python scripts/metrics/evaluate_encoder.py` | `tts-evaluate-encoder` |
| Compare RDP epsilon | `python scripts/metrics/compare_rdp_epsilon.py` | `tts-compare-rdp` |
