# Training Configs

This folder contains the configuration layer for in-repo Sketchformer
fine-tuning.

The configs are intentionally split by responsibility:

- `data/` controls dataset paths, sequence length, batching, and dataloader
  behavior.
- `model/` controls the native PyTorch Sketchformer-style architecture.
- `optimizer/` controls optimizer and learning-rate scheduler settings.
- `trainer/` controls device, precision, logging, checkpointing, and runtime
  behavior.
- `experiment/` selects concrete training runs such as smoke tests and anime
  fine-tuning.

The default entrypoint config is `train.yaml`.

## Research Objective

The in-repo fine-tuning path targets more sophisticated sketches than the
short, simple QuickDraw-style drawings used by the original Sketchformer setup.
For that reason, the base data/model configs are long-sequence capable:

- the default model consumes tok-dict token sequences built from the sketch
  token dictionary;
- the model supports sequences up to `2048` tok-dict tokens;
- the anime tok-dict data config trains at `2048` by default for server-side
  runs;
- the smoke-test experiment overrides sequence length down to `256`;
- attention is configured for PyTorch scaled dot-product attention with Flash
  Attention preferred when the hardware supports it;
- the anime tok-dict data config does not build full SDPA padding masks by default, so
  long 2048-token runs can stay on the Flash/memory-efficient attention path;
- sparse attention is represented in config but disabled until the dense
  Flash/SDPA baseline is correct.

`configs/train.yaml` defaults to `anime_tok_dict` plus
`sketchformer_tok_dict`. Continuous stroke3 configs remain in this folder for
legacy compatibility, but they are not the native fine-tuning objective.

The default trainer config uses CUDA `16-mixed` precision and TF32-friendly
runtime settings for an RTX 3090-class server. CPU development should use the
`smoke_test` experiment or explicit CLI overrides.
