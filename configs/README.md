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

- the model supports sequences up to `2048` stroke3 steps;
- the data config starts at `1024` and defines a curriculum toward `2048`;
- the smoke-test experiment overrides sequence length down to `256`;
- attention is configured for PyTorch scaled dot-product attention with Flash
  Attention preferred when the hardware supports it;
- sparse attention is represented in config but disabled until the dense
  Flash/SDPA baseline is correct.

This keeps the research direction ambitious while making implementation and
debugging staged.
