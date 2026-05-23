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
