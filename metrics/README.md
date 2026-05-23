# Metrics

This folder contains evaluation utilities for two stages of the project.

## Preprocessing Metrics

These modules evaluate the sketch extraction and vectorization pipeline:

- `compare_rdp_epsilon.py` ranks dense sketches and compares RDP simplification.
- `evaluate_encoder.py` checks sketch-token encode/decode reconstruction error.
- `evaluate_ordering.py` visualizes stroke ordering strategies.
- `visualisation.py` contains plotting helpers for original-vs-simplified sketches.

## Native Sketchformer Metrics

These modules support the in-repo Sketchformer fine-tuning path:

- `sketchformer_reconstruction.py` converts model outputs into stroke3
  predictions, collects target/prediction examples, and writes JSON metric
  reports.
- `sketchformer_visualisation.py` saves target-vs-prediction reconstruction
  plots for qualitative evaluation.

Use the native evaluation script to produce artifacts:

```bash
python scripts/sketchformer/evaluate.py \
  --experiment smoke_test \
  --checkpoint weights/finetuned/smoke_test/last.pt \
  --metrics-output weights/finetuned/smoke_test/eval_metrics.json \
  --plots-output-dir weights/finetuned/smoke_test/reconstruction_plots \
  --num-plots 8
```

During early fine-tuning, prioritize:

- reconstruction loss
- xy mean squared error
- xy L1 error
- pen-state accuracy
- qualitative target-vs-prediction plots
