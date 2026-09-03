# Evaluation

The bundled demo contains every ground-truth component required to exercise the
same evaluation interfaces used for benchmark submissions. It is a deterministic
software smoke test, not a reproduction of the paper's numerical results.

## Run the complete evaluation smoke test

Install `requirements-full.txt`, then run:

```bash
bash run_evaluation_demo.sh
```

The default uses `configs/demo/pix2pix.json` and CPU profiling. A different
method JSON and device can be supplied without changing the workflow:

```bash
bash run_evaluation_demo.sh configs/demo/rosie.json cuda:0
```

The script creates deterministic valid/test HDF5 predictions, profiles the
native architecture, and invokes the unified evaluator. This isolates and
verifies the evaluation stack without requiring a training run or checkpoint.
To evaluate predictions produced by a method, point the same `evaluate` command
at a directory containing `valid/` and `test/` subdirectories.

The equivalent commands are:

```bash
python scripts/prepare_demo_submission.py \
  --config configs/demo/pix2pix.json \
  --output outputs/full_evaluation_demo/submission

python -m heprobench profile \
  --config configs/demo/pix2pix.json \
  --output-dir outputs/full_evaluation_demo/profile \
  --device cpu

python -m heprobench evaluate \
  --config configs/demo/pix2pix.json \
  --pred-dir outputs/full_evaluation_demo/submission \
  --efficiency-json outputs/full_evaluation_demo/profile/efficiency.json
```

LPIPS and DISTS are enabled by the demo JSONs and require the full dependencies.
For a faster dependency-light check, pass `--no-perceptual`; all other image,
slide, and cell metrics still run.

## Image and slide metrics

The paper-compatible image metrics are RMSE, PSNR, local-window SSIM from
`skimage.metrics.structural_similarity`, LPIPS, and DISTS. MAE, MSE, and pixel
Pearson are retained as additional diagnostics. Values are first written per
tile and marker, then aggregated into per-slide/per-marker and per-slide tables.
The primary summary is a macro mean over slides, so slides with more tiles do
not dominate the result. The historical tile-weighted summary is also retained.

LPIPS and DISTS are calculated separately for each marker by repeating the
single marker as three channels and scaling it to `[-1, 1]`. Formal runs use
deterministic slide-balanced patch sampling; the eight-patch demo evaluates all
available test patches.

## Cell-level protocol

Each mask is an integer segmentation map. `0` denotes background and positive
integers are cell IDs that are unique within a slide. The evaluator averages
predicted marker intensity over every cell, including cells represented by more
than one patch.

Two cell-level analyses are then run:

- per-slide/per-marker PCC between predicted and ground-truth cell expression;
- marker-specific XGBoost classifiers fitted on validation-cell predicted
  features and evaluated on test cells, reporting AUROC, balanced accuracy, F1,
  and confusion counts.

The demo annotations contain continuous expression and synthetic median-gated
positive labels. These labels exist solely to make both classes available in a
small deterministic smoke test; they are not biological annotations or paper
results. Formal datasets can point `evaluation.cell.annotations_csv` to their
real segmentation-derived cell table and set the full XGBoost parameters in
JSON.

## Outputs

```text
<test-prediction-dir>/
  metrics.csv
  metrics_tile_channel.csv
  metrics_slide_channel.csv
  metrics_slide.csv
  summary.json
  cell_metrics/
    cell_values_valid.csv
    cell_values_test.csv
    cell_pcc_per_slide_marker.csv
    cell_pcc_summary.json
    cell_classification_per_marker.csv
    cell_classification_per_slide_marker.csv
    cell_classification_summary.json
```

`summary.json` also embeds the computational-efficiency report supplied through
`--efficiency-json`: parameter counts, FLOPs/GFLOPs per patch, latency per
region, device, precision, batch size, patch size, and region size.
