# Native train-to-evaluation demo

The native demo matrix verifies that every registered method can use the same
external workflow:

```text
JSON config
  -> unified train
  -> method checkpoint
  -> unified valid/test inference
  -> HDF5 contract validation
  -> image, slide-macro, and cell evaluation
```

These are software smoke runs on the bundled synthetic data. Their metric
values are not scientific benchmark results.

## Environment

Create the documented environment and install the upstream MUSK package used
internally by HEX:

```bash
conda env create -f environment.yml
conda activate heprobench-review
pip install "git+https://github.com/lilab-stanford/MUSK.git"
```

MUSK is not a standalone HEPROBench method or a DPT-FM encoder. HEX imports it
as part of the original HEX architecture. The HEX demo disables pretrained
MUSK weight download but still exercises the architecture and training logic.

## Run all methods

Choose an available GPU explicitly when running the full matrix:

```bash
bash run_all_method_demos.sh --device cuda:0
```

The command sequentially runs the following JSONs:

| Method | Demo JSON | Prediction directory name |
| --- | --- | --- |
| ROSIE | `configs/demo/rosie.json` | `ROSIE` |
| CUT | `configs/demo/cut.json` | `CUT` |
| Pix2Pix | `configs/demo/pix2pix.json` | `pix2pix` |
| CycleGAN | `configs/demo/cyclegan.json` | `cyclegan` |
| HistoPlexer | `configs/demo/histoplexer.json` | `HistoPlexer` |
| GigaTIME original | `configs/demo/gigatime_original.json` | `GigaTIME-Original` |
| GigaTIME regression | `configs/demo/gigatime_reg.json` | `GigaTIME-Reg` |
| MIPHEI-ViT | `configs/demo/miphei_vit.json` | `MIPHEI-ViT` |
| DPT-FM H0-mini | `configs/demo/dpt_fm_h0-mini.json` | `DPT-h0-mini` |
| HEX | `configs/demo/hex.json` | `HEX` |

The script records every attempted command and exits non-zero if any method or
artifact contract fails. The machine-readable report is written to:

```text
outputs/demo/verification_summary.json
```

By default LPIPS/DISTS are skipped so the matrix tests the training and data
contracts quickly while retaining RMSE, PSNR, SSIM, diagnostic image metrics,
slide aggregation, cell PCC, and XGBoost classification. Enable optional paths
with:

```bash
bash run_all_method_demos.sh --device cuda:0 --perceptual
bash run_all_method_demos.sh --device cuda:0 --profile
```

`--profile` runs each native efficiency profiler and attaches its report to the
corresponding evaluation summary. It is hardware-dependent and substantially
slower than the default smoke matrix.

One method can be selected while developing or reviewing an implementation:

```bash
bash run_all_method_demos.sh --device cuda:0 --method rosie
bash run_all_method_demos.sh --device cuda:0 --method hex
```

Existing checkpoints or predictions can be audited without retraining:

```bash
bash run_all_method_demos.sh \
  --device cuda:0 \
  --skip-train \
  --skip-infer
```

## Equivalent individual commands

ROSIE illustrates the exact sequence used for every method:

```bash
python -m heprobench train \
  --config configs/demo/rosie.json \
  --device cuda:0

python -m heprobench infer \
  --config configs/demo/rosie.json \
  --split valid \
  --device cuda:0

python -m heprobench infer \
  --config configs/demo/rosie.json \
  --split test \
  --device cuda:0

python -m heprobench validate-submission \
  --config configs/demo/rosie.json \
  --pred-dir outputs/demo/rosie/predictions/ROSIE/valid \
  --split valid

python -m heprobench validate-submission \
  --config configs/demo/rosie.json \
  --pred-dir outputs/demo/rosie/predictions/ROSIE/test \
  --split test

python -m heprobench evaluate \
  --config configs/demo/rosie.json \
  --pred-dir outputs/demo/rosie/predictions/ROSIE \
  --no-perceptual
```

The parent prediction directory is passed to `evaluate` because cell
classification fits on `valid/` cell features and scores on `test/` cell
features.

## Automated regression test

The native E2E test is opt-in because it needs the full dependency stack, a GPU
for HEX, and several gigabytes of generated model artifacts under `outputs/`:

```bash
HEPROBENCH_RUN_E2E=1 \
HEPROBENCH_E2E_DEVICE=cuda:0 \
pytest -q tests/test_end_to_end_demos.py
```

Normal unit tests remain fast and do not start native model training.
