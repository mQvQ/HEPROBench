# HEPROBench

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.1-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

HEPROBench provides a standardized interface for evaluating H&E-to-multiplex-
protein prediction methods. This review package includes a compact runnable
demo with method adapters, HDF5 submission writing, validation, and evaluation.

Benchmark website: https://heprobench.pages.dev/

The included data and tiny checkpoints are synthetic and are only used to run
the demo workflow. Training recipes, inference utilities, evaluation scripts, and complete pretrained
weights will be updated after publication.

## Installation

Conda:

```bash
conda env create -f environment.yml
conda activate heprobench-review
```

Pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run The Demo

From this directory:

```bash
bash run_demo.sh
```

The script runs the complete train/infer/evaluation workflow:

```bash
python scripts/prepare_demo_data.py
python -m heprobench validate-data --config configs/demo.yaml

python -m heprobench train \
  --config configs/demo.yaml \
  --method configs/methods/miphei_vit.yaml \
  --output-checkpoint outputs/miphei_vit_demo_trained.pt \
  --epochs 1

python -m heprobench infer \
  --config configs/demo.yaml \
  --method configs/methods/miphei_vit.yaml \
  --output outputs/miphei_vit

python -m heprobench validate-submission \
  --config configs/demo.yaml \
  --pred-dir outputs/miphei_vit

python -m heprobench evaluate \
  --config configs/demo.yaml \
  --pred-dir outputs/miphei_vit
```

The script repeats inference and evaluation for all bundled demo methods.

## Included Methods

The review package includes runnable tiny adapters for:

- `miphei_vit`: MIPHEI-ViT, corresponding to the H-Optimus-0 + ViTMatte-style architecture.
- `dpt_fm`: DPT decoder with a configurable pathology foundation model encoder.
- `cut`: CUT adapter.
- `hex`: HEX adapter.
- `gigatime`: GigaTIME adapter.
- `pytorch_cyclegan_and_pix2pix`: pix2pix/CycleGAN adapter.
- `rosie`: ROSIE adapter.

## Supported Pathology Foundation Models

Pathology foundation models are configured under the `dpt_fm` method family via
the `encoder_name` field:

```yaml
method:
  name: dpt_fm
  encoder_name: hoptimus0
```

The supported encoder names in this review package are:

```text
hoptimus0, h0-mini, ctranspath, conch, conchv1_5, uni, univ2,
gpfm, phikonv2, pathgen, chief, keep, virchow2, omiclip,
provgigapath
```

The registry is stored in `methods/pathology-foundation-models/registry.json`.
For this compact review demo, these names map to tiny runnable stand-ins rather
than full external foundation model checkpoints.

## Outputs

Each method writes one HDF5 file per slide:

```text
outputs/<method>/
  slide_001.h5
  slide_002.h5
  metrics.csv
  summary.json
```

Each `.h5` file follows this structured schema:

```text
root attrs:
  schema_version = "heprobench_h5_v1"
  method_name
  method_display_name
  encoder_name
  split
  patch_size

/data:
  pred          uint8  [N, H, W, C]      predicted multiplex channels
  rows          int32  [N]               tile row indices
  cols          int32  [N]               tile column indices
  grid_index    int64  [n_rows, n_cols]  maps slide grid coordinates to patch indices
  channel_names string [C]               marker names, e.g. DAPI/CD3/CD20/PanCK
  image_paths   string [N]               source H&E patch paths from metadata.csv
  target_paths  string [N]               target patch paths when ground truth is available
  has_gt        bool   [N]               whether each patch has ground truth
```

`metrics.csv` stores per-slide, per-tile, per-channel MAE, MSE, Pearson, PSNR,
and SSIM. `summary.json` stores the mean metrics over the demo set. See
`docs/submission_format.md` for the same schema in a shorter reference form.
