<div align="center">

# HEPROBench

**A standardized benchmark interface for H&E-to-multiplex protein prediction**

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.1-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

[Benchmark website](https://heprobench.pages.dev/) · [Quick start](#quick-start) · [Methods](#included-methods) · [Output format](#outputs)

</div>

---

HEPROBench provides a unified interface for evaluating H&E-to-multiplex protein
prediction methods. This repository contains a compact, runnable package with
method adapters, HDF5 submission writing, validation, and evaluation.

> [!NOTE]
> The included data and tiny checkpoints are synthetic and exist only for the
> runnable workflow. Training recipes, inference utilities, evaluation scripts,
> and complete pretrained weights will be updated after publication.

## At a Glance

| Component | What it provides |
| --- | --- |
| **Unified interface** | A consistent configuration and command-line workflow across methods. |
| **Runnable examples** | Synthetic H&E images, multiplex targets, and small checkpoints for local testing. |
| **Submission validation** | Structured HDF5 output validation before evaluation. |
| **Metrics** | Per-slide, per-tile, and per-channel MAE, MSE, Pearson, PSNR, and SSIM. |
| **Cell-level metrics** *(planned)* | Cell-aware quantitative evaluation, including per-cell marker intensity agreement and cell-level matching metrics. |

## Quick Start

### 1. Create an environment

<details open>
<summary><strong>Conda</strong></summary>

```bash
conda env create -f environment.yml
conda activate heprobench-review
```

</details>

<details>
<summary><strong>Pip</strong></summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

</details>

### 2. Run the workflow

```bash
bash run_demo.sh
```

The script prepares the data, validates it, trains, runs inference, validates
the submission, and evaluates every bundled example method.

<details>
<summary>Show the individual commands</summary>

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

</details>

## Project Layout

```text
HEPROBench/
├── configs/       # Demo and method configuration files
├── demo_data/     # Synthetic H&E patches and multiplex targets
├── docs/          # Data, method, and submission references
├── heprobench/    # CLI, training, inference, validation, and metrics
├── methods/       # Runnable method adapters and model registry
├── checkpoints/   # Tiny checkpoints used by the examples
└── outputs/       # Generated predictions and evaluation results
```

## Included Methods

| Method | Description |
| --- | --- |
| `miphei_vit` | MIPHEI-ViT with an H-Optimus-0 + ViTMatte-style architecture. |
| `dpt_fm` | DPT decoder with a configurable pathology foundation-model encoder. |
| `cut` | CUT adapter. |
| `hex` | HEX adapter. |
| `gigatime` | GigaTIME adapter. |
| `pytorch_cyclegan_and_pix2pix` | pix2pix/CycleGAN adapter. |
| `rosie` | ROSIE adapter. |

## Supported Pathology Foundation Models

Pathology foundation models are configured through the `dpt_fm` method family
using the `encoder_name` field:

```yaml
method:
  name: dpt_fm
  encoder_name: hoptimus0
```

<details>
<summary>Show supported encoder names</summary>

```text
hoptimus0, h0-mini, ctranspath, conch, conchv1_5, uni, univ2,
gpfm, phikonv2, pathgen, chief, keep, virchow2, omiclip,
provgigapath
```

</details>

The registry is stored in
[`methods/pathology-foundation-models/registry.json`](methods/pathology-foundation-models/registry.json).
For this compact package, these names map to tiny runnable stand-ins rather
than full external foundation-model checkpoints.

## Outputs

Each method produces one HDF5 file per slide, together with aggregate metrics:

```text
outputs/<method>/
├── slide_001.h5
├── slide_002.h5
├── metrics.csv
└── summary.json
```

Each `.h5` file contains predictions, tile coordinates, channel names, source
paths, target paths, and ground-truth availability. Its top-level metadata
records the schema version, method, encoder, split, and patch size.

```text
/data:
  pred          uint8  [N, H, W, C]      predicted multiplex channels
  rows          int32  [N]               tile row indices
  cols          int32  [N]               tile column indices
  grid_index    int64  [n_rows, n_cols]  maps grid coordinates to patch indices
  channel_names string [C]               marker names, e.g. DAPI/CD3/CD20/PanCK
  image_paths   string [N]               source H&E patch paths from metadata.csv
  target_paths  string [N]               target patch paths when ground truth is available
  has_gt        bool   [N]               whether each patch has ground truth
```

`metrics.csv` stores per-slide, per-tile, and per-channel MAE, MSE, Pearson,
PSNR, and SSIM. `summary.json` stores their mean values over the demo set.

## Documentation

- [Data format](docs/data_format.md)
- [Method notes](docs/methods.md)
- [Submission format](docs/submission_format.md)
- [Review notes](docs/review_notes.md)
