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

HEPROBench provides a unified interface for training and evaluating
H&E-to-multiplex protein prediction methods. A single CLI dispatches JSON
experiments to the method-specific training and inference implementations;
predictions then use one HDF5 contract and one evaluation interface.

> [!NOTE]
> A deidentified real CRC-CODEX reviewer demo is published separately on
> Hugging Face. The original synthetic demo remains committed for regression
> testing. Private datasets and full-size checkpoints are not distributed.

## At a Glance

| Component | What it provides |
| --- | --- |
| **Unified interface** | A consistent configuration and command-line workflow across methods. |
| **Runnable examples** | A separately hosted real CRC-CODEX reviewer bundle plus committed synthetic regression fixtures and small checkpoints. |
| **Submission validation** | Structured HDF5 output validation before evaluation. |
| **Metrics** | Paper image metrics (RMSE, PSNR, SSIM, LPIPS, DISTS), diagnostic MAE/MSE/Pearson, slide-macro aggregation, cell PCC/classification, and computational efficiency. |
| **Preprocessing/QC** | One JSON-driven pipeline plus dataset-specific adapters for all nine datasets: grouped split, registration/manual QC, FOV-aware tiling, normalization, patch QC, Mesmer, cell extraction, and GMM gating. |
| **Clinical evaluation** | JSON-driven feature extraction, patient-level folds, AMIL/MCAT training, held-out inference, survival analysis, and classification. |

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
pip install -r requirements-full.txt
pip install -e .
```

The native pipelines use `libvips`; install the system package first when it is
not available (for example, `apt install libvips` on Debian/Ubuntu). The
lighter `requirements.txt` remains available for the original review demo.

</details>

### 2. Inspect or launch the formal pipelines

```bash
python -m heprobench list-methods
python -m heprobench list-encoders

# Resolve all paths and show the ROSIE native training command.
python -m heprobench train \
  --config configs/experiments/rosie.json \
  --dry-run

# The same CLI selects one of the 15 frozen foundation-model encoders.
python -m heprobench train \
  --config configs/foundation_models/uni.json
```

Remove `--dry-run` after replacing the example dataset/checkpoint paths. Runtime
overrides do not require editing JSON, for example `--device cuda:1`,
`--batch-size 8`, `--split valid`, or `--set train.max_steps=1000`.

See [the unified CLI reference](docs/unified_cli.md) for the JSON contract and
the exact native entrypoint used by every method.

### 3. Train the native pipelines on bundled demo data

#### Real CRC-CODEX reviewer demo (recommended)

Download the public, non-gated dataset at the audited revision and verify its
file manifest:

```bash
python scripts/download_crc_codex_reviewer_demo.py
```

Dataset: [u3011706/HEPROBench-CRC-CODEX-review-demo](https://huggingface.co/datasets/u3011706/HEPROBench-CRC-CODEX-review-demo),
revision `442b41a1c7794993508558c899361b98d52e2879`.

The 12 real, registered, QC-passing H&E/CODEX patches include Mesmer cell-ID
masks and cell annotations. Run the same ten-method matrix against them with:

```bash
bash run_all_method_demos.sh \
  --device cuda:0 \
  --config-dir configs/reviewer_demo
```

These marker-balanced patches verify software execution; they are not a
representative sample and their metrics are not paper-result reproductions.
The bundle uses anonymous FOV names, contains no clinical outcomes or patient
mappings, and has no source paths or JPEG EXIF. Its provenance, GMM gates, and
SHA-256 manifest are included with the download. See the
[CRC-CODEX preprocessing reference](docs/preprocessing_crc_codex.md) for the
source and release audit.

#### Committed synthetic regression fixture

The repository includes eight 256×256 H&E/target/mask triplets split into four
train, two validation, and two test samples, plus slide-global cell IDs and a
128-cell annotation table. Every JSON below is a one-step smoke run;
`method.name` is read from the JSON, so all methods use the same command shape.

The complete train-to-evaluation matrix is one command:

```bash
bash run_all_method_demos.sh --device cuda:0
```

It trains all ten methods sequentially, runs valid/test inference, validates
both HDF5 submissions, evaluates image/per-slide/cell metrics, and writes
`outputs/demo/verification_summary.json`. See the
[native E2E demo guide](docs/end_to_end_demo.md) for individual commands,
method selection, optional LPIPS/DISTS and efficiency profiling, and the
opt-in regression test.

```bash
python -m heprobench train --config configs/demo/rosie.json
python -m heprobench train --config configs/demo/cut.json
python -m heprobench train --config configs/demo/pix2pix.json
python -m heprobench train --config configs/demo/cyclegan.json
python -m heprobench train --config configs/demo/histoplexer.json
python -m heprobench train --config configs/demo/gigatime_original.json
python -m heprobench train --config configs/demo/gigatime_reg.json
python -m heprobench train --config configs/demo/miphei_vit.json
python -m heprobench train --config configs/demo/dpt_fm_h0-mini.json
python -m heprobench train --config configs/demo/hex.json
```

MIPHEI-ViT and DPT-FM use a randomly initialized H0-mini encoder in the demo
only, avoiding gated weight downloads while retaining their real decoder,
loss, optimizer, and training loop. Formal configurations keep the pretrained
foundation-model protocol and may require `HF_TOKEN` or a local checkpoint.

HEX retains MUSK strictly as an internal architectural dependency, not as a
standalone benchmark method. Install it once before running HEX or the complete
matrix; the HEX demo uses one GPU and skips the pretrained-weight download:

```bash
pip install "git+https://github.com/lilab-stanford/MUSK.git"
```

Use `--dry-run` on any command to inspect the fully resolved native command
without importing the method stack. Formal experiment defaults and the demo
overrides are summarized in [the method notes](docs/methods.md).

### 4. Run every evaluation path on the bundled data

```bash
bash run_evaluation_demo.sh
```

This evaluation-only smoke test generates deterministic valid/test submissions,
runs native computational profiling, and evaluates image/tile, per-slide,
cell-level PCC and XGBoost classification metrics. It does not require training
a model first. Pass another method JSON and device to profile that architecture,
for example `bash run_evaluation_demo.sh configs/demo/rosie.json cuda:0`.

See [the evaluation reference](docs/evaluation.md) for individual commands,
the exact metric protocol, and the distinction between synthetic demo labels
and scientific benchmark results.

### 5. Run the downstream clinical demo

```bash
PYTHON_BIN=python bash run_clinical_demo.sh cpu
```

This generates deidentified synthetic feature bags and runs H&E-only,
virtual-only, and fusion survival models plus fusion classification through
patient-level splitting, training, inference, and evaluation. It reports
slide/patient C-index, patient-bootstrap confidence intervals, KM/log-rank,
Cox, AUC, accuracy, and macro-F1 as applicable. No real clinical data are
included or required. See [the clinical evaluation reference](docs/clinical_evaluation.md)
for formal configurations and the method-comparison workflow.

### 6. Inspect the CRC-CODEX preprocessing reference

The complete preprocessing/QC interface is also configuration-driven:

```bash
python -m heprobench preprocess \
  --config configs/preprocessing/crc_codex.json \
  --dry-run
```

It has an intentional manual checkpoint after VALIS registration. Dataset
layout, two-phase commands, the exact legacy normalization, robust-NMI
definition, Mesmer inputs, GMM gating, and output schemas are documented in
[the CRC-CODEX preprocessing reference](docs/preprocessing_crc_codex.md). The
bundled manifest row is a format example, not real study data.

The same CLI supports the other eight datasets through configs in
[`configs/preprocessing`](configs/preprocessing). Their panel maps, access
status, FOV rules, and historical-parameter audit status are summarized in the
[nine-dataset preprocessing reference](docs/preprocessing_datasets.md). No
private cohort data or real identifiers are stored in this repository.

### 7. Run the retained lightweight demo

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
├── demo_data/     # Synthetic H&E, targets, cell-ID masks, and cell annotations
├── reviewer_data/ # Ignored local download of the real CRC-CODEX reviewer demo
├── docs/          # Data, method, and submission references
├── heprobench/    # CLI, training, inference, clinical analysis, validation, and metrics
├── methods/       # Runnable method adapters and model registry
├── checkpoints/   # Tiny checkpoints used by the examples
└── outputs/       # Generated predictions and evaluation results
```

## Included Methods

| Method | Description |
| --- | --- |
| `miphei_vit` | MIPHEI-ViT dense regression. |
| `dpt_fm` | DPT decoder with one of 15 frozen pathology encoders. |
| `rosie` | ROSIE context-aggregated regression. |
| `hex` | HEX context-aggregated regression; MUSK is only an internal HEX dependency. |
| `histoplexer` | Paired consecutive-section generation with Gaussian-pyramid and patch-wise contrastive objectives. |
| `cut` | Contrastive unpaired translation. |
| `pix2pix` | Paired conditional GAN. |
| `cyclegan` | Unpaired cycle-consistent GAN. |
| `gigatime_original` | Original multi-label binary segmentation formulation. |
| `gigatime_reg` | Benchmark regression adaptation, named separately from the original. |

## Supported Pathology Foundation Models

Pathology foundation models are configured through the `dpt_fm` method family
using `model.encoder.name`:

```json
{
  "method": {"name": "dpt_fm"},
  "model": {"encoder": {"name": "hoptimus0"}}
}
```

<details>
<summary>Show supported encoder names</summary>

```text
hoptimus0, h0-mini, ctranspath, conch, conchv1_5, uni, univ2,
gpfm, phikonv2, pathgen, chief, keep, virchow2, omiclip,
provgigapath
```

</details>

The formal registry is stored in
[`methods/pfm/specs.json`](methods/pfm/specs.json), including source, checkpoint
status, input size, normalization, and feature-layer policy. The 15 inheriting
JSON experiments are in [`configs/foundation_models`](configs/foundation_models).
MUSK is intentionally absent from this benchmark registry.

## Outputs

Each method produces one HDF5 file per slide, together with aggregate metrics:

```text
outputs/<method>/
├── valid/*.h5
└── test/
    ├── *.h5
    ├── metrics_tile_channel.csv
    ├── metrics_slide_channel.csv
    ├── metrics_slide.csv
    ├── cell_metrics/
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

The metric CSVs separate tile/marker, slide/marker, and slide-level results.
`summary.json` records both slide-macro and historical tile-weighted aggregates,
cell-level results, and an optional native efficiency report.

Downstream clinical runs use a separate output contract containing one best
checkpoint and history per fold, out-of-fold prediction CSVs, patient-level
aggregates, and survival/classification summaries. See the
[clinical evaluation reference](docs/clinical_evaluation.md).

## Documentation

- [Data format](docs/data_format.md)
- [Method notes](docs/methods.md)
- [Submission format](docs/submission_format.md)
- [Evaluation protocol](docs/evaluation.md)
- [Clinical evaluation pipeline](docs/clinical_evaluation.md)
- [Unified CLI and JSON](docs/unified_cli.md)
- [Native train-to-evaluation demo](docs/end_to_end_demo.md)
- [Review notes](docs/review_notes.md)
- [Upstream code and licensing](UPSTREAM.md)
