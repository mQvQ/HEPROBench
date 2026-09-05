# CRC-CODEX preprocessing and quality control

CRC-CODEX is the reference implementation of the HEPROBench preprocessing
contract. It is a co-stained TMA dataset with 139 paired FOVs, 70 patients, 58
retained CODEX channels, 5,072 QC-passing patches, and 377,785 cells in the
reported benchmark. One TMA core is one FOV; `tile` below always means a
non-overlapping 256x256 training/evaluation patch within that FOV.

The implementation is driven by
[`configs/preprocessing/crc_codex.json`](../configs/preprocessing/crc_codex.json).
It contains the dataset-specific channel selection, thresholds, and model
inputs that were previously implicit in local scripts. The Python
implementation is [`heprobench/preprocess.py`](../heprobench/preprocess.py).

## What is and is not included

The code, channel map, configuration, input-manifest template, and output
schemas are included. Raw H&E/CODEX images and the study split manifest are not
redistributed. The one-row CSV under `configs/preprocessing/templates/` is a
schema example with fictional paths and is not study data.

The source study is Schürch et al., Mendeley Data DOI
`10.17632/mpjzbtfgfr.1` (CC BY 4.0). The reviewer demo is distributed
separately and must include its own file manifest and SHA-256 checksums; binary
data are not committed to this repository.

For exact paper-result reproduction, the authors must additionally release the
de-identified patient-level split manifest for this public cohort, or at least
its stable IDs and checksum. If a supplied slide manifest already contains a
complete `split` column, `reuse_existing: true` preserves it. Otherwise the
pipeline generates a seeded 70/10/20 grouped split and keeps every patient in
exactly one split.

## Environment

Install the benchmark requirements and preprocessing extras in a Python 3.10
environment:

```bash
pip install -r requirements-preprocessing.txt
```

VALIS uses native WSI/JVM dependencies. DeepCell/Mesmer uses TensorFlow and may
need a separate compatible environment on some systems. No credential is
stored in the repository. If a DeepCell release requires authentication, pass
it through the provider-documented environment variable.

## Input manifest

Copy the template into the configured dataset root:

```text
data/crc_codex/
  manifests/
    slides.csv
  raw/
    he/
    codex/
```

Required columns are:

```text
slide_id,patient_id,he_path,target_path,split
```

Paths may be absolute or relative to `dataset.root_dir`. `slide_id` identifies
one TMA core/FOV. `patient_id` is the grouping unit used to prevent leakage.
`split` may be blank when a new deterministic split is desired, but the exact
study split should be supplied for paper-result verification.

Inspect the resolved plan without reading or writing slides:

```bash
python -m heprobench preprocess \
  --config configs/preprocessing/crc_codex.json \
  --dry-run
```

Paths and parameters can be changed without editing Python:

```bash
python -m heprobench preprocess \
  --config configs/preprocessing/crc_codex.json \
  --output-dir /path/to/crc_codex_preprocessed \
  --set dataset.root_dir=/path/to/crc_codex_raw \
  --dry-run
```

## Two-phase execution

Registration QC deliberately contains a manual checkpoint. First run:

```bash
python -m heprobench preprocess \
  --config configs/preprocessing/crc_codex.json \
  --stage split \
  --stage register \
  --stage registration-qc
```

VALIS treats CODEX as the fixed reference and warps H&E using rigid and
non-rigid registration. It saves the registered H&E, VALIS error table, and a
red/green nuclear overlay for each FOV. Open:

```text
outputs/preprocessing/crc_codex/registration_qc/registration_qc.csv
```

Inspect every overlay and change `decision` from `pending` to either
`accepted` or `rejected`; record the reason in `notes`. Tiling refuses to
continue while decisions are pending. The QC command preserves decisions when
overlays are regenerated.

Then run the deterministic image and cell pipeline:

```bash
python -m heprobench preprocess \
  --config configs/preprocessing/crc_codex.json \
  --stage tile \
  --stage normalize \
  --stage patch-qc \
  --stage segment \
  --stage cell-extract \
  --stage gate
```

After manual decisions exist, `--stage all` is also valid. Registration outputs
are reused when `registration.skip_existing` is true.

## Exact CRC-CODEX protocol

### Channel selection and tiling

The raw 92-channel CODEX stack is reduced to the 58-channel panel in
`crc_codex_channels.json`. The retained raw indices are recorded in JSON,
including `HOECHST2` at raw index 4 and `DRAQ5` at raw index 91. Blank/empty
channels and repeated Hoechst acquisitions are excluded. Each accepted TMA
core is tiled into complete, spatially paired 256x256 H&E/target patches.

### Protein-intensity normalization

The 99.9th percentile `q` is estimated independently for every marker using
only training patches. Sampling is deterministic (up to 10,000 regularly
spaced pixels per patch) and accumulated with t-digest. The transformation
used by the benchmark experiment code is:

```text
cast_to_input_dtype(clip(log1p(clip(x, 0, q) / q) * 255, 0, 255))
```

`divide_by_log2` is intentionally `false`, and the NumPy container dtype is
preserved because the historical script assigned per-channel `uint8` values
back into the loaded array before saving. This reproduces the retained
experiment scripts, whose theoretical maximum is `255 * log(2)`, rather than a
formula divided by `log(2)`. The manuscript must use the same equation or
explicitly label a corrected normalization as a new preprocessing version.

The fitted quantiles, estimator, fit split, and transformation are written to
`channel_stats.json`.

### Patch QC

CRC-CODEX uses `DRAQ5`, not `HOECHST2`, as the nuclear reference. The H&E
reference is the inverse red channel (`255 - H&E[..., 0]`). A patch passes only
when both strict conditions hold:

```text
std(DRAQ5) > 11
robust_nmi(inverse_red, DRAQ5) > 0.03
```

The robust NMI implementation first applies Gaussian smoothing with sigma 1.5,
quantizes both images to 16 levels, retains the union of nonzero foreground
pixels, and computes `2*MI/(H(X)+H(Y))`. It therefore differs from unqualified
whole-image NMI. Standard 64-bin NMI is also emitted as a diagnostic but is not
the filtering variable.

Every patch, including failures, is retained in `patch_qc.csv` with its two
scores and pass flags. Training/validation/test CSVs contain only passing
patches.

### Mesmer and cell expression

Mesmer is run once on each reconstructed TMA-core FOV so that cells crossing a
256x256 patch boundary retain the same integer ID. Its two inputs are:

- nuclear image: min-max-normalized `DRAQ5`;
- membrane image: min-max-normalized `Na-K-ATPase` (the configured maximum
  reduction is a no-op for this single channel).

The recorded settings are `image_mpp=0.69` and
`maxima_algorithm=peak_local_max`. Zero is background. Positive cell IDs are
unique within a FOV/`slide_name`.

Marker intensity is the mean normalized value over all pixels belonging to a
cell. Area and global centroid are recorded. A cell enters evaluation only if
its centroid lies in a QC-passing patch, assigning each cell to one patch and
preventing double counting.

### GMM gating and marker QC

A two-component one-dimensional GMM is fitted separately for each marker. The
fit uses positive values, at most 300,000 cells per marker, and seed 42. The
component with the larger mean is positive; a cell label is positive when its
posterior probability for that component exceeds 0.5.

A marker is valid for binary cell evaluation only when:

```text
component separation >= 1.5
0.01 <= cohort positivity rate <= 0.99
```

Continuous cell PCC remains available for every marker. GMM parameters and QC
decisions are exported separately, so reviewers can verify which binary
metrics were excluded.

## Outputs and hand-off to training/evaluation

```text
outputs/preprocessing/crc_codex/
  resolved_preprocessing_config.json
  preprocessing_manifest.json
  channel_names.json
  channel_stats.json
  gmm_params.json
  gmm_marker_qc.csv
  cell_annotations.csv
  manifests/
    slides_with_split.csv
    registered_slides.csv
    patches_raw.csv
    patches_normalized.csv
    patch_qc.csv
    patches_with_masks.csv
  registration_qc/
    registration_qc.csv
    overlays/
  patches/
    he/
    target_raw/
    target_norm/
    masks/
  splits/
    train.csv
    valid.csv
    test.csv
  reports/
    <stage>.json
```

Each stage report includes the resolved configuration hash and input/output
checksums. The three split CSVs follow the same columns consumed by the unified
training and evaluation CLI: `slide_name`, `row`, `col`, `split`, `image_path`,
`target_path`, and `mask_path` after segmentation.

To train a method, set its experiment JSON `data.root_dir` to this output root,
set `data.csv_path` to `splits/{split}.csv`, and set
`data.channel_names_file` to `channel_names.json`. Cell evaluation should use
`cell_annotations.csv` and restrict binary metrics to markers marked valid in
`gmm_marker_qc.csv`.

## Scope

This page is the detailed CRC-CODEX tutorial. The other eight dataset-specific
channel references, boundary panels, FOV definitions, access declarations, and
parameter-audit states are recorded in
[`preprocessing_datasets.md`](preprocessing_datasets.md). Only adapters marked
`verified` should currently be described as exact historical reproductions.
