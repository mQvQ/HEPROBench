# Clinical Evaluation Pipeline

The clinical pipeline turns HEPROBench virtual-staining submissions into
patch-level protein features and evaluates their downstream utility. It is a
privacy-safe refactoring of the internal benchmark survival/classification
workflow: all paths, cohort identifiers, labels, results, and checkpoints are
supplied externally through JSON.

No real clinical records are included in this repository. The runnable demo is
deterministic synthetic data intended only to verify the software path.

## Reviewer quick start

Install the full environment, then run:

```bash
PYTHON_BIN=python bash run_clinical_demo.sh cpu
```

This creates 24 deidentified synthetic patients and exercises four complete
cross-validation runs:

- H&E-only AMIL survival;
- virtual-protein-only AMIL survival;
- aligned H&E + virtual-protein MCAT survival;
- aligned H&E + virtual-protein MCAT binary classification.

Each run performs patient-level splitting, training, held-out-fold inference,
and evaluation. Outputs are written below `outputs/clinical_demo/`. They are
software checks, not scientific benchmark results.

## One CLI, seven stages

```bash
python -m heprobench clinical \
  --config configs/clinical/tcga_survival_fusion.json \
  --stage all
```

`--stage` may be repeated when only part of the workflow is needed:

| Stage | Input | Output |
| --- | --- | --- |
| `prepare-outcomes` | A local clinical table | Canonical `patient_id`, `slide_id`, outcome manifest |
| `extract-features` | Per-slide HEPROBench HDF5 predictions | Per-marker and marker-aggregated DINOv2 features |
| `align-features` | Virtual features, H&E features, patch-ID lists | Patch-aligned H&E and virtual bags |
| `make-splits` | Canonical manifest and feature folders | Patient-disjoint fold CSVs |
| `train` | Feature bags, outcomes, split CSVs | Best checkpoint and history for each fold |
| `infer` | Fold checkpoints and `train`, `val`, or `all` bags | Slide-level risks or class probabilities |
| `evaluate` | Clinical prediction CSV | Fold, patient, survival/classification summaries |

Use `--dry-run` to inspect the resolved plan without reading patient data or
writing outputs. Every executed run stores `resolved_config.clinical.json` and
its SHA-256 under `_heprobench`. All JSON values can be changed without editing
Python:

```bash
python -m heprobench clinical \
  --config configs/clinical/tcga_survival_virtual.json \
  --device cuda:1 \
  --output-dir /local/output/rosie_survival \
  --set train.epochs=100 \
  --set train.gradient_accumulation=32
```

## Applying the same evaluation to every benchmark method

Clinical evaluation is intentionally method-agnostic. First run the normal
HEPROBench inference command for ROSIE, HEX, HistoPlexer, CUT,
Pix2Pix/CycleGAN, GigaTIME Original, GigaTIME-Reg, MIPHEI-ViT, or a DPT-FM
encoder. Then point the clinical JSON to that method's HDF5 directory and give
it a method-specific feature and result directory:

```bash
python -m heprobench clinical \
  --config configs/clinical/tcga_survival_virtual.json \
  --set clinical.prediction_method=rosie \
  --set data.prediction_h5_dir=/local/predictions/rosie/test \
  --set data.manifest_csv=/local/private/clinical_survival_mapped.csv \
  --set data.split_dir=/local/private/shared_clinical_splits \
  --set features.root=/local/features/rosie \
  --output-dir /local/results/rosie_survival \
  --stage extract-features \
  --stage make-splits \
  --stage train \
  --stage infer \
  --stage evaluate
```

Use the same manifest and split directory for all methods in a comparison.
Only `prediction_h5_dir`, `features.root`, `clinical.prediction_method`, and the
run directory should change. The H&E-only baseline does not require virtual
predictions. The fusion configuration additionally requires CONCH H&E features
and `align-features` before training.

The example absolute paths in `configs/clinical/` are deliberate placeholders.
Real TCGA manifests and controlled-cohort identifiers must remain outside Git.

## Input contracts

### Canonical clinical manifest

Survival requires:

```text
patient_id,slide_id,event_time,censor
```

`censor=1` means censored; `censor=0` means the event was observed. Multiple
slides may belong to one patient, but their outcomes must agree. Classification
replaces `event_time,censor` with `label`; `clinical.class_names` defines the
stable label order.

`prepare-outcomes` maps configurable source columns into this schema. It does
not anonymize a cohort, so the resulting file is still local controlled data.
If `slide_id` is unavailable, the patient ID is used and `make-splits` can match
a unique feature directory by prefix. The formal configs use
`split.unmatched_policy="error"` so an absent or ambiguous match cannot silently
change the cohort.

### Virtual-staining HDF5

The feature extractor accepts the standard `heprobench_h5_v1` structure:

```text
/data/pred           [patch, height, width, marker]
/data/channel_names  [marker]
/data/image_paths    [patch]                 preferred patch identity
/data/rows, /data/cols                       fallback patch identity
/meta slide_name attribute                    slide identity
```

For each marker, the grayscale predicted patch is replicated to RGB, resized to
224×224, ImageNet-normalized, and embedded with DINOv2 ViT-S/14. Channel order
comes from `channel_names`. The formal feature dimension is 384 and the pinned
checkpoint SHA-256 is
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
The first run uses `torch.hub` and therefore needs network access or a populated
PyTorch cache. The `summary_stats` extractor is only for fast tests.

Each slide is materialized as:

```text
features/<slide_id>/
  virtual_channels.npy    # [patch, marker, 384]
  virtual_aggregated.npy  # [patch, 384], mean over markers
  he_aligned.npy          # [patch, 512], fusion/H&E runs
  patch_ids.json
  channels.json
```

Formal H&E features are 512-dimensional CONCH patch embeddings. Patch identities
must match the virtual prediction identities; `strict_alignment=true` rejects
missing matches. This avoids combining features from different tissue tiles.

## Models and historical defaults

The public JSON defaults reproduce the settings in the audited internal run:

| Setting | Formal value | Demo value |
| --- | ---: | ---: |
| Cross-validation folds | 5 | 2 |
| Split seed | 1 | 1 |
| Survival time bins | 4 | 4 |
| Epochs | 100 | 1 |
| Bag batch size | 1 | 1 |
| Gradient accumulation | 32 | 2 |
| Optimizer | Adam | Adam |
| Learning rate | 2e-4 | 1e-3 |
| Weight decay | 1e-5 | 1e-5 |
| Survival objective | discrete-time NLL, alpha 0 | same |
| MCAT latent/heads/layers | 256 / 8 / 2 | 16 / 2 / 1 |
| Dropout | 0.25 | 0 |

AMIL is the gated-attention single-bag baseline. `input_modality="he"` consumes
H&E features; `input_modality="virtual"` consumes marker-averaged DINOv2
features. MCAT projects aligned H&E and per-marker virtual features, aggregates
markers with a transformer, applies cross-attention and bag attention, and
fuses both representations before prediction.

Survival bin edges are quantiles of the loaded analysis manifest before fold
training, matching the audited pipeline. Each fold selects the checkpoint with
the lowest validation loss. `inference.split="val"` produces one out-of-fold
prediction for every slide; use this setting for reported cross-validation
metrics.

## Metrics and outputs

Survival evaluation reports both slide-level and patient-level fold-macro
Harrell C-index. When a patient has multiple slides, risks are averaged before
patient evaluation. Patient-cluster bootstrap resampling produces the C-index
confidence interval. Fold-specific median risk groups feed Kaplan-Meier curves,
the Mantel-Cox log-rank test, and a univariable standardized-risk Cox model.

Classification evaluation averages slide probabilities by patient and reports
AUC, accuracy, and macro-F1, together with fold-macro results. AUC is binary for
two classes and macro one-vs-rest for multiclass tasks.

```text
outputs/clinical/<run>/
  resolved_config.clinical.json
  checkpoints/fold_<k>_best.pt
  history/fold_<k>.csv
  training_summary.json
  clinical_predictions.csv
  inference_summary.json
  evaluation/
    summary.json
    survival_slide_per_fold.csv
    survival_patient_per_fold.csv
    patient_risk_predictions.csv
    kaplan_meier.csv
    kaplan_meier.svg
```

Classification runs instead write `classification_per_fold.csv` and
`patient_predictions.csv`.

## Internal source audit

The implementation was checked against the internal benchmark
`downstream/survival` snapshot on 2026-09-05. The most relevant source hashes
are recorded here so the refactor can be traced without publishing private
paths or cohorts:

| Audited source | SHA-256 |
| --- | --- |
| `s0_gen_meta.py` | `fcc3498ed428f5bf179c21cd3fb41c6af043f23784d788e34cb70300222f21e2` |
| `s1_extract_feats.py` | `ac6422d287700a894329b0e2fa30ec50e67249e233a0e91cfd6034e72d3563f4` |
| `s2_merge_codex_he.py` | `93e36f3a33c1d850eada8a6ef739a1cdaac340d3d4cbd62904e99afa1575140a` |
| `s3_make_splits.py` | `5511e5ee617056141f2378d15c958d97fa4188d7d3076c1ee335712793d0ac5b` |
| `s4_clinical_to_survival.py` | `80a73789b942c0e29e73922dea19f5e8b5f709730f15f5452fa563780b3fbc80` |
| `mica/train_mica.py` | `7daae8b37a0490375834210ece04966ea6bf04758b38ed37eaf55e7900172d7c` |
| `mica/test_mica.py` | `47574937d4dc7af1c62e1ec40b23c5e8d9f0746d2d11c38f46666b87a6861315` |
| `mica/core_utils.py` | `38d320df3aee0f6d65443bb9174414bdfe5151711f81cd7f94039e6d3c2d11f1` |
| `mica/datasets/dataset_survival.py` | `4aacc90cf73b837bddb2a21effe0d0befee0cc9505c2e420bda3f7ebe0706827` |
| `mica/models/model_coattn.py` | `d84e643033b6cac26f163f3f3b34de78b9b49a405bb00d8b9716ea722e48b2e5` |
| `mica/models/model_set_mil.py` | `a370971a84b758b77b1fb23a5e9bcd704d4bba3bdc8f010b554691a42f7e93af` |
| `plot_km_mica.py` | `451fccdda935dddcee170ac5da1acc076d77f6fb167a108e9acd6ee6191c8ecb` |
| `cox_compare_mica.py` | `1cdcabb544e53e1fd4f6e7226d1920d1fc9699c771406aabe9965308b3271eb6` |

The refactor removes hard-coded storage locations and replaces the copied
legacy multi-head-attention implementation with current
`torch.nn.MultiheadAttention`. It also adds explicit patient-leakage checks,
machine-readable outputs, patient aggregation, deterministic bootstrap CIs,
and classification metrics. It is intended to reproduce the analysis logic,
not to load historical private checkpoints byte-for-byte.
