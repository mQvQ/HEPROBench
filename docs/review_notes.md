# Review Package Notes

The repository contains two clearly separated layers: the original lightweight
synthetic review demo and the formal method-specific framework added in the
current repository. The original review state is preserved by the
`review-demo-before-unified-framework` Git tag.

It includes:

- runnable demo data;
- retained training and inference implementations for all registered methods;
- one JSON CLI with method and encoder registries;
- original and regression GigaTIME variants under distinct names;
- HistoPlexer Gaussian-pyramid and patch-wise contrastive objectives;
- tiny checkpoints for the retained legacy demo and one-step checkpoint
  generation through the native demo JSONs;
- standardized HDF5 output;
- validation and evaluation CLI commands, including paper image metrics,
  slide-macro aggregation, cell-level PCC/classification, and native efficiency;
- bundled integer cell-ID masks and synthetic valid/test cell annotations for a
  complete reviewer-side evaluation smoke test;
- conda and pip environment files;
- an executable native train-to-inference-to-evaluation verification matrix for
  all ten registered methods, with a machine-readable pass/fail report;
- a fully specified CRC-CODEX preprocessing/QC reference pipeline, including
  VALIS/manual registration QC, legacy normalization, robust-NMI filtering,
  Mesmer, cell extraction, and GMM marker QC;
- privacy-safe preprocessing JSONs and adapters for the other eight benchmark
  datasets, with access and parameter-audit status stated explicitly;
- a privacy-safe refactor of the complete downstream clinical path: HDF5
  feature extraction, patch alignment, patient-level folds, AMIL/MCAT
  survival/classification training, held-out inference, patient aggregation,
  C-index/bootstrap/KM/log-rank/Cox, and classification metrics;
- four runnable synthetic clinical checks covering H&E-only, virtual-only, and
  fusion survival plus fusion classification.

It intentionally does not include:

- original WSI datasets;
- full pretrained model checkpoints;
- full training outputs or experiment logs;
- private paths or unreleased data.

Real clinical records, patient/slide mappings, cohort split manifests, and
historical clinical checkpoints remain outside the repository. Their absence
does not hide implementation choices: the full executable pipeline, formal
defaults, input schemas, internal source hashes, and synthetic end-to-end check
are documented in `docs/clinical_evaluation.md`.

All nine cohorts now have executable preprocessing JSONs. CRC-CODEX is the
detailed tutorial and published real-data reviewer demo; the other eight are
reference implementations because their data are not redistributed. Three
cohorts still have explicitly marked historical-parameter audit gaps, and
deidentified exact study split manifests remain required before claiming exact
paper-result reproduction for every cohort. See `docs/preprocessing_datasets.md`.

Remote foundation-model revisions/checksums and redistribution permission for
source trees without an explicit upstream license remain release-blocking
provenance items; see `UPSTREAM.md` and `methods/pfm/specs.json`.

The included synthetic data and generated demo checkpoints are only for
checking that the software reaches each retained training loop. They are not
scientific benchmark results.

Run `bash run_all_method_demos.sh --device cuda:0` to reproduce the complete
native software check. The default evaluation omits LPIPS/DISTS for speed but
retains image, slide-macro, and cell-level evaluation; `--perceptual` and
`--profile` exercise the optional perceptual and computational-efficiency
paths.
