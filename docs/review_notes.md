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
- conda and pip environment files.
- a fully specified CRC-CODEX preprocessing/QC reference pipeline, including
  VALIS/manual registration QC, legacy normalization, robust-NMI filtering,
  Mesmer, cell extraction, and GMM marker QC.

It intentionally does not include:

- original WSI datasets;
- full pretrained model checkpoints;
- full training outputs or experiment logs;
- private paths or unreleased data.

CRC-CODEX is currently the only cohort with an end-to-end preprocessing JSON.
The other eight cohort adapters and the de-identified exact study split
manifest remain required before claiming full benchmark-wide preprocessing
reproducibility.

Remote foundation-model revisions/checksums and redistribution permission for
source trees without an explicit upstream license remain release-blocking
provenance items; see `UPSTREAM.md` and `methods/pfm/specs.json`.

The included synthetic data and generated demo checkpoints are only for
checking that the software reaches each retained training loop. They are not
scientific benchmark results.
