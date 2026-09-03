# Review Package Notes

The repository contains two clearly separated layers: the original lightweight
synthetic review demo and the formal method-specific framework added in the
`unified-framework` branch.

It includes:

- runnable demo data;
- retained training and inference implementations for all registered methods;
- one JSON CLI with method and encoder registries;
- original and regression GigaTIME variants under distinct names;
- HistoPlexer Gaussian-pyramid and patch-wise contrastive objectives;
- tiny synthetic checkpoints;
- standardized HDF5 output;
- validation and evaluation CLI commands;
- conda and pip environment files.

It intentionally does not include:

- original WSI datasets;
- full pretrained model checkpoints;
- full training outputs or experiment logs;
- private paths or unreleased data.

Remote foundation-model revisions/checksums and redistribution permission for
source trees without an explicit upstream license remain release-blocking
provenance items; see `UPSTREAM.md` and `methods/pfm/specs.json`.

The included synthetic data and tiny checkpoints are only for checking that the
software runs end-to-end.
