# Pathology Foundation Models

This folder is retained unchanged for the original lightweight review demo.
Its tiny encoders are used only by the legacy YAML workflow.

The runnable demo uses tiny stand-in encoders and synthetic checkpoints so the
package can be distributed as a small zip file. Full external foundation model
weights are not redistributed in this review package.

See `registry.json` for the names accepted by `configs/methods/dpt_fm_*.yaml`.
The formal pipelines use `methods/pfm/specs.json` and the JSON experiments in
`configs/foundation_models/` instead.
