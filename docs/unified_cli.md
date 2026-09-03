# Unified CLI and JSON Experiments

The external interface is always:

```bash
python -m heprobench train --config EXPERIMENT.json
python -m heprobench infer --config EXPERIMENT.json
```

`method.name` in the JSON selects the runner. `--method METHOD` remains an
optional explicit override and must agree with the JSON value. It is still
required by the retained legacy YAML demo.

Use `--dry-run` to write the resolved configuration and print both the runner
command and the final native command without importing a model stack or
starting training. Each run records `resolved_config.<task>.json`, the
materialized native config when applicable, and `native_command.json`.

## Shared JSON fields

- `method.name`: registry key.
- `data`: common dataset paths plus method-specific dataset settings.
- `model`: architecture, encoder, and checkpoint selection.
- `train` and `inference`: common task parameters.
- `runtime`: Python executable and device.
- `output.run_dir`: audit files and method artifacts.
- `native.train` / `native.infer`: parameters that belong only to the retained
  original program.

Shared fields are the public control surface. For example,
`train.batch_size`, `train.learning_rate`, and `runtime.device` are translated
to the corresponding native arguments and take precedence over defaults under
`native.train`. This means the following command really launches ROSIE with a
batch size of 3:

```bash
python -m heprobench train \
  --config configs/demo/rosie.json \
  --batch-size 3
```

Relative paths are resolved from the user-authored JSON, not from the runner's
working directory. `extends` accepts a JSON path or a list of paths and performs
a recursive object merge. This is how all foundation-model experiments inherit
one DPT comparison protocol.

## Overrides

```bash
python -m heprobench train \
  --config configs/experiments/hex.json \
  --device cuda:1 \
  --batch-size 8 \
  --set train.max_iters=1000 \
  --set native.train.launcher.nproc_per_node=1 \
  --dry-run
```

`--set` values are parsed as JSON when possible. Overrides are included in the
resolved-config SHA-256 recorded under `_heprobench`.

## Native routing

The method registry selects a small `runner.py`. The runner translates common
fields, writes a native JSON/YAML when required, and starts the method's own
program. Plain Python, Hydra, and `torchrun` launches are supported. See
`docs/methods.md` for the full routing table.

Inference outputs are written under:

```text
<output.run_dir>/predictions/<method_name>/<split>/*.h5
```

All formal inference scripts write `heprobench_h5_v1`. The same experiment JSON
can be passed to validation/evaluation:

```bash
python -m heprobench validate-submission \
  --config configs/experiments/rosie.json \
  --pred-dir outputs/rosie/predictions/ROSIE/test

python -m heprobench evaluate \
  --config configs/experiments/rosie.json \
  --pred-dir outputs/rosie/predictions/ROSIE/test
```

## Foundation-model authentication

Gated Hugging Face models read `HF_TOKEN` from the environment. Tokens must not
be written into source files or JSON. Local-only weights are referenced by path
in their encoder JSON. `methods/pfm/specs.json` explicitly marks revisions and
checksums that still need pinning before an archival reproducibility release.

The `miphei_vit` and `dpt_fm` demo JSON files deliberately use an unpretrained
H0-mini encoder. They exercise the retained decoder and training pipeline
without downloading a gated multi-gigabyte checkpoint; formal encoder JSONs
retain their pretrained foundation-model settings.
