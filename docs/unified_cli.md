# Unified CLI and JSON Experiments

The external interface is always:

```bash
python -m heprobench train --config EXPERIMENT.json
python -m heprobench infer --config EXPERIMENT.json
```

All ten bundled method JSONs are exercised through train, valid/test inference,
HDF5 validation, and evaluation by:

```bash
bash run_all_method_demos.sh --device cuda:0
```

The script and its machine-readable verification report are described in
[the native E2E demo guide](end_to_end_demo.md).

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
- `model.checkpoint_path` / `model.checkpoint_dir`: the stable checkpoint read
  by a later inference command. HEPRO-derived native training retains its
  timestamped log directory and also publishes `config.yaml` plus
  `model.weights.ckpt` here.
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
  --pred-dir outputs/rosie/predictions/ROSIE
```

Cell classification requires validation predictions for fitting XGBoost and
test predictions for scoring. If both are below one parent directory, only the
parent is needed:

```text
predictions/
  valid/<slide>.h5
  test/<slide>.h5
```

```bash
python -m heprobench evaluate \
  --config configs/demo/rosie.json \
  --pred-dir outputs/evaluation_demo
```

Use `--valid-pred-dir` when validation predictions live elsewhere. Demo JSONs
enable cell evaluation and paper image metrics including LPIPS/DISTS; use
`--no-cells` or `--no-perceptual` for targeted checks.

Computational efficiency uses the same method JSON and its native architecture:

```bash
python -m heprobench profile \
  --config configs/demo/rosie.json \
  --output-dir outputs/demo/rosie/profile \
  --device cuda:0

python -m heprobench evaluate \
  --config configs/demo/rosie.json \
  --pred-dir outputs/demo/rosie/predictions \
  --efficiency-json outputs/demo/rosie/profile/efficiency.json
```

The profile records parameters, forward FLOPs, and inference latency together
with hardware and input-shape metadata. `--dry-run` prints both native profiler
commands without importing the method stack.

## Clinical evaluation

The downstream clinical workflow uses the same configuration pattern and a
dedicated subcommand:

```bash
python -m heprobench clinical \
  --config configs/demo/clinical/survival_fusion.json \
  --stage make-splits \
  --stage train \
  --stage infer \
  --stage evaluate \
  --device cpu
```

`clinical.task` selects survival or classification, `model.name` selects AMIL
or MCAT, and `model.input_modality` selects H&E, virtual protein, or both. The
same `--set KEY=VALUE`, `--device`, `--output-dir`, and `--dry-run` controls are
available. `--stage all` expands to `pipeline.stages` in the JSON.

The runnable matrix is:

```bash
PYTHON_BIN=python bash run_clinical_demo.sh cpu
```

See [the clinical evaluation reference](clinical_evaluation.md) for the HDF5,
feature-bag, outcome-manifest, split, metric, and privacy contracts.

## Preprocessing reference

CRC-CODEX preprocessing uses a dataset JSON rather than a method JSON:

```bash
python -m heprobench preprocess \
  --config configs/preprocessing/crc_codex.json \
  --dry-run
```

`--stage` may be repeated to run selected stages. Registration and its manual
QC should be run first; after every FOV is accepted or rejected, run the image
and cell stages:

```bash
python -m heprobench preprocess \
  --config configs/preprocessing/crc_codex.json \
  --stage tile --stage normalize --stage patch-qc \
  --stage segment --stage cell-extract --stage gate
```

All parameters support `--set KEY=VALUE`, for example
`--set patch_qc.robust_nmi_min=0.03`. See
[the CRC-CODEX preprocessing reference](preprocessing_crc_codex.md) for the
input contract, manual-QC checkpoint, scientific definitions, and outputs.

## Foundation-model authentication

Gated Hugging Face models read `HF_TOKEN` from the environment. Tokens must not
be written into source files or JSON. Local-only weights are referenced by path
in their encoder JSON. `methods/pfm/specs.json` explicitly marks revisions and
checksums that still need pinning before an archival reproducibility release.

The `miphei_vit` and `dpt_fm` demo JSON files deliberately use an unpretrained
H0-mini encoder. They exercise the retained decoder and training pipeline
without downloading a gated multi-gigabyte checkpoint; formal encoder JSONs
retain their pretrained foundation-model settings.
