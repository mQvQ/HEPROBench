# Preprocessing adapters for all benchmark datasets

HEPROBench exposes one preprocessing CLI and nine dataset JSONs. The Python
stages are shared; each JSON fixes the panel, raw-to-canonical channel map,
nuclear and boundary inputs, FOV definition, normalization, QC thresholds, and
data-access status for one benchmark dataset.

```bash
python -m heprobench preprocess \
  --config configs/preprocessing/<dataset>.json \
  --dry-run
```

The repository contains code and fictional manifest schemas only. It does not
contain raw images, patient metadata, real study IDs, private filesystem paths,
or credentials. `dataset.root_dir` can be changed at runtime with
`--set dataset.root_dir=/local/authorized/path`; that resolved path is written
only under the ignored output directory.

## Dataset matrix

| Dataset | JSON | Scenario / FOV | Channels | Access declaration | Parameter audit |
| --- | --- | --- | ---: | --- | --- |
| CRC-CODEX | `crc_codex.json` | co-stained / TMA core | 58 | public, not bundled | verified |
| AML-CODEX | `aml_codex.json` | co-stained / 2048-pixel WSI region | 54 | external, not redistributed | verified |
| MT-CODEX | `mt_codex.json` | co-stained / TMA core | 60 | external, not redistributed | verified |
| HEMIT | `hemit.json` | co-stained / supplied region | 3 | external, not redistributed | verified |
| NSCLC-IMC | `nsclc_imc.json` | adjacent / TMA core | 28 | external, not redistributed | verified |
| CRC-mIHC-P1 | `crc_mihc_p1.json` | co-stained / 2048-pixel WSI region | 7 | controlled/private | partially verified |
| CRC-mIHC-P2 | `crc_mihc_p2.json` | co-stained / 2048-pixel WSI region | 7 | controlled/private | partially verified |
| STAD-mIHC | `stad_mihc.json` | co-stained / TMA core | 11 | controlled/private | source audit pending |
| SPATCH-CODEX | `spatch_codex.json` | adjacent / 2048-pixel WSI region | 17 | external, not redistributed | source audit pending |

`external_not_redistributed` intentionally makes no new public/private claim.
The final accession and license for those cohorts must match the accepted
manuscript Data Availability statement before release.

The three non-verified rows are executable reconstructions of the documented
benchmark-wide protocol, but they are not labelled exact paper-result
reproductions. Their JSONs identify the unresolved Mesmer/source settings with
`parameter_status`; those fields should be updated only from archived source
logs or a confirmed upstream release. This distinction prevents undocumented
guesses from being presented to reviewers as historical settings.

## Privacy-safe input contract

Copy the matching file from `configs/preprocessing/templates/` to
`<dataset.root_dir>/manifests/slides.csv`, then replace the fictional rows
locally. Required columns are:

```text
sample_id,slide_id,fov_id,group_id,he_path,target_path,origin_y_px,origin_x_px,split
```

- `sample_id` uniquely identifies one input image pair.
- `slide_id` is the downstream per-slide aggregation key.
- `fov_id` is the Mesmer grouping key for an already cropped TMA core or region.
- `group_id` is a deidentified patient/slide grouping key used to prevent split
  leakage. It must not be an MRN or another direct identifier.
- `origin_y_px` and `origin_x_px` are zero for a whole-slide input or an FOV
  represented as its own `slide_id`. They are required when multiple supplied
  FOV files share a downstream slide and therefore need non-overlapping global
  coordinates.
- `split` should contain the released paper assignment. When blank, the code
  creates a deterministic grouped 70/10/20 split; this is useful for a new
  dataset but does not reproduce the paper split.

The manifest is deliberately not committed. For controlled cohorts, keep it
inside the authorized storage boundary and use pseudonymous values. Generated
stage reports remain under `outputs/`, which is ignored by Git.

## Dataset-specific behavior

WSI configs split each registered image into non-overlapping 2048x2048 FOVs,
then into 256x256 tiles. TMA and supplied-region configs treat the entire input
pair as one FOV. Mesmer runs on the FOV mosaic so that a cell crossing a tile
boundary retains one label. Labels from different FOVs are offset so
`(slide_name, global_cell_id)` remains unique for cell evaluation.

HEMIT explicitly records the physical source order `[PanCK, CD3, DAPI]` and
selects raw indices `[2, 0, 1]` to produce the canonical order
`[DAPI, PanCK, CD3]`. It uses the historical pre-normalized target arrays and
does not pass `image_mpp` to Mesmer. The other verified adapters retain the
historical 99.9-percentile `log1p` transform, including its lack of division by
`log(2)`, and preserve the input array container dtype while mapping values to
the benchmark 0--255 range.

Run the registration/manual-QC phase first:

```bash
python -m heprobench preprocess \
  --config configs/preprocessing/aml_codex.json \
  --stage split --stage register --stage registration-qc
```

After setting every `decision` to `accepted` or `rejected` in the generated QC
CSV, run the image and cell phases:

```bash
python -m heprobench preprocess \
  --config configs/preprocessing/aml_codex.json \
  --stage tile --stage normalize --stage patch-qc \
  --stage segment --stage cell-extract --stage gate
```

For HEMIT, `registration.method=pre_registered` passes the supplied registered
pair through without invoking VALIS. `image_mpp: null` means the argument is
omitted from `Mesmer.predict`, matching its retained segmentation call.

CRC-CODEX remains the detailed end-to-end tutorial and real reviewer demo.
The other adapters are first-class executable configs, not illustrative
pseudocode, but their datasets are not redistributed by this repository.
