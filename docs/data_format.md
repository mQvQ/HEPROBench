# Dataset Format

The demo dataset is stored under `demo_data/`.

```text
demo_data/
  metadata.csv
  channel_names.json
  channel_stats.json
  splits/
    train.csv
    valid.csv
    test.csv
  images/
  targets/
  masks/
  cell_annotations.csv
```

The bundle contains eight 256×256 RGB H&E tiles, eight matching 256×256×4
`uint8` targets (`DAPI`, `CD3`, `CD20`, and `PanCK`), and eight 256×256
`int32` cell-ID masks. The split CSVs contain
four training, two validation, and two test rows. `metadata.csv` retains all
eight rows for the original lightweight demo.

Each CSV contains one row per tile:

```text
slide_name,row,col,split,image_path,target_path,mask_path
```

Required columns:

- `slide_name`: slide identifier used to group patches into per-slide outputs.
- `row`, `col`: tile coordinates in the slide grid.
- `split`: dataset split (`train`, `valid`, or `test`; the retained legacy
  metadata uses `demo`).
- `image_path`: relative path to RGB H&E patch.
- `target_path`: relative path to multiplex target array.
- `mask_path`: relative path to the integer cell-segmentation mask.

Mask value `0` is background. Positive IDs are unique within a slide rather
than globally across the whole cohort, so the stable cell key is
`(slide_name, global_cell_id)`. The same ID may occur in adjacent patches when
a segmented cell crosses a patch boundary; cell expression is aggregated over
all of those pixels.

`cell_annotations.csv` contains one row per cell:

```text
slide_name,global_cell_id,split,DAPI,DAPI_pos,CD3,CD3_pos,...
```

Continuous marker columns provide ground truth for cell-level PCC. The
`<marker>_pos` columns provide binary labels for valid-to-test XGBoost
classification. Demo labels are deterministic median gates within each split
and are intended only for software verification.

`channel_names.json` is a JSON list such as:

```json
["DAPI", "CD3", "CD20", "PanCK"]
```

Targets are stored as NumPy arrays with shape `[H, W, C]` and `uint8` values.
