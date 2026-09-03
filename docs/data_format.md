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
```

The bundle contains eight 256×256 RGB H&E tiles and eight matching 256×256×4
`uint8` targets (`DAPI`, `CD3`, `CD20`, and `PanCK`). The split CSVs contain
four training, two validation, and two test rows. `metadata.csv` retains all
eight rows for the original lightweight demo.

Each CSV contains one row per tile:

```text
slide_name,row,col,split,image_path,target_path
```

Required columns:

- `slide_name`: slide identifier used to group patches into per-slide outputs.
- `row`, `col`: tile coordinates in the slide grid.
- `split`: dataset split (`train`, `valid`, or `test`; the retained legacy
  metadata uses `demo`).
- `image_path`: relative path to RGB H&E patch.
- `target_path`: relative path to multiplex target array.

`channel_names.json` is a JSON list such as:

```json
["DAPI", "CD3", "CD20", "PanCK"]
```

Targets are stored as NumPy arrays with shape `[H, W, C]` and `uint8` values.
