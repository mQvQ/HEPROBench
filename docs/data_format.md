# Dataset Format

The demo dataset is stored under `demo_data/`.

```text
demo_data/
  metadata.csv
  channel_names.json
  images/
  targets/
```

`metadata.csv` contains one row per tile:

```text
slide_name,row,col,split,image_path,target_path
```

Required columns:

- `slide_name`: slide identifier used to group patches into per-slide outputs.
- `row`, `col`: tile coordinates in the slide grid.
- `split`: dataset split. The review demo uses `demo`.
- `image_path`: relative path to RGB H&E patch.
- `target_path`: relative path to multiplex target array.

`channel_names.json` is a JSON list such as:

```json
["DAPI", "CD3", "CD20", "PanCK"]
```

Targets are stored as NumPy arrays with shape `[H, W, C]` and `uint8` values.

