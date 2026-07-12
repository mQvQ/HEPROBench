# Submission Format

HEPROBench predictions are written as one HDF5 file per slide.

Required root attribute:

```text
schema_version = "heprobench_h5_v1"
```

Required datasets under `/data`:

```text
pred          uint8 [N, H, W, C]
rows          int32 [N]
cols          int32 [N]
grid_index    int64 [n_rows, n_cols]
channel_names string [C]
image_paths   string [N]
target_paths  string [N]
has_gt        bool [N]
```

The CLI validates that tile coordinates and channel names match the dataset
metadata:

```bash
python -m heprobench validate-submission \
  --config configs/demo.yaml \
  --pred-dir outputs/miphei_vit
```

