#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${1:-cpu}

"$PYTHON_BIN" scripts/prepare_clinical_demo_data.py
"$PYTHON_BIN" -m heprobench clinical \
  --config configs/demo/clinical/survival_he_only.json \
  --stage make-splits \
  --device "$DEVICE"

for CONFIG in \
  configs/demo/clinical/survival_he_only.json \
  configs/demo/clinical/survival_virtual.json \
  configs/demo/clinical/survival_fusion.json
do
  "$PYTHON_BIN" -m heprobench clinical \
    --config "$CONFIG" \
    --stage train \
    --stage infer \
    --stage evaluate \
    --device "$DEVICE"
done

"$PYTHON_BIN" -m heprobench clinical \
  --config configs/demo/clinical/classification_fusion.json \
  --stage make-splits \
  --stage train \
  --stage infer \
  --stage evaluate \
  --device "$DEVICE"

echo "Clinical demo complete: outputs/clinical_demo"
