#!/usr/bin/env bash
set -euo pipefail

${PYTHON:-python} scripts/prepare_demo_data.py

${PYTHON:-python} -m heprobench validate-data --config configs/demo.yaml

${PYTHON:-python} -m heprobench train \
  --config configs/demo.yaml \
  --method configs/methods/miphei_vit.yaml \
  --output-checkpoint outputs/miphei_vit_demo_trained.pt \
  --epochs 1

for method in \
  miphei_vit \
  dpt_fm_hoptimus0 \
  dpt_fm_conch \
  cut \
  hex \
  gigatime \
  pytorch_cyclegan_and_pix2pix \
  rosie
do
  echo "Running ${method}"
  ${PYTHON:-python} -m heprobench infer \
    --config configs/demo.yaml \
    --method configs/methods/${method}.yaml \
    --output outputs/${method}
  ${PYTHON:-python} -m heprobench validate-submission \
    --config configs/demo.yaml \
    --pred-dir outputs/${method}
  ${PYTHON:-python} -m heprobench evaluate \
    --config configs/demo.yaml \
    --pred-dir outputs/${method}
done

echo "Demo complete. Metrics are under outputs/<method>/metrics.csv"

