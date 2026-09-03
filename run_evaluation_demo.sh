#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
CONFIG_PATH="${1:-configs/demo/pix2pix.json}"
PROFILE_DEVICE="${2:-cpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/full_evaluation_demo}"
SUBMISSION_ROOT="${OUTPUT_ROOT}/submission"
PROFILE_ROOT="${OUTPUT_ROOT}/profile"

for split in train valid test; do
  "${PYTHON_BIN}" -m heprobench validate-data \
    --config "${CONFIG_PATH}" \
    --split "${split}"
done

"${PYTHON_BIN}" scripts/prepare_demo_submission.py \
  --config "${CONFIG_PATH}" \
  --output "${SUBMISSION_ROOT}"

"${PYTHON_BIN}" -m heprobench profile \
  --config "${CONFIG_PATH}" \
  --output-dir "${PROFILE_ROOT}" \
  --device "${PROFILE_DEVICE}"

"${PYTHON_BIN}" -m heprobench evaluate \
  --config "${CONFIG_PATH}" \
  --pred-dir "${SUBMISSION_ROOT}" \
  --efficiency-json "${PROFILE_ROOT}/efficiency.json"

echo "Full evaluation demo complete: ${SUBMISSION_ROOT}/test/summary.json"
