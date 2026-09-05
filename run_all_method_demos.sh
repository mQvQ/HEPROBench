#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

python_bin="${PYTHON:-python}"
exec "$python_bin" scripts/run_all_method_demos.py "$@"
