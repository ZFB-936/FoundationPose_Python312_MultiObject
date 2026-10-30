#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Only the pose-clustering extension is needed for model-based estimation.
cmake -S "${PROJ_ROOT}/mycpp" -B "${PROJ_ROOT}/mycpp/build" \
  -DPYTHON_EXECUTABLE="$(command -v python)" "$@"
cmake --build "${PROJ_ROOT}/mycpp/build" --parallel "$(nproc)"
