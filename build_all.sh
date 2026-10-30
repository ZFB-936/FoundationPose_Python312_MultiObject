#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "${PROJ_ROOT}/build_all_conda.sh" "$@"
