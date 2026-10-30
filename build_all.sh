#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=$(command -v python)
PYTHON_TAG=$("${PYTHON_BIN}" -c 'import sys; print(sys.implementation.cache_tag)')
export MAX_JOBS=${MAX_JOBS:-4}

# Fail before compiling if the active environment lacks build dependencies.
"${PYTHON_BIN}" -c 'import torch, setuptools, wheel, ninja, pybind11'

# Build the vendored renderer offline, without artifacts from another environment.
BUILD_TMP=$(mktemp -d "${TMPDIR:-/tmp}/foundationpose-build.XXXXXXXX")
trap 'rm -rf -- "${BUILD_TMP}"' EXIT
cp -a "${PROJ_ROOT}/third_party/nvdiffrast" "${BUILD_TMP}/nvdiffrast"
"${PYTHON_BIN}" -m pip install --no-index --no-deps --no-build-isolation \
  --force-reinstall "${BUILD_TMP}/nvdiffrast"

# Build pose clustering separately for each Python ABI.
cmake -S "${PROJ_ROOT}/mycpp" -B "${PROJ_ROOT}/mycpp/build/${PYTHON_TAG}" \
  -DPython_EXECUTABLE="${PYTHON_BIN}" \
  -Dpybind11_DIR="$("${PYTHON_BIN}" -m pybind11 --cmakedir)" "$@"
cmake --build "${PROJ_ROOT}/mycpp/build/${PYTHON_TAG}" --parallel "${MAX_JOBS}"
