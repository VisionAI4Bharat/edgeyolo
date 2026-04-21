#!/usr/bin/env bash
# Copyright (C) 2026 swatah.ai. All rights reserved.
#
# Dual-licensed: GPLv3 (non-commercial) / proprietary (commercial).
# See LICENSE for details.
#
# Build deepSightAI_Config_Server — headless x86-64 server build
# (no Qt, OpenVINO + ONNX Runtime inference backends).
#
# Usage:
#   ./build_headless.sh           # configure + build
#   ./build_headless.sh clean     # wipe build/headless first
#   ./build_headless.sh install   # install to install/headless/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build/headless"
INSTALL_DIR="${SCRIPT_DIR}/install/headless"

if [[ "${1:-}" == "clean" ]]; then
    echo "[build_headless] Removing ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
    shift
fi

echo "[build_headless] Configuring..."
cmake -S "${SCRIPT_DIR}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_QT_GUI=OFF \
      -DBUILD_HEADLESS=ON \
      -DWITH_OPENVINO=ON \
      -DWITH_ONNXRT=ON \
      -DWITH_RKNN=OFF \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      "$@"

echo "[build_headless] Building..."
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

echo ""
echo "[build_headless] Done.  Binary: ${BUILD_DIR}/deepSightAI-headless"
echo "  To install: cmake --install ${BUILD_DIR}"
echo "  Run:        ${BUILD_DIR}/deepSightAI-headless --config /path/to/config.yaml"
echo "  Dashboard:  http://localhost:8080"
