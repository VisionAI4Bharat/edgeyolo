#!/usr/bin/env bash
# Copyright (C) 2026 swatah.ai. All rights reserved.
#
# Dual-licensed: GPLv3 (non-commercial) / proprietary (commercial).
# See LICENSE for details.
#
# Build deepSightAI-desktop — Qt desktop GUI with OpenVINO + ONNX Runtime (x86-64).
#
# Usage:
#   ./build_desktop.sh           # configure + build
#   ./build_desktop.sh clean     # wipe build/desktop first
#   ./build_desktop.sh install   # install to install/desktop/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build/desktop"
INSTALL_DIR="${SCRIPT_DIR}/install/desktop"

if [[ "${1:-}" == "clean" ]]; then
    echo "[build_desktop] Removing ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
    shift
fi

echo "[build_desktop] Configuring..."
cmake -S "${SCRIPT_DIR}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_QT_GUI=ON \
      -DBUILD_HEADLESS=OFF \
      -DWITH_OPENVINO=ON \
      -DWITH_ONNXRT=ON \
      -DWITH_RKNN=OFF \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      "$@"

echo "[build_desktop] Building..."
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

echo ""
echo "[build_desktop] Done.  Binary: ${BUILD_DIR}/deepSightAI-desktop"
echo "  To install: cmake --install ${BUILD_DIR}"
