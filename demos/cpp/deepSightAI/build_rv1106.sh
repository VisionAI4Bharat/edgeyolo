#!/usr/bin/env bash
# Cross-compile deepSightAI_Config_Server for Luckfox Pico RV1106 (ARM Cortex-A7, uclibc).
#
# Usage:
#   ./build_rv1106.sh           # configure + build
#   ./build_rv1106.sh clean     # wipe build directory first
#   ./build_rv1106.sh install   # install to install_rv1106/
#
# Prerequisites:
#   - Luckfox SDK at /home/abhinav/luckfox/luckfox-pico  (or export LUCKFOX_SDK_PATH)
#   - third_party/rknn/       and  third_party/luckfox_mpi/  already populated

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build/rv1106"
INSTALL_DIR="${SCRIPT_DIR}/install/rv1106"

SDK_PATH="${LUCKFOX_SDK_PATH:-/home/abhinav/luckfox/luckfox-pico}"
BR_SYSROOT="${SDK_PATH}/sysdrv/source/buildroot/buildroot-2023.02.6/output/host/arm-buildroot-linux-uclibcgnueabihf/sysroot"

RKNN_ROOT="${SCRIPT_DIR}/third_party/rockchip/RV110X/rknn"
LUCKFOX_MPI="${SCRIPT_DIR}/third_party/rockchip/RV110X/mpi"

# RV110X ships an ARM OpenCV 4.10 inside the mpi bundle
OPENCV_DIR="${LUCKFOX_MPI}/lib/cmake/opencv4"

if [[ "${1:-}" == "clean" ]]; then
    echo "[build_rv1106] Removing ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
    shift
fi

echo "[build_rv1106] Configuring..."
cmake -S "${SCRIPT_DIR}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/cmake/toolchain-rv1106.cmake" \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_RKNN=ON \
      -DWITH_OPENVINO=OFF \
      -DWITH_ONNXRT=OFF \
      -DRKNN_ROOT="${RKNN_ROOT}" \
      -DLUCKFOX_SDK_ROOT="${LUCKFOX_MPI}" \
      -DOpenCV_DIR="${OPENCV_DIR}" \
      -DCMAKE_PREFIX_PATH="${BR_SYSROOT}/usr" \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,${LUCKFOX_MPI}/lib:${BR_SYSROOT}/usr/lib" \
      "$@"

echo "[build_rv1106] Building..."
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

echo ""
echo "[build_rv1106] Done.  Binaries in: ${BUILD_DIR}"
echo "  To install:  cmake --install ${BUILD_DIR}"
echo "  To deploy:   scp ${BUILD_DIR}/deepSightAI-headless root@<device-ip>:/usr/bin/"
echo ""
echo "  On device, run:"
echo "    deepSightAI-headless --config /etc/deepSightAI/config.yaml"
echo "  Then open http://<device-ip>:8080 to configure."
