#!/bin/bash
# Install dependencies for EdgeYOLO C++ demo on resource-constrained devices
# Run as: sudo ./install_dependencies.sh

set -e  # Exit on any error

echo "Installing EdgeYOLO C++ demo dependencies for resource-constrained device..."

# Update package list
apt-get update

# Install core build tools
apt-get install -y \
    cmake \
    git \
    build-essential \
    pkg-config

# Install Qt6 core modules only (skip multimedia for resource constraints)
echo "Installing Qt6 core modules..."
apt-get install -y \
    qt6-base-dev \
    qt6-tools-dev \
    # Skip multimedia to save resources: qt6-multimedia-dev \
    # Skip tools to save resources: qt6-tools-dev-tools \

# Install essential libraries only
apt-get install -y \
    libonnxruntime-dev \
    libopencv-dev \
    libyaml-cpp-dev

echo "Dependencies installed successfully!"
echo ""
echo "To build the demo:"
echo "  cd demos/cpp/edgeyolo_qt_gui"
echo "  mkdir -p build && cd build"
echo ""
echo "# Build with Qt6 core modules (multimedia disabled):"
echo "  cmake .."
echo ""
echo "# If Qt6 not found or to force Qt5:"
echo "  cmake .. -DUSE_QT5=ON"
echo ""
echo "# Build with parallel jobs (adjust based on available cores)"
echo "  make -j\$(nproc)"
echo ""
echo "Note: Only essential Qt modules (Core, Gui, Widgets) are installed."
echo "Multimedia and other non-essential modules are skipped to save resources."
echo "OpenCV, yaml-cpp, and ONNX Runtime are installed from system packages."
echo "OpenVINO and RKNN backends are disabled by default - enable via cmake flags if needed."