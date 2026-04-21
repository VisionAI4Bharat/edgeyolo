# Copyright (C) 2026 swatah.ai. All rights reserved.
#
# This software is dual-licensed:
# 1. GNU General Public License v3.0 (GPLv3)
# 2. A proprietary license for commercial use.
#
# You may use this software under the terms of the GPLv3 if you are using it
# for non-commercial purposes. For commercial usage, a separate commercial 
# license must be obtained from swatah.ai (info@swatah.ai).
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
# for more details.
#
# Trademarks: All trademarks, service marks, and logos are the property of 
# their respective owners.

# Cross-compile toolchain for Luckfox Pico RV1106 (ARM Cortex-A7, uclibc)
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-rv1106.cmake \
#         -DWITH_RKNN=ON [other options] -B build_rv1106 -S .

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Resolve toolchain root — prefer env var so CI can override without editing this file
if(DEFINED ENV{LUCKFOX_SDK_PATH})
    set(_TC_ROOT "$ENV{LUCKFOX_SDK_PATH}/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf")
else()
    set(_TC_ROOT "/home/abhinav/luckfox/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf")
endif()

set(CMAKE_C_COMPILER   "${_TC_ROOT}/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc")
set(CMAKE_CXX_COMPILER "${_TC_ROOT}/bin/arm-rockchip830-linux-uclibcgnueabihf-g++")
set(CMAKE_AR           "${_TC_ROOT}/bin/arm-rockchip830-linux-uclibcgnueabihf-ar"   CACHE FILEPATH "Archiver")
set(CMAKE_STRIP        "${_TC_ROOT}/bin/arm-rockchip830-linux-uclibcgnueabihf-strip" CACHE FILEPATH "Strip")

# Sysroot for correct libc / header resolution
set(CMAKE_SYSROOT "${_TC_ROOT}/arm-rockchip830-linux-uclibcgnueabihf/sysroot")

# Buildroot sysroot contains Qt5, zlib, libpng etc. for the target.
# Allow CMAKE_FIND_ROOT_PATH to be extended by the build script.
if(DEFINED ENV{LUCKFOX_SDK_PATH})
    set(_BR_SYSROOT "$ENV{LUCKFOX_SDK_PATH}/sysdrv/source/buildroot/buildroot-2023.02.6/output/host/arm-buildroot-linux-uclibcgnueabihf/sysroot")
else()
    set(_BR_SYSROOT "/home/abhinav/luckfox/luckfox-pico/sysdrv/source/buildroot/buildroot-2023.02.6/output/host/arm-buildroot-linux-uclibcgnueabihf/sysroot")
endif()

list(APPEND CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}" "${_BR_SYSROOT}")

# Don't search host paths when cross-compiling
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# RV1106 is ARM Cortex-A7 with NEON — enable hard-float and NEON
set(CMAKE_C_FLAGS_INIT   "-march=armv7-a -mfpu=neon -mfloat-abi=hard")
set(CMAKE_CXX_FLAGS_INIT "-march=armv7-a -mfpu=neon -mfloat-abi=hard")

# RV1106 platform defines expected by Rockchip SDK headers
add_compile_definitions(
    RV1106_1103
    ISP_HW_V30
    RKPLATFORM=1
    ARCH64=0
    UAPI2
    _LARGEFILE_SOURCE
    _LARGEFILE64_SOURCE
    _FILE_OFFSET_BITS=64
)
