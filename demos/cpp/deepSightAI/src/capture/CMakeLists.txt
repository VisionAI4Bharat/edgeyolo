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

cmake_minimum_required(VERSION 3.16)

# Standalone project declaration allows this directory to be built
# independently with:
#   cmake -S src/capture -B build_capture -DWITH_RKNN=ON -DLUCKFOX_SDK_ROOT=...
# When included via add_subdirectory() from the parent CMakeLists.txt the
# parent's cached variables (WITH_RKNN, LUCKFOX_SDK_ROOT, OpenCV_DIR) take
# precedence over the option() defaults here.
project(DeepSightCapture VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── build options ────────────────────────────────────────────────────────────
option(WITH_RKNN "Enable Rockchip RK MPI hardware camera capture" OFF)

set(LUCKFOX_SDK_ROOT "" CACHE PATH
    "Rockchip / Luckfox SDK root (contains include/, lib/, common/ with RK MPI + RKAIQ)")

# ─── dependencies ─────────────────────────────────────────────────────────────
# When built standalone OpenCV must be found via -DOpenCV_DIR=... or system.
# When built as a subdirectory the parent has already run find_package(OpenCV).
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED)
endif()

# ─── library target ───────────────────────────────────────────────────────────
add_library(deepsight_capture SHARED rockchip_capture.cpp)

set_target_properties(deepsight_capture PROPERTIES
    PUBLIC_HEADER rockchip_capture.h
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(deepsight_capture
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/deepsight>
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(deepsight_capture PUBLIC ${OpenCV_LIBS})

# ─── Rockchip RK MPI / RKAIQ (cross-compile for RV1106 / RK3588) ─────────────
if(WITH_RKNN)
    if(NOT LUCKFOX_SDK_ROOT)
        message(FATAL_ERROR
            "WITH_RKNN=ON requires -DLUCKFOX_SDK_ROOT=/path/to/luckfox_sdk\n"
            "  The SDK root must contain:\n"
            "    include/       (rk_mpi_vi.h, rk_mpi_sys.h, …)\n"
            "    include/rkaiq/ (RKAIQ ISP headers)\n"
            "    common/isp3.x/ (sample_comm.h wrapper)\n"
            "    lib/           (librockit.so, librkaiq.so, libsample_comm.a, …)")
    endif()

    message(STATUS "[deepsight_capture] Rockchip hardware enabled")
    message(STATUS "[deepsight_capture] SDK root: ${LUCKFOX_SDK_ROOT}")

    target_compile_definitions(deepsight_capture
        PUBLIC  WITH_RKNN                      # exposed to consumers
        PRIVATE
            ISP_HW_V30                         # selects isp3.x code paths in sample_comm
            RV1106_1103                        # RV1106/RV1103 chip family
            RKPLATFORM=1                       # enables Rockchip MPI paths
    )

    # RV1106 is 32-bit ARM Cortex-A7 — enable NEON for OpenCV colour convert
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM")
        target_compile_options(deepsight_capture PRIVATE -mfpu=neon -mfloat-abi=hard)
    endif()

    # luckfox_mpi layout: all headers (rk_mpi_*.h, sample_comm.h, rkaiq/) live
    # directly under include/ — there is no separate common/ subdirectory.
    # RKAIQ headers use flat #include "Foo.h" style across sub-packages, so we
    # need every sub-directory of rkaiq/ on the search path.
    file(GLOB_RECURSE _RKAIQ_SUBDIRS
         LIST_DIRECTORIES true
         "${LUCKFOX_SDK_ROOT}/include/rkaiq/*")
    set(_RKAIQ_INCDIRS "")
    foreach(_d ${_RKAIQ_SUBDIRS})
        if(IS_DIRECTORY "${_d}")
            list(APPEND _RKAIQ_INCDIRS "${_d}")
        endif()
    endforeach()

    target_include_directories(deepsight_capture PRIVATE
        ${LUCKFOX_SDK_ROOT}/include
        ${LUCKFOX_SDK_ROOT}/include/rkaiq
        ${_RKAIQ_INCDIRS}
    )

    target_link_directories(deepsight_capture PUBLIC ${LUCKFOX_SDK_ROOT}/lib)
    target_link_libraries(deepsight_capture PUBLIC
        rockit          # RK MPI runtime
        rkaiq           # RKAIQ ISP library
        sample_comm     # ISP init/run/stop helpers (libsample_comm.a)
        pthread
    )
endif()

# ─── install rules ────────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS deepsight_capture
    LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/deepsight
)
