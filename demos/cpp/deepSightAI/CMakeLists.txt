# Copyright (C) 2026 swatah.ai. All rights reserved.
#
# Dual-licensed: GPLv3 (non-commercial) / proprietary (commercial).
# See LICENSE for details.

cmake_minimum_required(VERSION 3.16)
project(deepSightAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── build options ────────────────────────────────────────────────────────────
option(USE_QT5           "Use Qt5 instead of Qt6"                  OFF)
option(BUILD_QT_GUI      "Build the Qt desktop GUI application"    OFF)
option(BUILD_HEADLESS    "Build the headless embedded application"  OFF)
option(WITH_OPENVINO     "Enable OpenVINO inference backend"        ON)
option(WITH_RKNN         "Enable RKNN inference backend"           OFF)
option(WITH_ONNXRT       "Enable ONNX Runtime inference backend"   ON)

# WITH_RKNN implies headless (no Qt on embedded target) and disables x86 backends.
# BUILD_HEADLESS can also be set manually for server deployments with any backend.
if(WITH_RKNN)
    set(BUILD_HEADLESS ON  CACHE BOOL "" FORCE)
    set(BUILD_QT_GUI   OFF CACHE BOOL "" FORCE)
    if(WITH_ONNXRT)
        message(STATUS "WITH_RKNN=ON: disabling ONNX Runtime (no ARM uclibc prebuilt)")
        set(WITH_ONNXRT OFF CACHE BOOL "" FORCE)
    endif()
    if(WITH_OPENVINO)
        message(STATUS "WITH_RKNN=ON: disabling OpenVINO (x86_64 only)")
        set(WITH_OPENVINO OFF CACHE BOOL "" FORCE)
    endif()
endif()

# If Qt GUI is explicitly disabled (but RKNN not selected), auto-enable headless.
if(NOT BUILD_QT_GUI AND NOT BUILD_HEADLESS)
    message(STATUS "BUILD_QT_GUI=OFF: enabling BUILD_HEADLESS automatically")
    set(BUILD_HEADLESS ON CACHE BOOL "" FORCE)
endif()

include(FetchContent)

# ─── Qt (desktop GUI only) ────────────────────────────────────────────────────
if(BUILD_QT_GUI)
    if(USE_QT5)
        message(STATUS "Using Qt5")
        find_package(Qt5 COMPONENTS Core Gui Widgets Concurrent REQUIRED)
        set(QT_VERSION 5)
        set(QT_CORE_LIB Qt5::Core)
        set(QT_GUI_LIB Qt5::Gui)
        set(QT_WIDGETS_LIB Qt5::Widgets)
        set(QT_CONCURRENT_LIB Qt5::Concurrent)
    else()
        message(STATUS "Using Qt6")
        find_package(Qt6 COMPONENTS Core Gui Widgets Concurrent REQUIRED)
        set(QT_VERSION 6)
        set(QT_CORE_LIB Qt6::Core)
        set(QT_GUI_LIB Qt6::Gui)
        set(QT_WIDGETS_LIB Qt6::Widgets)
        set(QT_CONCURRENT_LIB Qt6::Concurrent)
    endif()
endif()

# ─── OpenCV ───────────────────────────────────────────────────────────────────
# Desktop: system or -DOpenCV_DIR=...
# ARM cross-compile: use luckfox_mpi bundled opencv (set via -DOpenCV_DIR=...)
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET CONFIG)
endif()

if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    set(OCVM_VERSION "4.10.0" CACHE STRING "opencv-mobile version")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(_OCVM_PLATFORM "aarch64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        set(_OCVM_PLATFORM "arm-linux-gnueabihf")
    else()
        message(FATAL_ERROR
            "OpenCV not found.\n"
            "Install:  sudo apt install libopencv-dev\n"
            "or set -DOpenCV_DIR=/path/to/opencv4/cmake.")
    endif()
    set(_OCVM_CACHE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/opencv-mobile-${OCVM_VERSION}-${_OCVM_PLATFORM}")
    set(_OCVM_URL "https://github.com/nihui/opencv-mobile/releases/download/v${OCVM_VERSION}/opencv-mobile-${OCVM_VERSION}-${_OCVM_PLATFORM}.zip")
    if(NOT EXISTS "${_OCVM_CACHE_DIR}/lib/cmake/opencv4/OpenCVConfig.cmake")
        message(STATUS "Fetching opencv-mobile ${OCVM_VERSION} (${_OCVM_PLATFORM})…")
        FetchContent_Declare(ocvm URL "${_OCVM_URL}" SOURCE_DIR "${_OCVM_CACHE_DIR}" DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_Populate(ocvm)
    else()
        message(STATUS "opencv-mobile cached: ${_OCVM_CACHE_DIR}")
    endif()
    set(OpenCV_DIR "${_OCVM_CACHE_DIR}/lib/cmake/opencv4")
    find_package(OpenCV REQUIRED CONFIG)
    message(STATUS "opencv-mobile: ${OpenCV_VERSION}")
endif()

# ─── yaml-cpp ─────────────────────────────────────────────────────────────────
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL     OFF CACHE BOOL "" FORCE)
FetchContent_Declare(yaml-cpp
    URL      https://github.com/jbeder/yaml-cpp/releases/download/yaml-cpp-0.9.0/yaml-cpp-yaml-cpp-0.9.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(yaml-cpp)

# ─── ONNX Runtime (x86_64 desktop only) ──────────────────────────────────────
if(WITH_ONNXRT)
    set(ORT_VERSION   "1.18.1"    CACHE STRING "ONNX Runtime version")
    set(ORT_PLATFORM  "linux-x64" CACHE STRING "ONNX Runtime platform")
    set(ORT_CACHE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}")
    if(DEFINED ONNXRUNTIME_ROOT)
        set(ORT_CACHE_DIR "${ONNXRUNTIME_ROOT}")
    endif()
    if(NOT EXISTS "${ORT_CACHE_DIR}/include/onnxruntime_cxx_api.h")
        set(_ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.tgz")
        message(STATUS "Fetching ONNX Runtime ${ORT_VERSION}…")
        FetchContent_Declare(ort_pkg URL "${_ORT_URL}" SOURCE_DIR "${ORT_CACHE_DIR}" DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_Populate(ort_pkg)
    else()
        message(STATUS "ONNX Runtime cached: ${ORT_CACHE_DIR}")
    endif()
    set(ONNXRUNTIME_INCLUDE_DIR "${ORT_CACHE_DIR}/include")
    set(ONNXRUNTIME_LIB         "${ORT_CACHE_DIR}/lib/libonnxruntime.so")
    if(NOT EXISTS "${ONNXRUNTIME_LIB}")
        message(FATAL_ERROR "ONNX Runtime library not found: ${ONNXRUNTIME_LIB}")
    endif()
    message(STATUS "ONNX Runtime: ${ONNXRUNTIME_INCLUDE_DIR}")
else()
    set(ONNXRUNTIME_INCLUDE_DIR "")
    set(ONNXRUNTIME_LIB         "")
    message(STATUS "ONNX Runtime: disabled")
endif()

# ─── OpenVINO (x86_64 desktop only) ──────────────────────────────────────────
if(WITH_OPENVINO)
    set(OV_URL "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.1/linux/openvino_toolkit_ubuntu22_2026.1.0.21367.63e31528c62_x86_64.tgz")
    message(STATUS "[OV] Fetching OpenVINO 2026.1…")
    FetchContent_Declare(openvino URL ${OV_URL})
    FetchContent_MakeAvailable(openvino)
    set(OpenVINO_DIR "${openvino_SOURCE_DIR}/runtime/cmake")
    find_package(OpenVINO REQUIRED)
    message(STATUS "[OV] ${openvino_SOURCE_DIR}")
endif()

# ─── RKNN toolkit (ARM cross-compile — RV110X: RV1103/RV1106) ─────────────────
if(WITH_RKNN)
    if(NOT DEFINED RKNN_ROOT)
        set(RKNN_ROOT "${PROJECT_SOURCE_DIR}/third_party/rockchip/RV110X/rknn" CACHE PATH
            "RKNN toolkit root (include/rknn_api.h, lib/librknnmrt.so)")
    endif()
    set(RKNN_INCLUDE_DIR "${RKNN_ROOT}/include")
    set(RKNN_LIB         "${RKNN_ROOT}/lib/librknnmrt.so")
    if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
        message(FATAL_ERROR "rknn_api.h not found in ${RKNN_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${RKNN_LIB}")
        message(FATAL_ERROR "librknnmrt.so not found: ${RKNN_LIB}")
    endif()
    message(STATUS "RKNN include: ${RKNN_INCLUDE_DIR}")
    message(STATUS "RKNN lib:     ${RKNN_LIB}")
endif()

# ─── Rockchip MPI/ISP SDK (RV110X: arm32 uclibc) ─────────────────────────────
if(WITH_RKNN AND NOT LUCKFOX_SDK_ROOT)
    set(LUCKFOX_SDK_ROOT "${PROJECT_SOURCE_DIR}/third_party/rockchip/RV110X/mpi" CACHE PATH
        "Rockchip MPI SDK root for RV110X (include/ + lib/)")
else()
    set(LUCKFOX_SDK_ROOT "" CACHE PATH "Rockchip MPI SDK root (include/ + lib/)")
endif()

# ─── deepsight_capture shared library (no Qt, always built) ──────────────────
add_subdirectory(src/capture)

# ─── shared inference sources ─────────────────────────────────────────────────
set(INFERENCE_SOURCES
    src/inference/DetectorFactory.cpp
    src/inference/edgeyolo_bridge.cpp
)
if(WITH_ONNXRT)
    list(APPEND INFERENCE_SOURCES src/inference/OnnxDetector.cpp)
endif()
if(WITH_OPENVINO)
    list(APPEND INFERENCE_SOURCES src/inference/OpenVinoDetector.cpp)
endif()
if(WITH_RKNN)
    list(APPEND INFERENCE_SOURCES src/inference/RknnDetector.cpp)
endif()

# ─── helper: apply RKNN/stdc++fs link flags to a target ──────────────────────
function(target_apply_rknn_flags TARGET_NAME)
    if(WITH_RKNN)
        target_include_directories(${TARGET_NAME} PRIVATE ${RKNN_INCLUDE_DIR})
        target_link_directories(${TARGET_NAME} PRIVATE
            ${LUCKFOX_SDK_ROOT}/lib
            ${PROJECT_SOURCE_DIR}/third_party/rknn/lib)
        target_compile_definitions(${TARGET_NAME} PRIVATE WITH_RKNN)
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
            target_link_libraries(${TARGET_NAME} PRIVATE stdc++fs)
        endif()
        target_link_libraries(${TARGET_NAME} PRIVATE ${RKNN_LIB} rockit rkaiq sample_comm pthread)
    endif()
endfunction()

# ─── Qt GUI (desktop) ─────────────────────────────────────────────────────────
if(BUILD_QT_GUI)
    set(GUI_SOURCES
        src/main.cpp
        src/debug_log.cpp
        src/app_config.cpp
        src/ui/mainwindow.cpp
        src/ui/videowidget.cpp
        src/ui/configdialog.cpp
        src/ui/classlabelsdialog.cpp
        ${INFERENCE_SOURCES}
    )

    set(RESOURCES assets/assets.qrc)
    add_executable(deepSightAI-desktop ${GUI_SOURCES} ${RESOURCES})

    set_target_properties(deepSightAI-desktop PROPERTIES
        AUTOMOC ON
        AUTORCC ON
        BUILD_RPATH           "${ORT_CACHE_DIR}/lib"
        INSTALL_RPATH         "${ORT_CACHE_DIR}/lib"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )

    target_include_directories(deepSightAI-desktop PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/ui
        ${PROJECT_SOURCE_DIR}/src/capture
        ${PROJECT_SOURCE_DIR}/../../third_party/edgeyolo/deployment/yolo
    )
    if(WITH_ONNXRT)
        target_include_directories(deepSightAI-desktop PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    endif()

    target_compile_definitions(deepSightAI-desktop PRIVATE QT_VERSION=${QT_VERSION})
    if(WITH_ONNXRT)
        target_compile_definitions(deepSightAI-desktop PRIVATE WITH_ONNXRT)
    endif()
    if(WITH_OPENVINO)
        target_compile_definitions(deepSightAI-desktop PRIVATE WITH_OPENVINO)
    endif()

    target_link_libraries(deepSightAI-desktop PRIVATE
        ${QT_CORE_LIB} ${QT_GUI_LIB} ${QT_WIDGETS_LIB} ${QT_CONCURRENT_LIB}
        ${OpenCV_LIBS}
        yaml-cpp::yaml-cpp
        deepsight_capture
    )
    if(WITH_ONNXRT)
        target_link_libraries(deepSightAI-desktop PRIVATE ${ONNXRUNTIME_LIB})
    endif()
    if(WITH_OPENVINO)
        target_link_libraries(deepSightAI-desktop PRIVATE openvino::runtime)
    endif()

    install(TARGETS deepSightAI-desktop DESTINATION bin)
    message(STATUS "Qt GUI target: deepSightAI-desktop")
endif()

# ─── Headless binary (no Qt — works with RKNN, OpenVINO, or ONNX Runtime) ─────
if(BUILD_HEADLESS)
    set(HEADLESS_SOURCES
        src/main_headless.cpp
        src/debug_log.cpp
        src/app_config.cpp
        src/headless/headless_app.cpp
        src/headless/web_config.cpp
        ${INFERENCE_SOURCES}
    )

    add_executable(deepSightAI-headless ${HEADLESS_SOURCES})

    # RPATH: include every backend lib dir that is active
    set(_HEADLESS_RPATH "")
    if(WITH_RKNN)
        list(APPEND _HEADLESS_RPATH "${LUCKFOX_SDK_ROOT}/lib")
        list(APPEND _HEADLESS_RPATH "${PROJECT_SOURCE_DIR}/third_party/rknn/lib")
    endif()
    if(WITH_ONNXRT)
        list(APPEND _HEADLESS_RPATH "${ORT_CACHE_DIR}/lib")
    endif()
    string(JOIN ":" _HEADLESS_RPATH_STR ${_HEADLESS_RPATH})

    set_target_properties(deepSightAI-headless PROPERTIES
        BUILD_RPATH           "${_HEADLESS_RPATH_STR}"
        INSTALL_RPATH         "${_HEADLESS_RPATH_STR}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )

    target_include_directories(deepSightAI-headless PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/capture
        ${PROJECT_SOURCE_DIR}/src/headless
        ${PROJECT_SOURCE_DIR}/../../third_party/edgeyolo/deployment/yolo
    )
    if(WITH_ONNXRT)
        target_include_directories(deepSightAI-headless PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    endif()

    # Compile definitions — must match whatever INFERENCE_SOURCES uses
    if(WITH_ONNXRT)
        target_compile_definitions(deepSightAI-headless PRIVATE WITH_ONNXRT)
    endif()
    if(WITH_OPENVINO)
        target_compile_definitions(deepSightAI-headless PRIVATE WITH_OPENVINO)
    endif()

    # RKNN + stdc++fs (ARM GCC < 9)
    target_apply_rknn_flags(deepSightAI-headless)

    target_link_libraries(deepSightAI-headless PRIVATE
        ${OpenCV_LIBS}
        yaml-cpp::yaml-cpp
        deepsight_capture
    )
    if(WITH_ONNXRT)
        target_link_libraries(deepSightAI-headless PRIVATE ${ONNXRUNTIME_LIB})
    endif()
    if(WITH_OPENVINO)
        target_link_libraries(deepSightAI-headless PRIVATE openvino::runtime)
    endif()

    install(TARGETS deepSightAI-headless DESTINATION bin)
    message(STATUS "Headless target: deepSightAI-headless  [RKNN=${WITH_RKNN} OV=${WITH_OPENVINO} ORT=${WITH_ONNXRT}]")
endif()

# ─── thirdparty cache dir ─────────────────────────────────────────────────────
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/thirdparty")
    file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/thirdparty")
endif()
