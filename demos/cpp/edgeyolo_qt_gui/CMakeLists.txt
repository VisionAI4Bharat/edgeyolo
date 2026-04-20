cmake_minimum_required(VERSION 3.16)
project(EdgeYOLOQtGui LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── build options ────────────────────────────────────────────────────────────
option(USE_QT5        "Use Qt5 instead of Qt6"              OFF)
option(BUILD_DEMO     "Build the demo application"          ON)
option(WITH_OPENVINO  "Enable OpenVINO inference backend"   ON)
option(WITH_RKNN      "Enable RKNN inference backend"       OFF)

include(FetchContent)

# ─── Qt ───────────────────────────────────────────────────────────────────────
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

# ─── opencv-mobile ────────────────────────────────────────────────────────────
# Prebuilt packages from https://github.com/nihui/opencv-mobile/releases.
# Desktop (x86_64 Linux): pass -DOpenCV_DIR=/usr/lib/... for system OpenCV,
# or let CMake detect automatically.  For arm/aarch64 targets the prebuilt
# package is downloaded automatically.
#
# If OpenCV is already found via an explicit -DOpenCV_DIR or a previous
# find_package(), skip the FetchContent path entirely.
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET CONFIG)
endif()

if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    # Determine platform tag used in opencv-mobile release assets
    set(OCVM_VERSION "4.10.0" CACHE STRING "opencv-mobile version")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(_OCVM_PLATFORM "aarch64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        set(_OCVM_PLATFORM "arm-linux-gnueabihf")
    else()
        # x86_64 — opencv-mobile has no prebuilt Linux binary for x64;
        # the user must install OpenCV (apt install libopencv-dev).
        message(FATAL_ERROR
            "OpenCV not found and opencv-mobile has no x86_64 Linux prebuilt.\n"
            "Install it with:  sudo apt install libopencv-dev\n"
            "or set -DOpenCV_DIR=/path/to/opencv4/cmake.")
    endif()

    set(_OCVM_CACHE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/opencv-mobile-${OCVM_VERSION}-${_OCVM_PLATFORM}")
    set(_OCVM_URL
        "https://github.com/nihui/opencv-mobile/releases/download/v${OCVM_VERSION}/opencv-mobile-${OCVM_VERSION}-${_OCVM_PLATFORM}.zip")

    if(NOT EXISTS "${_OCVM_CACHE_DIR}/lib/cmake/opencv4/OpenCVConfig.cmake")
        message(STATUS "Fetching opencv-mobile ${OCVM_VERSION} (${_OCVM_PLATFORM})…")
        FetchContent_Declare(ocvm
            URL        "${_OCVM_URL}"
            SOURCE_DIR "${_OCVM_CACHE_DIR}"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_Populate(ocvm)
    else()
        message(STATUS "opencv-mobile cached: ${_OCVM_CACHE_DIR}")
    endif()

    set(OpenCV_DIR "${_OCVM_CACHE_DIR}/lib/cmake/opencv4")
    find_package(OpenCV REQUIRED CONFIG)
    message(STATUS "opencv-mobile: ${OpenCV_VERSION}")
endif()

# ─── yaml-cpp (always via FetchContent) ──────────────────────────────────────
set(YAML_CPP_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL       OFF CACHE BOOL "" FORCE)
FetchContent_Declare(yaml-cpp
    URL      https://github.com/jbeder/yaml-cpp/releases/download/yaml-cpp-0.9.0/yaml-cpp-yaml-cpp-0.9.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(yaml-cpp)
message(STATUS "yaml-cpp: built from source via FetchContent")

# ─── ONNX Runtime (always required) ──────────────────────────────────────────
# Downloaded from GitHub prebuilt releases and cached in thirdparty/.
# Pass -DORT_VERSION=x.y.z to override.
# For cross-compile: set -DORT_PLATFORM accordingly or supply -DONNXRUNTIME_ROOT.
set(ORT_VERSION  "1.18.1"          CACHE STRING "ONNX Runtime version to fetch")
set(ORT_PLATFORM "linux-x64"       CACHE STRING "ONNX Runtime platform string (linux-x64 / linux-aarch64)")
set(ORT_CACHE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}")

if(DEFINED ONNXRUNTIME_ROOT)
    # Explicit override (e.g. cross-compile sysroot)
    set(ORT_CACHE_DIR          "${ONNXRUNTIME_ROOT}")
endif()

if(NOT EXISTS "${ORT_CACHE_DIR}/include/onnxruntime_cxx_api.h")
    set(_ORT_URL
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.tgz")
    message(STATUS "Fetching ONNX Runtime ${ORT_VERSION} (${ORT_PLATFORM}) from GitHub…")
    FetchContent_Declare(ort_pkg
        URL        "${_ORT_URL}"
        SOURCE_DIR "${ORT_CACHE_DIR}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_Populate(ort_pkg)
else()
    message(STATUS "ONNX Runtime found in cache: ${ORT_CACHE_DIR}")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ORT_CACHE_DIR}/include")
set(ONNXRUNTIME_LIB         "${ORT_CACHE_DIR}/lib/libonnxruntime.so")

if(NOT EXISTS "${ONNXRUNTIME_LIB}")
    message(FATAL_ERROR "ONNX Runtime library not found after fetch: ${ONNXRUNTIME_LIB}")
endif()
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib:     ${ONNXRUNTIME_LIB}")

# ─── OpenVINO (optional) ─────────────────────────────────────────────────────
if(WITH_OPENVINO)
    include(FetchContent)

    if(WITH_OPENVINO)
        include(FetchContent)

        # Determine the archive name based on your system (x86_64 for Ubuntu)
        set(OV_VER "2026.1.0")
        set(OV_PLATFORM "ubuntu24_x86_64") # Adjust to ubuntu22 if needed
        set(OV_URL "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.1/linux/openvino_toolkit_ubuntu22_2026.1.0.21367.63e31528c62_x86_64.tgz")

        message(STATUS "[OV] Fetching Pre-built OpenVINO ${OV_VER}...")

        FetchContent_Declare(
          openvino
          URL ${OV_URL}
          # This skips the entire build process and just uses the headers/libs
        )

        FetchContent_MakeAvailable(openvino)

        # Manual path setup because pre-built archives don't always
        # export the openvino::runtime target immediately in FetchContent
        set(OpenVINO_DIR "${openvino_SOURCE_DIR}/runtime/cmake")
        find_package(OpenVINO REQUIRED)

        message(STATUS "[OV] Using Pre-built from: ${openvino_SOURCE_DIR}")
    endif()
    # 4. Use the 'openvino_POPULATED' variable to check status
    FetchContent_GetProperties(openvino)
    if(openvino_POPULATED)
        message(STATUS "[DIAGNOSTIC] OpenVINO Source Dir: ${openvino_SOURCE_DIR}")
        message(STATUS "[DIAGNOSTIC] OpenVINO Binary Dir: ${openvino_BINARY_DIR}")
    endif()

    # 5. Query the target directly
    if(TARGET openvino::runtime)
        # We use 'LOCATION' and 'INTERFACE_INCLUDE_DIRECTORIES'
        # Note: Some properties are only available after the Generate grade,
        # but INTERFACE_INCLUDE_DIRECTORIES should be readable now.
        get_target_property(TEMP_OV_INC openvino::runtime INTERFACE_INCLUDE_DIRECTORIES)

        message(STATUS "-------------------------------------------------------")
        message(STATUS " [SUCCESS] OpenVINO Target Found")

        # If the list is empty, CMake might be using Generator Expressions
        if("${TEMP_OV_INC}" STREQUAL "TEMP_OV_INC-NOTFOUND" OR "${TEMP_OV_INC}" STREQUAL "")
             message(STATUS " [OV] Include Dirs: (Internal/Generator Expression)")
        else()
             message(STATUS " [OV] Include Dirs: ${TEMP_OV_INC}")
        endif()
        message(STATUS "-------------------------------------------------------")
    else()
        message(FATAL_ERROR " [ERROR] OpenVINO target NOT created. Check network or GIT_TAG.")
    endif()
endif()

# ─── RKNN (optional, cross-compile for RV1106) ───────────────────────────────
if(WITH_RKNN)
    if(NOT DEFINED RKNN_ROOT)
        message(FATAL_ERROR
            "WITH_RKNN requires -DRKNN_ROOT=/path/to/rknn-toolkit-lite2")
    endif()
    set(RKNN_INCLUDE_DIR "${RKNN_ROOT}/include")
    set(RKNN_LIB         "${RKNN_ROOT}/lib/librknnrt.so")
    if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
        message(FATAL_ERROR "rknn_api.h not found in ${RKNN_INCLUDE_DIR}")
    endif()
    message(STATUS "RKNN include: ${RKNN_INCLUDE_DIR}")
    message(STATUS "RKNN lib:     ${RKNN_LIB}")
endif()

# ─── sources ─────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/debug_log.cpp
    src/ui/mainwindow.cpp
    src/ui/videowidget.cpp
    src/ui/configdialog.cpp
    src/ui/classlabelsdialog.cpp
    src/inference/DetectorFactory.cpp
    src/inference/OnnxDetector.cpp
    src/inference/OpenVinoDetector.cpp
    src/inference/RknnDetector.cpp
    src/inference/edgeyolo_bridge.cpp
)

set(HEADERS
    src/ui/mainwindow.h
    src/ui/videowidget.h
    src/ui/configdialog.h
    src/ui/classlabelsdialog.h
    src/inference/IDetector.h
    src/inference/DetectorFactory.h
    src/inference/OnnxDetector.h
    src/inference/OpenVinoDetector.h
    src/inference/RknnDetector.h
    src/inference/edgeyolo_bridge.h
)

# ─── executable ──────────────────────────────────────────────────────────────
if(BUILD_DEMO)
    set(RESOURCES assets/assets.qrc)
    set_source_files_properties(${RESOURCES} PROPERTIES AUTORCC ON)

    add_executable(edgeyolo_qt_gui ${SOURCES}
        src/inference/edgeyolo_bridge.cpp
        src/inference/edgeyolo_bridge.h
        ${RESOURCES})

    # Qt auto-moc + runtime search path for the fetched ONNX Runtime shared library
    set_target_properties(edgeyolo_qt_gui PROPERTIES
        AUTOMOC ON
        AUTORCC ON
        BUILD_RPATH           "${ORT_CACHE_DIR}/lib"
        INSTALL_RPATH         "${ORT_CACHE_DIR}/lib"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )

    # ── include paths ─────────────────────────────────────────────────────
    target_include_directories(edgeyolo_qt_gui PRIVATE
        ${PROJECT_SOURCE_DIR}/src          # root for inference/ includes
        ${PROJECT_SOURCE_DIR}/src/ui       # ui headers visible to each other
        ${ONNXRUNTIME_INCLUDE_DIR}
        # EdgeYOLO common utilities (letterbox, NMS, Object struct)
        ${PROJECT_SOURCE_DIR}/../../third_party/edgeyolo/deployment/yolo
    )

    if(WITH_RKNN)
        target_include_directories(edgeyolo_qt_gui PRIVATE
            ${RKNN_INCLUDE_DIR}
            ${PROJECT_SOURCE_DIR}/../../third_party/edgeyolo/cpp/rknn/include
        )
    endif()


    # ── compile definitions ───────────────────────────────────────────────
    target_compile_definitions(edgeyolo_qt_gui PRIVATE
        QT_VERSION=${QT_VERSION}
    )
    if(WITH_OPENVINO)
        target_compile_definitions(edgeyolo_qt_gui PRIVATE WITH_OPENVINO)
    endif()
    if(WITH_RKNN)
        target_compile_definitions(edgeyolo_qt_gui PRIVATE WITH_RKNN)
    endif()

    # ── link libraries ────────────────────────────────────────────────────
    target_link_libraries(edgeyolo_qt_gui
        PRIVATE
            ${QT_CORE_LIB}
            ${QT_GUI_LIB}
            ${QT_WIDGETS_LIB}
            ${QT_CONCURRENT_LIB}
            ${OpenCV_LIBS}
            yaml-cpp::yaml-cpp
            ${ONNXRUNTIME_LIB}
    )

    if(WITH_OPENVINO)
        target_link_libraries(edgeyolo_qt_gui PRIVATE openvino::runtime)
    endif()

    if(WITH_RKNN)
        target_link_libraries(edgeyolo_qt_gui PRIVATE ${RKNN_LIB})
    endif()

    install(TARGETS edgeyolo_qt_gui DESTINATION bin)
endif()

# ─── thirdparty cache dir ─────────────────────────────────────────────────────
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/thirdparty")
    file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/thirdparty")
endif()
