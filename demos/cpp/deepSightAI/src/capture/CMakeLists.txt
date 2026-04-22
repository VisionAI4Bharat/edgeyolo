# Copyright (C) 2026 swatah.ai. All rights reserved.

find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(OpenCV QUIET COMPONENTS videoio)

set(CAPTURE_SOURCES 
    CaptureFactory.cpp 
    GenericCapture.cpp
)

if(WITH_RKNN)
    list(APPEND CAPTURE_SOURCES rv1106_capture.cpp)
endif()

add_library(deepSightAI_capture SHARED ${CAPTURE_SOURCES})

target_include_directories(deepSightAI_capture PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

if(OpenCV_videoio_FOUND)
    target_compile_definitions(deepSightAI_capture PRIVATE HAVE_OPENCV_VIDEOIO)
endif()

if(WITH_RKNN)
    target_compile_definitions(deepSightAI_capture PRIVATE WITH_RKNN)
    target_include_directories(deepSightAI_capture PRIVATE
        ${LUCKFOX_SDK_ROOT}/include
        ${LUCKFOX_SDK_ROOT}/include/rkaiq
        ${LUCKFOX_SDK_ROOT}/include/rkaiq/uAPI2
        ${LUCKFOX_SDK_ROOT}/include/rkaiq/common
        ${LUCKFOX_SDK_ROOT}/include/rkaiq/xcore
        ${LUCKFOX_SDK_ROOT}/include/rkaiq/iq_parser
        ${LUCKFOX_SDK_ROOT}/include/rkaiq/iq_parser_v2
        ${LUCKFOX_SDK_ROOT}/include/rkaiq/algos
    )
    target_link_directories(deepSightAI_capture PUBLIC ${LUCKFOX_SDK_ROOT}/lib)
    target_link_libraries(deepSightAI_capture PUBLIC rtsp sample_comm rockit rockchip_mpp rkaiq)
endif()

target_link_libraries(deepSightAI_capture PUBLIC ${OpenCV_LIBS} pthread)

include(GNUInstallDirs)
install(TARGETS deepSightAI_capture
    LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/deepSightAI
)
