# Tests — no backend libraries required.
# DetectorFactory.cpp is compiled here WITHOUT WITH_ONNXRT / WITH_OPENVINO /
# WITH_RKNN so the linker needs nothing except yaml-cpp and the C++ stdlib.

add_executable(test_extension_check
    test_extension_check.cpp
    ../src/inference/DetectorFactory.cpp
    ../src/debug_log.cpp
)

target_include_directories(test_extension_check PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_extension_check PRIVATE
    yaml-cpp::yaml-cpp
)

add_test(NAME ExtensionCheck COMMAND test_extension_check)
