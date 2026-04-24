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

# ─── test_config_log ──────────────────────────────────────────────────────────
add_executable(test_config_log
    test_config_log.cpp
    ../src/app_config.cpp
    ../src/debug_log.cpp
)

target_include_directories(test_config_log PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_config_log PRIVATE
    yaml-cpp::yaml-cpp
)

add_test(NAME ConfigLog COMMAND test_config_log)

# Run all tests automatically as part of every build
add_custom_target(run_tests ALL
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
    DEPENDS test_extension_check test_config_log
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tests..."
)
