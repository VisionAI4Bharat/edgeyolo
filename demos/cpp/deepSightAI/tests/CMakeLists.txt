# Tests — no backend libraries required.
# DetectorFactory.cpp is compiled here WITHOUT WITH_ONNXRT / WITH_OPENVINO /
# WITH_RKNN so the linker needs nothing except yaml-cpp and the C++ stdlib.

# Link stdc++fs if using GCC < 9 (required for std::filesystem)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    set(FS_LIB stdc++fs)
else()
    set(FS_LIB "")
endif()

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
    ${FS_LIB}
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
    ${FS_LIB}
)

add_test(NAME ConfigLog COMMAND test_config_log)

# Run all tests automatically as part of every build
# ─── test_comment_preservation ───────────────────────────────────────────────
add_executable(test_comment_preservation
    test_comment_preservation.cpp
    ../src/app_config.cpp
    ../src/debug_log.cpp
)

target_include_directories(test_comment_preservation PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_comment_preservation PRIVATE
    yaml-cpp::yaml-cpp
    ${FS_LIB}
)

add_test(NAME CommentPreservation COMMAND test_comment_preservation)

add_custom_target(run_tests ALL
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
    DEPENDS test_extension_check test_config_log test_comment_preservation
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tests..."
)
