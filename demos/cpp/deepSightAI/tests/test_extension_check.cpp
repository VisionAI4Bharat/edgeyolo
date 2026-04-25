/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 *
 * Unit tests for DetectorFactory validation helpers:
 *   - dsai_validateModelExtension
 *   - dsai_validateThresholds
 * No backend libraries (ONNX / OpenVINO / RKNN) are needed — the test
 * links only against DetectorFactory.cpp compiled without any backend defines.
 */

#include "inference/DetectorFactory.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

using inference::Backend;
using inference::DetectorFactory;

// ─── minimal test harness ─────────────────────────────────────────────────────
static int s_passed = 0;
static int s_failed = 0;

#define EXPECT_NO_THROW(expr, label)                                    \
    do {                                                                 \
        try { (expr); printf("[PASS] %s\n", label); ++s_passed; }       \
        catch (const std::exception& e) {                               \
            printf("[FAIL] %s  -- unexpected exception: %s\n",          \
                   label, e.what()); ++s_failed;                        \
        }                                                                \
    } while (0)

#define EXPECT_THROW_CONTAINING(expr, fragment, label)                  \
    do {                                                                 \
        try {                                                            \
            (expr);                                                      \
            printf("[FAIL] %s  -- expected exception not thrown\n",     \
                   label); ++s_failed;                                   \
        } catch (const std::runtime_error& e) {                         \
            if (std::string(e.what()).find(fragment) != std::string::npos) { \
                printf("[PASS] %s\n", label); ++s_passed;               \
            } else {                                                     \
                printf("[FAIL] %s  -- exception missing '%s': %s\n",    \
                       label, fragment, e.what()); ++s_failed;          \
            }                                                            \
        }                                                                \
    } while (0)

// ─── tests ────────────────────────────────────────────────────────────────────

static void test_onnx() {
    EXPECT_NO_THROW(
        DetectorFactory::dsai_validateModelExtension(Backend::ONNX, "model.onnx"),
        "ONNX: .onnx accepted");

    EXPECT_NO_THROW(
        DetectorFactory::dsai_validateModelExtension(Backend::ONNX, "MODEL.ONNX"),
        "ONNX: .ONNX (uppercase) accepted");

    EXPECT_NO_THROW(
        DetectorFactory::dsai_validateModelExtension(Backend::ONNX, "/path/to/edgeyolo_tiny.onnx"),
        "ONNX: absolute path accepted");

    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateModelExtension(Backend::ONNX, "model.xml"),
        ".xml", "ONNX: .xml rejected with extension in message");

    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateModelExtension(Backend::ONNX, "model.rknn"),
        "ONNX Runtime", "ONNX: .rknn rejected with backend name in message");

    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateModelExtension(Backend::ONNX, "model"),
        ".onnx", "ONNX: no extension rejected");
}

static void test_openvino() {
    EXPECT_NO_THROW(
        DetectorFactory::dsai_validateModelExtension(Backend::OPENVINO, "model.xml"),
        "OpenVINO: .xml accepted");

    EXPECT_NO_THROW(
        DetectorFactory::dsai_validateModelExtension(Backend::OPENVINO, "model.bin"),
        "OpenVINO: .bin accepted");

    EXPECT_NO_THROW(
        DetectorFactory::dsai_validateModelExtension(Backend::OPENVINO, "MODEL.XML"),
        "OpenVINO: .XML (uppercase) accepted");

    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateModelExtension(Backend::OPENVINO, "model.onnx"),
        "OpenVINO", "OpenVINO: .onnx rejected with backend name in message");

    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateModelExtension(Backend::OPENVINO, "model.rknn"),
        ".xml", "OpenVINO: .rknn rejected with expected ext in message");

    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateModelExtension(Backend::OPENVINO, "model"),
        "OpenVINO", "OpenVINO: no extension rejected");
}

static void test_rknn() {
    EXPECT_NO_THROW(
        DetectorFactory::dsai_validateModelExtension(Backend::RKNN, "model.rknn"),
        "RKNN: .rknn accepted");

    EXPECT_NO_THROW(
        DetectorFactory::dsai_validateModelExtension(Backend::RKNN, "MODEL.RKNN"),
        "RKNN: .RKNN (uppercase) accepted");

    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateModelExtension(Backend::RKNN, "model.onnx"),
        "RKNN", "RKNN: .onnx rejected with backend name in message");

    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateModelExtension(Backend::RKNN, "model.xml"),
        ".rknn", "RKNN: .xml rejected with expected ext in message");

    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateModelExtension(Backend::RKNN, "model"),
        "RKNN", "RKNN: no extension rejected");
}

static void test_message_contains_path() {
    // The error message must include the offending path so the user knows what to fix.
    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateModelExtension(
            Backend::ONNX, "/models/yolo_int8.xml"),
        "/models/yolo_int8.xml",
        "Error message contains the offending file path");
}

static void test_thresholds_valid() {
    // Boundary values that must not throw
    EXPECT_NO_THROW(
        DetectorFactory::dsai_validateThresholds(0.01f, 0.01f),
        "thresholds: minimum valid (0.01, 0.01)");
    EXPECT_NO_THROW(
        DetectorFactory::dsai_validateThresholds(1.0f, 1.0f),
        "thresholds: maximum valid (1.0, 1.0)");
    EXPECT_NO_THROW(
        DetectorFactory::dsai_validateThresholds(0.5f, 0.35f),
        "thresholds: typical (0.5, 0.35)");
}

static void test_thresholds_invalid_conf() {
    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateThresholds(0.0f, 0.45f),
        "confThres",
        "conf=0.0 rejected with name in message");
    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateThresholds(-0.1f, 0.45f),
        "confThres",
        "conf=-0.1 rejected with name in message");
    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateThresholds(1.1f, 0.45f),
        "confThres",
        "conf=1.1 rejected with name in message");
    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateThresholds(0.0f, 0.45f),
        "0.0000",
        "conf=0.0 error message contains the bad value");
}

static void test_thresholds_invalid_nms() {
    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateThresholds(0.5f, 0.0f),
        "nmsThres",
        "nms=0.0 rejected with name in message");
    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateThresholds(0.5f, -0.5f),
        "nmsThres",
        "nms=-0.5 rejected with name in message");
    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateThresholds(0.5f, 1.5f),
        "nmsThres",
        "nms=1.5 rejected with name in message");
    EXPECT_THROW_CONTAINING(
        DetectorFactory::dsai_validateThresholds(0.5f, 1.5f),
        "1.5000",
        "nms=1.5 error message contains the bad value");
}

// ─── entry point ─────────────────────────────────────────────────────────────
int main() {
    printf("=== DetectorFactory extension validation tests ===\n\n");
    test_onnx();
    printf("\n");
    test_openvino();
    printf("\n");
    test_rknn();
    printf("\n");
    test_message_contains_path();
    printf("\n");
    test_thresholds_valid();
    printf("\n");
    test_thresholds_invalid_conf();
    printf("\n");
    test_thresholds_invalid_nms();
    printf("\n=== Results: %d passed, %d failed ===\n", s_passed, s_failed);
    return s_failed == 0 ? 0 : 1;
}
