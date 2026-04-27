/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 *
 * Unit tests for AppConfig::dsai_logConfigToString.
 * Tests call dsai_logConfigToString() directly — no stdout capture needed.
 */

#include "app_config.h"
#include <cstdio>
#include <string>

// ─── minimal test harness (shared style with test_extension_check) ────────────
static int s_passed = 0;
static int s_failed = 0;

#define EXPECT_CONTAINS(str, fragment, label)                               \
    do {                                                                     \
        if ((str).find(fragment) != std::string::npos) {                    \
            printf("[PASS] %-45s  checked: '%s'\n", label, fragment);       \
            ++s_passed;                                                      \
        } else {                                                             \
            printf("[FAIL] %-45s  missing: '%s'\n  output was:\n%s\n",      \
                   label, fragment, (str).c_str()); ++s_failed;             \
        }                                                                    \
    } while (0)

#define EXPECT_NOT_CONTAINS(str, fragment, label)                           \
    do {                                                                     \
        if ((str).find(fragment) == std::string::npos) {                    \
            printf("[PASS] %-45s  absent:  '%s'\n", label, fragment);       \
            ++s_passed;                                                      \
        } else {                                                             \
            printf("[FAIL] %-45s  unexpectedly found: '%s'\n  output:\n%s\n",\
                   label, fragment, (str).c_str()); ++s_failed;             \
        }                                                                    \
    } while (0)

// Print the full config output once per test function for transparency
#define SHOW_OUTPUT(str) \
    printf("  output:\n    %s\n\n", (str).c_str())

// ─── tests ────────────────────────────────────────────────────────────────────

static void test_backend_names() {
    AppConfig cfg;

    cfg.backend = Backend::ONNX;
    SHOW_OUTPUT(cfg.dsai_logConfigToString());
    EXPECT_CONTAINS(cfg.dsai_logConfigToString(), "ONNX Runtime", "backend ONNX name");

    cfg.backend = Backend::OpenVINO;
    EXPECT_CONTAINS(cfg.dsai_logConfigToString(), "OpenVINO", "backend OpenVINO name");

    cfg.backend = Backend::RKNN;
    EXPECT_CONTAINS(cfg.dsai_logConfigToString(), "RKNN", "backend RKNN name");
}

static void test_model_paths() {
    AppConfig cfg;
    cfg.backend   = Backend::OpenVINO;
    cfg.modelFile = "/models/edgeyolo_tiny.xml";
    cfg.yamlFile  = "/models/edgeyolo_tiny.yaml";
    SHOW_OUTPUT(cfg.dsai_logConfigToString());

    std::string s = cfg.dsai_logConfigToString();
    EXPECT_CONTAINS(s, "/models/edgeyolo_tiny.xml",  "model file path present");
    EXPECT_CONTAINS(s, "/models/edgeyolo_tiny.yaml", "yaml file path present");
}

static void test_empty_paths_show_placeholders() {
    AppConfig cfg;
    cfg.modelFile = "";
    cfg.yamlFile  = "";

    std::string s = cfg.dsai_logConfigToString();
    EXPECT_CONTAINS(s, "(none)", "empty model shows (none)");
    EXPECT_CONTAINS(s, "(auto)", "empty yaml shows (auto)");
}

static void test_thresholds() {
    AppConfig cfg;
    cfg.confThreshold = 0.50f;
    cfg.nmsThreshold  = 0.35f;

    std::string s = cfg.dsai_logConfigToString();
    EXPECT_CONTAINS(s, "conf=0.50", "conf threshold formatted");
    EXPECT_CONTAINS(s, "nms=0.35",  "nms threshold formatted");
}

static void test_class_labels() {
    AppConfig cfg;
    cfg.classLabels = {"person", "forklift"};

    std::string s = cfg.dsai_logConfigToString();
    EXPECT_CONTAINS(s, "person",   "class label person present");
    EXPECT_CONTAINS(s, "forklift", "class label forklift present");
}

static void test_source_camera() {
    AppConfig cfg;
    cfg.source          = SourceType::Camera;
    cfg.cameraDeviceId  = 0;
    cfg.captureWidth    = 640;
    cfg.captureHeight   = 480;
    cfg.captureFps      = 30;

    std::string s = cfg.dsai_logConfigToString();
    SHOW_OUTPUT(s);
    EXPECT_CONTAINS(s, "Camera",  "source type Camera");
    EXPECT_CONTAINS(s, "device=0","camera device id");
    EXPECT_CONTAINS(s, "640x480", "camera resolution");
    EXPECT_CONTAINS(s, "30fps",   "camera fps");
}

static void test_source_videofile() {
    AppConfig cfg;
    cfg.source    = SourceType::VideoFile;
    cfg.videoFile = "/videos/test.mp4";

    std::string s = cfg.dsai_logConfigToString();
    EXPECT_CONTAINS(s, "VideoFile",       "source type VideoFile");
    EXPECT_CONTAINS(s, "/videos/test.mp4","video file path present");
}

static void test_source_rtsp() {
    AppConfig cfg;
    cfg.source  = SourceType::Rtsp;
    cfg.rtspUrl = "rtsp://192.168.1.1/live";

    std::string s = cfg.dsai_logConfigToString();
    EXPECT_CONTAINS(s, "RTSP",                   "source type RTSP");
    EXPECT_CONTAINS(s, "rtsp://192.168.1.1/live", "rtsp url present");
}

static void test_roi_disabled() {
    AppConfig cfg;
    cfg.roiEnabled = false;

    EXPECT_CONTAINS(cfg.dsai_logConfigToString(), "disabled", "ROI disabled label");
}

static void test_roi_enabled() {
    AppConfig cfg;
    cfg.roiEnabled  = true;
    cfg.roi = {10, 20, 300, 200};

    std::string s = cfg.dsai_logConfigToString();
    EXPECT_CONTAINS(s, "x=10",  "ROI x");
    EXPECT_CONTAINS(s, "y=20",  "ROI y");
    EXPECT_CONTAINS(s, "w=300", "ROI width");
    EXPECT_CONTAINS(s, "h=200", "ROI height");
    EXPECT_NOT_CONTAINS(s, "disabled", "ROI enabled — no 'disabled' label");
}

static void test_web_port() {
    AppConfig cfg;
    cfg.webPort = 9090;
    EXPECT_CONTAINS(cfg.dsai_logConfigToString(), "9090", "web port present");
}

static void test_section_markers() {
    AppConfig cfg;
    std::string s = cfg.dsai_logConfigToString();
    EXPECT_CONTAINS(s, "--- active configuration ---", "opening section marker");
    EXPECT_CONTAINS(s, "---",                          "closing section marker");
}

// ─── entry point ─────────────────────────────────────────────────────────────
int main() {
    printf("=== AppConfig startup log tests ===\n\n");
    test_backend_names();
    printf("\n");
    test_model_paths();
    test_empty_paths_show_placeholders();
    printf("\n");
    test_thresholds();
    test_class_labels();
    printf("\n");
    test_source_camera();
    test_source_videofile();
    test_source_rtsp();
    printf("\n");
    test_roi_disabled();
    test_roi_enabled();
    test_web_port();
    test_section_markers();
    printf("\n=== Results: %d passed, %d failed ===\n", s_passed, s_failed);
    return s_failed == 0 ? 0 : 1;
}
