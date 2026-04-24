/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 *
 * Verifies that AppConfig::dsai_saveToFile preserves every comment line
 * from the original YAML file.  No comment text is hardcoded — the test
 * generates the YAML, collects comment lines dynamically, then checks they
 * all survive a round-trip save.
 */

#include "app_config.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ─── minimal harness ─────────────────────────────────────────────────────────
static int s_passed = 0, s_failed = 0;

#define EXPECT_TRUE(cond, label) \
    do { \
        if (cond) { printf("[PASS] %-55s\n", label); ++s_passed; } \
        else      { printf("[FAIL] %-55s\n", label); ++s_failed; } \
    } while(0)

#define EXPECT_CONTAINS(str, frag, label) \
    do { \
        std::string _frag(frag); \
        if ((str).find(_frag) != std::string::npos) { \
            printf("[PASS] %-55s  found: '%s'\n", label, _frag.c_str()); ++s_passed; \
        } else { \
            printf("[FAIL] %-55s  missing: '%s'\n  file:\n%s\n", \
                   label, _frag.c_str(), (str).c_str()); ++s_failed; \
        } \
    } while(0)

// ─── helpers ─────────────────────────────────────────────────────────────────

// Collect all lines that begin with # (after optional whitespace).
static std::vector<std::string> commentLines(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        size_t s = line.find_first_not_of(' ');
        if (s != std::string::npos && line[s] == '#')
            out.push_back(line);
    }
    return out;
}

static std::string readFile(const std::string& path) {
    std::ifstream ifs(path);
    std::ostringstream ss; ss << ifs.rdbuf();
    return ss.str();
}

static void writeFile(const std::string& path, const std::string& content) {
    std::ofstream ofs(path);
    ofs << content;
}

// Build a full valid YAML string that has comments scattered throughout.
// Comments are generated from a caller-supplied prefix so they are unique
// and detectable without hardcoding specific text in the test logic.
static std::string buildYamlWithComments(const std::string& commentPrefix) {
    return
        "# " + commentPrefix + ": top-of-file header\n"
        "# " + commentPrefix + ": second header line\n"
        "#\n"
        "# " + commentPrefix + ": backend key description\n"
        "backend: 0\n"
        "model_file: \"/tmp/model.onnx\"\n"
        "yaml_file: \"\"\n"
        "# " + commentPrefix + ": threshold section\n"
        "conf_threshold: 0.25\n"
        "nms_threshold: 0.45\n"
        "class_labels:\n"
        "  - person\n"
        "hidden_class_ids:\n"
        "# " + commentPrefix + ": source section\n"
        "source: 0\n"
        "camera_device_id: 0\n"
        "video_file: \"\"\n"
        "rtsp_url: \"/live/0\"\n"
        "iq_dir: \"/etc/iqfiles\"\n"
        "resolution_index: 0\n"
        "fps_index: 2\n"
        "gain: 50\n"
        "gamma: 100\n"
        "brightness: 0\n"
        "rockchip_hw: false\n"
        "# " + commentPrefix + ": roi section\n"
        "roi_enabled: false\n"
        "roi:\n"
        "  x: 0\n"
        "  y: 0\n"
        "  width: 0\n"
        "  height: 0\n"
        "# " + commentPrefix + ": system section\n"
        "web_port: 8080\n"
        "debug_logging: false\n"
        "rtsp_port: 8554\n";
}

// ─── tests ───────────────────────────────────────────────────────────────────

static void test_all_comments_survive_roundtrip() {
    const std::string tmpPath = "/tmp/dsai_test_comments.yaml";
    const std::string prefix  = "ROUNDTRIP_TEST";

    std::string original = buildYamlWithComments(prefix);
    auto originalComments = commentLines(original);

    printf("  original has %zu comment lines\n", originalComments.size());
    EXPECT_TRUE(originalComments.size() >= 4,
                "test fixture has at least 4 comment lines");

    // Write the YAML, load it, change a value, save back
    writeFile(tmpPath, original);
    AppConfig cfg = AppConfig::dsai_loadFromFile(tmpPath);
    cfg.confThreshold = 0.75f;
    cfg.webPort       = 9090;
    cfg.dsai_saveToFile(tmpPath);

    std::string saved = readFile(tmpPath);
    auto savedComments = commentLines(saved);

    printf("  saved has %zu comment lines\n", savedComments.size());

    // Every comment from the original must still be present in the saved file
    int missing = 0;
    for (const auto& c : originalComments) {
        if (saved.find(c) == std::string::npos) {
            printf("  MISSING comment: %s\n", c.c_str());
            missing++;
        }
    }
    EXPECT_TRUE(missing == 0, "no comment lines were lost after save");
    EXPECT_TRUE(savedComments.size() == originalComments.size(),
                "comment count unchanged after save");
}

static void test_values_updated_comments_preserved() {
    const std::string tmpPath = "/tmp/dsai_test_values.yaml";
    const std::string prefix  = "VALUE_TEST";

    writeFile(tmpPath, buildYamlWithComments(prefix));
    AppConfig cfg = AppConfig::dsai_loadFromFile(tmpPath);

    cfg.backend       = Backend::OpenVINO;
    cfg.confThreshold = 0.60f;
    cfg.nmsThreshold  = 0.30f;
    cfg.webPort       = 7070;
    cfg.dsai_saveToFile(tmpPath);

    std::string saved = readFile(tmpPath);

    // Values must be updated
    EXPECT_CONTAINS(saved, "1",    "backend updated to OpenVINO (1)");
    EXPECT_CONTAINS(saved, "0.6",  "conf_threshold updated");
    EXPECT_CONTAINS(saved, "0.3",  "nms_threshold updated");
    EXPECT_CONTAINS(saved, "7070", "web_port updated");

    // Comments must still be there
    EXPECT_CONTAINS(saved, "# " + prefix + ": top-of-file header",
                    "top-of-file comment preserved");
    EXPECT_CONTAINS(saved, "# " + prefix + ": threshold section",
                    "threshold section comment preserved");
    EXPECT_CONTAINS(saved, "# " + prefix + ": roi section",
                    "roi section comment preserved");
    EXPECT_CONTAINS(saved, "# " + prefix + ": system section",
                    "system section comment preserved");
}

static void test_inline_comments_preserved() {
    const std::string tmpPath = "/tmp/dsai_test_inline.yaml";

    std::string yaml =
        "backend: 0  # 0=ONNX 1=OpenVINO 2=RKNN\n"
        "model_file: \"/tmp/m.onnx\"\n"
        "yaml_file: \"\"\n"
        "conf_threshold: 0.25  # lower = more detections\n"
        "nms_threshold: 0.45\n"
        "class_labels:\n"
        "  - person\n"
        "hidden_class_ids:\n"
        "source: 0\n"
        "camera_device_id: 0\n"
        "video_file: \"\"\n"
        "rtsp_url: \"/live/0\"\n"
        "iq_dir: \"/etc/iqfiles\"\n"
        "resolution_index: 0\n"
        "fps_index: 2\n"
        "gain: 50\n"
        "gamma: 100\n"
        "brightness: 0\n"
        "rockchip_hw: false\n"
        "roi_enabled: false\n"
        "roi:\n"
        "  x: 0\n"
        "  y: 0\n"
        "  width: 0\n"
        "  height: 0\n"
        "web_port: 8080\n"
        "debug_logging: false\n"
        "rtsp_port: 8554\n";

    writeFile(tmpPath, yaml);
    AppConfig cfg = AppConfig::dsai_loadFromFile(tmpPath);
    cfg.backend       = Backend::RKNN;
    cfg.confThreshold = 0.50f;
    cfg.dsai_saveToFile(tmpPath);

    std::string saved = readFile(tmpPath);
    EXPECT_CONTAINS(saved, "# 0=ONNX 1=OpenVINO 2=RKNN",
                    "inline comment on backend line preserved");
    EXPECT_CONTAINS(saved, "# lower = more detections",
                    "inline comment on conf_threshold preserved");
}

static void test_fresh_write_when_no_file() {
    const std::string tmpPath = "/tmp/dsai_test_fresh.yaml";
    std::remove(tmpPath.c_str());

    AppConfig cfg;
    cfg.confThreshold = 0.55f;
    cfg.dsai_saveToFile(tmpPath);

    std::string saved = readFile(tmpPath);
    EXPECT_CONTAINS(saved, "conf_threshold", "fresh file contains conf_threshold");
    EXPECT_CONTAINS(saved, "0.55",           "fresh file has correct value");
}

// ─── entry point ─────────────────────────────────────────────────────────────
int main() {
    printf("=== AppConfig comment preservation tests ===\n\n");
    test_all_comments_survive_roundtrip();
    printf("\n");
    test_values_updated_comments_preserved();
    printf("\n");
    test_inline_comments_preserved();
    printf("\n");
    test_fresh_write_when_no_file();
    printf("\n=== Results: %d passed, %d failed ===\n", s_passed, s_failed);
    return s_failed == 0 ? 0 : 1;
}
