/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 *
 * This software is dual-licensed:
 * 1. GNU General Public License v3.0 (GPLv3)
 * 2. A proprietary license for commercial use.
 *
 * You may use this software under the terms of the GPLv3 if you are using it
 * for non-commercial purposes. For commercial usage, a separate commercial
 * license must be obtained from swatah.ai (info@swatah.ai).
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * Trademarks: All trademarks, service marks, and logos are the property of
 * their respective owners.
 */

#pragma once

#include <string>
#include <vector>
#include <stdexcept>

// AppConfig mirrors every field in ConfigDialog's YAML schema so the same
// config.yaml works for both the Qt GUI and the headless embedded binary.

struct RoiConfig {
    int x      = 0;
    int y      = 0;
    int width  = 0;
    int height = 0;
};

enum class SourceType { Camera = 0, VideoFile = 1, Rtsp = 2 };
enum class Backend    { ONNX   = 0, OpenVINO  = 1, RKNN = 2  };

struct AppConfig {
    // ── inference ──────────────────────────────────────────────────────────
    Backend     backend       = Backend::RKNN;
    std::string modelFile;
    std::string yamlFile;
    float       confThreshold = 0.25f;
    float       nmsThreshold  = 0.45f;
    std::vector<std::string> classLabels;

    // ── video source ───────────────────────────────────────────────────────
    SourceType  source          = SourceType::Camera;
    int         cameraDeviceId  = 0;
    std::string videoFile;
    std::string rtspUrl;          // e.g. rtsp://192.168.1.x:554/stream
    std::string iqDir           = "/etc/iqfiles";  // RKAIQ ISP tuning files

    // resolution / fps (mirrors combo-box indices from ConfigDialog)
    int         resolutionIndex = 0;   // 0=640×480  1=1280×720  2=1920×1080
    int         fpsIndex        = 2;   // 0=15  1=25  2=30  3=60  4=90

    // V4L2 / Camera controls
    int         gain            = 50;
    int         gamma           = 100;
    int         brightness      = 0;

    // backend-specific
    bool        rockchipHw      = false;

    // ── ROI ────────────────────────────────────────────────────────────────
    bool        roiEnabled      = false;
    RoiConfig   roi;

    // ── headless / web server ──────────────────────────────────────────────
    int         webPort         = 8080;   // web config dashboard port
    bool        debugLogging    = false;

    // ── helpers ────────────────────────────────────────────────────────────
    int.width()  const;
    int.height() const;
    int  dsai_fps()    const;

    // Load from / save to a YAML file. Throws std::runtime_error on failure.
    static AppConfig dsai_loadFromFile(const std::string& path);
    void dsai_saveToFile(const std::string& path) const;

    // Default config-file location (XDG or /etc fallback)
    static std::string dsai_defaultPath();
};
