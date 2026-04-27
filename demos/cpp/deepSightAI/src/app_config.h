/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "inference/IPostProcessor.h"

struct RoiConfig {
    int x      = 0;
    int y      = 0;
    int width  = 0;
    int height = 0;
};

enum class SourceType { Camera = 0, VideoFile = 1, Rtsp = 2 };
enum class Backend    { ONNX   = 0, OpenVINO  = 1, RKNN = 2  };

struct AppConfig {
    Backend     backend       = Backend::RKNN;
    std::string modelFile;
    std::string yamlFile;
    float       confThreshold = 0.25f;
    float       nmsThreshold  = 0.45f;
    std::vector<std::string> classLabels;
    std::vector<int>         hiddenClassIds;

    SourceType  source          = SourceType::Camera;
    int         cameraDeviceId  = 0;
    std::string videoFile;
    std::string rtspUrl         = "/live/0";
    std::string iqDir           = "/etc/iqfiles";

    int         captureWidth    = 640;
    int         captureHeight   = 480;
    int         captureFps      = 30;
    int         gain            = 50;
    int         gamma           = 100;
    int         brightness      = 0;
    bool        rockchipHw      = false;

    bool        roiEnabled      = false;
    RoiConfig   roi;

    int         webPort         = 8080;
    int         rtspPort        = 8554;
    bool        debugLogging    = false;

    static AppConfig dsai_loadFromFile(const std::string& path);
    void dsai_saveToFile(const std::string& path) const;
    static std::string dsai_defaultPath();

    std::string dsai_logConfigToString() const;
    void        dsai_logConfig() const;
};
