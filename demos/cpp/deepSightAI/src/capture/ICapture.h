/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "../inference/IDetector.h"

namespace inference { struct Detection; }
struct AppConfig;

namespace deepSightAI {

struct CameraMode {
    int width  = 0;
    int height = 0;
    int fps    = 0;

    std::string label() const {
        return std::to_string(width) + " × " + std::to_string(height)
               + " @ " + std::to_string(fps) + " fps";
    }
};

class ICapture {
public:
    virtual ~ICapture() = default;

    virtual void dsai_setAppConfig(const AppConfig& cfg) {}
    virtual void dsai_setModelInputSize(int w, int h) {}

    virtual bool dsai_openCamera(int devId, int width, int height, double fps) = 0;
    virtual bool dsai_openSource(const std::string& path) = 0;
    virtual bool dsai_read(cv::Mat& frame) = 0;
    virtual void dsai_release() = 0;
    virtual bool dsai_isOpened() const = 0;
    virtual void dsai_setOSD(const std::vector<inference::Detection>& detections) = 0;

    virtual int    dsai_captureWidth()  const = 0;
    virtual int    dsai_captureHeight() const = 0;
    virtual double dsai_captureFps()    const = 0;
    virtual std::string dsai_lastError() const = 0;

    /**
     * Enumerate modes available on a V4L2 camera device.
     * Returns an empty list on non-Linux platforms or if the device
     * cannot be queried. The list is sorted: highest resolution first,
     * highest fps first within the same resolution.
     */
    static std::vector<CameraMode> dsai_enumerateModes(int devId);
};

} // namespace deepSightAI
