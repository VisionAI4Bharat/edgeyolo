/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once

#include "ICapture.h"
#include <memory>

namespace deepSightAI {

class RV1106Capture : public ICapture {
public:
    RV1106Capture();
    ~RV1106Capture() override;

    bool dsai_openCamera(int devId, int width, int height, double fps) override;
    bool dsai_openSource(const std::string& path) override;
    bool dsai_read(cv::Mat& frame) override;
    void dsai_release() override;
    bool dsai_isOpened() const override;

    void dsai_setAppConfig(const AppConfig& cfg) override;
    void dsai_setModelInputSize(int w, int h) override;

    void dsai_setOSD(const std::vector<inference::Detection>& detections) override;

    int    dsai_captureWidth()  const override;
    int    dsai_captureHeight() const override;
    double dsai_captureFps()    const override;
    std::string dsai_lastError() const override;

    struct Impl;

private:
    std::unique_ptr<Impl> pImpl_;
};

} // namespace deepSightAI
