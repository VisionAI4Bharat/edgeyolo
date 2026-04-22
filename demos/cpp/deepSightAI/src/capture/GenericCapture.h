/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once
#include "ICapture.h"
#include <opencv2/core.hpp>

// On desktop/headless, we usually have videoio. On ARM/mobile, we might not.
#if __has_include(<opencv2/videoio.hpp>)
#include <opencv2/videoio.hpp>
#ifndef HAVE_OPENCV_VIDEOIO
#define HAVE_OPENCV_VIDEOIO
#endif
#endif

namespace deepSightAI {

class GenericCapture : public ICapture {
public:
    GenericCapture();
    ~GenericCapture() override;

    bool dsai_openCamera(int devId, int width, int height, double fps) override;
    bool dsai_read(cv::Mat& frame) override;
    void dsai_release() override;
    bool dsai_isOpened() const override;
    void dsai_setOSD(const std::vector<inference::Detection>&) override {}

    int    dsai_captureWidth()  const override;
    int    dsai_captureHeight() const override;
    double dsai_captureFps()    const override;
    std::string dsai_lastError() const override { return lastErr_; }

private:
#ifdef HAVE_OPENCV_VIDEOIO
    cv::VideoCapture cap_;
#endif
    bool isOpen_ = false;
    std::string lastErr_;
};

}
