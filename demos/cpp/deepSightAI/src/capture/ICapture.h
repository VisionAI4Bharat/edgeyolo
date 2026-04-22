/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "../inference/IDetector.h" // For Detection struct

namespace deepSightAI {

/**
 * Abstract interface for multi-platform video capture.
 * Implementations handle hardware-specific specifics (VI/VENC/OpenCV).
 */
class ICapture {
public:
    virtual ~ICapture() = default;

    /** Open platform camera with requested params. */
    virtual bool dsai_openCamera(int devId, int width, int height, double fps) = 0;
    virtual bool dsai_openSource(const std::string& path) = 0;
    
    /** Open network RTSP/RTMP stream. */
    
    /** 
     * Get the next frame for inference.
     * Blocks or polls depending on implementation.
     */
    virtual bool dsai_read(cv::Mat& frame) = 0;
    
    /** Stop threads and release hardware handles. */
    virtual void dsai_release() = 0;
    
    virtual bool dsai_isOpened() const = 0;

    /** 
     * Update hardware-accelerated overlays. 
     * No-op if platform doesn't support RGN/OSD.
     */
    virtual void dsai_setOSD(const std::vector<inference::Detection>& detections) = 0;

    virtual int    dsai_captureWidth()  const = 0;
    virtual int    dsai_captureHeight() const = 0;
    virtual double dsai_captureFps()    const = 0;
    virtual std::string dsai_lastError() const = 0;
};

} // namespace deepSightAI
