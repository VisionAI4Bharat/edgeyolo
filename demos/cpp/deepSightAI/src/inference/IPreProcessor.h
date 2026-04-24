/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once
#include <opencv2/core.hpp>

namespace inference {

struct PreProcessContext {
    int targetWidth;
    int targetHeight;
    void* dstBuffer;   // Pointer to model input tensor memory
    size_t dstSize;    // Size of the input tensor in bytes (used for HWC uint8 memcpy path)
    bool outputCHW = false; // If true, write float32 CHW into dstBuffer; otherwise write uint8 HWC
};

class IPreProcessor {
public:
    virtual ~IPreProcessor() = default;
    virtual void dsai_process(const cv::Mat& src, const PreProcessContext& ctx) = 0;
};

}
