/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once
#include "IPreProcessor.h"
#include <opencv2/imgproc.hpp>
#include <cstring>

#ifdef WITH_RGA
#include "im2d.h"
#include "rga.h"
#endif

namespace inference {

class EdgeYoloPreProcessor : public IPreProcessor {
public:
    void dsai_process(const cv::Mat& src, const PreProcessContext& ctx) override {
        bool accelerated = false;

#ifdef WITH_RGA
        try {
            rga_buffer_t rga_src = wrapbuffer_virtualaddr((void*)src.data, src.cols, src.rows, RK_FORMAT_BGR_888);
            rga_buffer_t rga_dst = wrapbuffer_virtualaddr(ctx.dstBuffer, ctx.targetWidth, ctx.targetHeight, RK_FORMAT_BGR_888);
            if (imresize(rga_src, rga_dst) == IM_STATUS_SUCCESS) {
                accelerated = true;
            }
        } catch (...) { accelerated = false; }
#endif

        if (!accelerated) {
            cv::Mat resized;
            cv::resize(src, resized, cv::Size(ctx.targetWidth, ctx.targetHeight));
            // EdgeYOLO: BGR 0-255, no normalization. 
            // Most backends (OpenVINO/ONNX) expect CHW, so we handle that if needed, 
            // but RKNN expects HWC (standard BGR).
            
            // Note: For OpenVINO/ONNX which often want FP32 CHW, 
            // the backend-specific dsai_infer should handle the HWC->CHW conversion 
            // after this pre-processor provides the resized BGR frame.
            memcpy(ctx.dstBuffer, resized.data, ctx.dstSize);
        }
    }
};

}
