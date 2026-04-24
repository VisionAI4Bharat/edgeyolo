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
        cv::Mat resized;
        bool accelerated = false;

#ifdef WITH_RGA
        // RGA hardware resize writes HWC uint8 directly — only used for RKNN (outputCHW=false)
        if (!ctx.outputCHW) {
            try {
                rga_buffer_t rga_src = wrapbuffer_virtualaddr((void*)src.data, src.cols, src.rows, RK_FORMAT_BGR_888);
                rga_buffer_t rga_dst = wrapbuffer_virtualaddr(ctx.dstBuffer, ctx.targetWidth, ctx.targetHeight, RK_FORMAT_BGR_888);
                if (imresize(rga_src, rga_dst) == IM_STATUS_SUCCESS) {
                    accelerated = true;
                }
            } catch (...) { accelerated = false; }
        }
#endif

        if (!accelerated) {
            cv::resize(src, resized, cv::Size(ctx.targetWidth, ctx.targetHeight));
        }

        if (ctx.outputCHW) {
            // SIMD-vectorized HWC BGR uint8 → CHW BGR float32 (no normalization, range 0-255)
            cv::Mat channels[3];
            cv::split(resized, channels);
            float* dst = static_cast<float*>(ctx.dstBuffer);
            const int plane = ctx.targetWidth * ctx.targetHeight;
            for (int c = 0; c < 3; c++) {
                cv::Mat chanFloat(ctx.targetHeight, ctx.targetWidth, CV_32F, dst + c * plane);
                channels[c].convertTo(chanFloat, CV_32F);
            }
        } else if (!accelerated) {
            memcpy(ctx.dstBuffer, resized.data, ctx.dstSize);
        }
    }
};

}
