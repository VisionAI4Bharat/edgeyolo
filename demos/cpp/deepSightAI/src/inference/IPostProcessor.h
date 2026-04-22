/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once
#include <vector>
#include <string>
#include <opencv2/core.hpp>

namespace inference {

struct Detection {
    cv::Rect2f rect;
    float      confidence;
    int        classId;
};

/**
 * Context for post-processing results from different backends.
 */
struct PostProcessContext {
    int modelWidth;
    int modelHeight;
    int numClasses;
    std::vector<std::string> classNames;
    
    // Backend specific output raw pointers
    std::vector<const float*>  outputFloats;
    std::vector<const int8_t*> outputInt8s;
    std::vector<float>         outputScales;
    std::vector<int>           outputZps;
};

class IPostProcessor {
public:
    virtual ~IPostProcessor() = default;
    virtual std::vector<Detection> dsai_process(const PostProcessContext& ctx, float confThr, float nmsThr) = 0;
};

}
