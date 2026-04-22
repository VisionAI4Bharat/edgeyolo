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

struct PostProcessContext {
    const float* data;
    size_t       numProposals;
    int          numClasses;
    std::vector<std::string> classNames;
    int          modelWidth;
    int          modelHeight;
    float        scaleX; 
    float        scaleY;
};

class IPostProcessor {
public:
    virtual ~IPostProcessor() = default;
    virtual std::vector<Detection> dsai_process(const PostProcessContext& ctx, float confThr, float nmsThr) = 0;
};

}
