/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#ifndef IDETECTOR_H
#define IDETECTOR_H

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include "IPostProcessor.h"
#include "IPreProcessor.h"

namespace inference {

class IDetector {
public:
    virtual ~IDetector() = default;

    virtual void dsai_load(const std::string& modelPath,
                      float confThres = 0.25f,
                      float nmsThres  = 0.45f) = 0;

    virtual std::vector<Detection> dsai_infer(const cv::Mat& frame) = 0;
    virtual const std::vector<std::string>& dsai_classNames() const = 0;
    virtual void dsai_setClassLabels(const std::vector<std::string>& labels) = 0;
    virtual cv::Size dsai_inputSize() const = 0;
    virtual bool dsai_isLoaded() const = 0;
    virtual void dsai_setYamlPath(const std::string& path) = 0;

protected:
    std::unique_ptr<IPostProcessor> postProcessor_;
    std::unique_ptr<IPreProcessor> preProcessor_;
};

} // namespace inference

#endif
