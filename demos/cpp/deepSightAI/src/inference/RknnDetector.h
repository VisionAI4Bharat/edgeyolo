/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once
#include "IDetector.h"
#include <memory>

namespace inference {

class RknnDetector : public IDetector {
public:
    RknnDetector();
    ~RknnDetector() override;

    void dsai_load(const std::string& modelPath, float confThres, float nmsThres) override;
    std::vector<Detection> dsai_infer(const cv::Mat& frame) override;
    const std::vector<std::string>& dsai_classNames() const override;
    void dsai_setClassLabels(const std::vector<std::string>& labels) override;
    cv::Size dsai_inputSize() const override;
    bool dsai_isLoaded() const override;
    void dsai_setYamlPath(const std::string& path) override;

    struct Impl;
private:
    std::unique_ptr<Impl> pImpl_;
};

}
