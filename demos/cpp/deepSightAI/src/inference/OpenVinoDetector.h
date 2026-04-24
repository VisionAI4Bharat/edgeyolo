#pragma once
#include "IDetector.h"
#include <openvino/openvino.hpp>

namespace inference {

class OpenVinoDetector : public IDetector {
public:
    explicit OpenVinoDetector() = default;
    ~OpenVinoDetector() override = default;

    void dsai_load(const std::string& modelPath, float confThres, float nmsThres) override;
    std::vector<Detection> dsai_infer(const cv::Mat& frame) override;
    const std::vector<std::string>& dsai_classNames() const override { return classNames_; }
    void dsai_setClassLabels(const std::vector<std::string>& labels) override { classNames_ = labels; numClasses_ = labels.size(); }
    cv::Size dsai_inputSize() const override { return inputSize_; }
    bool dsai_isLoaded() const override { return loaded_; }
    void dsai_setYamlPath(const std::string& path) override { yamlPath_ = path; }

private:
    ov::Core core_;
    ov::CompiledModel compiledModel_;
    ov::InferRequest inferRequest_;
    cv::Size inputSize_{0, 0};
    std::vector<std::string> classNames_;
    int numClasses_ = 0;
    float confThres_ = 0.25f;
    float nmsThres_ = 0.45f;
    bool loaded_ = false;
    std::string yamlPath_;
    std::vector<float> inputBlob_;
};

}
