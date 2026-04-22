#pragma once
#include "IDetector.h"
#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

namespace inference {

class OnnxDetector : public IDetector {
public:
    OnnxDetector() : env_(ORT_LOGGING_LEVEL_WARNING, "deepSightAI") {}
    ~OnnxDetector() override = default;

    void dsai_load(const std::string& modelPath, float confThres, float nmsThres) override;
    std::vector<Detection> dsai_infer(const cv::Mat& frame) override;
    const std::vector<std::string>& dsai_classNames() const override { return classNames_; }
    void dsai_setClassLabels(const std::vector<std::string>& labels) override { classNames_ = labels; numClasses_ = labels.size(); }
    cv::Size dsai_inputSize() const override { return inputSize_; }
    bool dsai_isLoaded() const override { return loaded_; }
    void dsai_setYamlPath(const std::string& path) override { yamlPath_ = path; }

private:
    void loadYaml(const std::string& modelPath);
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    cv::Size inputSize_{0, 0};
    std::vector<std::string> classNames_;
    int numClasses_ = 0;
    float confThres_ = 0.25f;
    float nmsThres_ = 0.45f;
    bool loaded_ = false;
    std::string yamlPath_;
};

}
