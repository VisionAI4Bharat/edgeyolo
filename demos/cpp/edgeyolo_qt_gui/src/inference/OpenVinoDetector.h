#ifndef OPENVINODETECTOR_H
#define OPENVINODETECTOR_H

#ifdef WITH_OPENVINO

#include "IDetector.h"

#include <openvino/openvino.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>
#include <memory>

namespace inference {

/**
 * EdgeYOLO OpenVINO backend.
 *
 * Accepts the same EdgeYOLO fused-head ONNX export as OnnxDetector:
 *   input  : [B, 3, H, W]
 *   output : [B, N, 7]   (cx, cy, w, h, obj_conf, cls0, cls1, …)
 *
 * OpenVINO compiles the .onnx directly — no separate IR conversion needed.
 * Guarded by WITH_OPENVINO; the class does not exist if the flag is absent.
 *
 * Thread safety: NOT thread-safe. Create one instance per thread.
 */
class OpenVinoDetector : public IDetector {
public:
    explicit OpenVinoDetector() = default;
    ~OpenVinoDetector() override = default;

    /** Optional: set YAML path before load(). Falls back to <model>.yaml. */
    void setYamlPath(const std::string& yamlPath) { yamlPath_ = yamlPath; }

    // ── IDetector ─────────────────────────────────────────────────────────
    void load(const std::string& modelPath,
              float confThres = 0.25f,
              float nmsThres  = 0.45f) override;

    std::vector<Detection> infer(const cv::Mat& frame) override;

    const std::vector<std::string>& classNames() const override { return classNames_; }
    void setClassLabels(const std::vector<std::string>& labels) override {
        classNames_ = labels; numClasses_ = static_cast<int>(labels.size());
    }
    cv::Size inputSize() const override { return inputSize_; }
    bool     isLoaded()  const override { return loaded_; }

private:
    void loadYaml(const std::string& modelPath);

    std::vector<Detection> postProcess(const float* data,
                                       size_t       numDets,
                                       size_t       arrayLen,
                                       float        factor,
                                       cv::Size     oriSize) const;

    ov::Core                   core_;
    ov::CompiledModel          compiledModel_;
    ov::InferRequest           inferRequest_;

    cv::Size                   inputSize_{ 416, 416 };
    int                        numClasses_{ 0 };
    std::vector<std::string>   classNames_;

    float confThres_{ 0.25f };
    float nmsThres_{  0.45f };

    std::string yamlPath_;
    bool        loaded_{ false };
};

} // namespace inference

#endif // WITH_OPENVINO
#endif // OPENVINODETECTOR_H
