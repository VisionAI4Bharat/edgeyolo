#ifndef ONNXDETECTOR_H
#define ONNXDETECTOR_H

#include "IDetector.h"

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/core.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace inference {

/**
 * EdgeYOLO ONNX Runtime backend.
 *
 * Expects the EdgeYOLO fused-head ONNX export:
 *   input  : [B, 3, H, W]   (model was exported with B=16; we run B=1)
 *   output : [B, N, 7]       where 7 = cx, cy, w, h, obj_conf, cls0_score, cls1_score, …
 *
 * A sidecar YAML (same basename as the .onnx, or supplied via setYamlPath) must
 * contain at minimum:
 *   names: [classA, classB, …]
 * Optionally:
 *   img_size: [H, W]   (overrides shape read from the model graph)
 *
 * The model exported batch size is ignored at runtime; a single frame is
 * forwarded by constructing a [1, 3, H, W] input tensor.
 *
 * Thread safety: NOT thread-safe. Create one instance per thread.
 */
class OnnxDetector : public IDetector {
public:
    explicit OnnxDetector() = default;
    ~OnnxDetector() override = default;

    /**
     * Optionally set the YAML config path before calling load().
     * If not called, load() looks for <modelBaseName>.yaml next to the model.
     */
    void setYamlPath(const std::string& yamlPath) { yamlPath_ = yamlPath; }

    // ── IDetector ─────────────────────────────────────────────────────────

    /**
     * @throws std::runtime_error if model or YAML cannot be loaded,
     *         or if the output tensor shape is unexpected.
     */
    void load(const std::string& modelPath,
              float confThres = 0.25f,
              float nmsThres  = 0.45f) override;

    /**
     * @throws std::runtime_error on ORT session run failure.
     */
    std::vector<Detection> infer(const cv::Mat& frame) override;

    const std::vector<std::string>& classNames() const override { return classNames_; }
    void setClassLabels(const std::vector<std::string>& labels) override {
        classNames_ = labels; numClasses_ = static_cast<int>(labels.size());
    }
    cv::Size inputSize() const override { return inputSize_; }
    bool     isLoaded()  const override { return loaded_; }

private:
    void loadYaml(const std::string& modelPath);
    void buildSession(const std::string& modelPath);

    // ORT objects
    Ort::Env                          env_{ ORT_LOGGING_LEVEL_WARNING, "OnnxDetector" };
    Ort::SessionOptions               sessionOptions_;
    std::unique_ptr<Ort::Session>     session_;
    Ort::AllocatorWithDefaultOptions  allocator_;

    // Input / output names (owned strings kept alive for the session lifetime)
    std::vector<std::string>          inputNameStrs_;
    std::vector<std::string>          outputNameStrs_;
    std::vector<const char*>          inputNames_;
    std::vector<const char*>          outputNames_;

    // Model metadata
    cv::Size                          inputSize_{ 416, 416 };
    int                               numClasses_{ 0 };
    std::vector<std::string>          classNames_;

    // Inference parameters
    float confThres_{ 0.25f };
    float nmsThres_{  0.45f };

    // Pre-allocated preprocessing buffers (avoid per-frame heap allocs)
    std::vector<float>        blob_;
    std::vector<cv::Mat>      splitChannels_;
    std::array<int64_t, 4>   inShape_{ 1, 3, 0, 0 };  // filled in buildSession
    Ort::MemoryInfo           memInfo_{ Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault) };

    std::string yamlPath_;
    bool        loaded_{ false };
};

} // namespace inference

#endif // ONNXDETECTOR_H
