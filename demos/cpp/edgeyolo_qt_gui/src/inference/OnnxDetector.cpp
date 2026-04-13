#include "OnnxDetector.h"

#include <opencv2/imgproc.hpp>

// Re-use EdgeYOLO's own letterbox + NMS utilities
#include "../../../../../third_party/edgeyolo/deployment/yolo/common.hpp"

#include <filesystem>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;

namespace inference {

// ─── helpers ──────────────────────────────────────────────────────────────────

namespace {

// Build the [1, 3, H, W] NCHW float32 input tensor from a letterboxed BGR frame.
// Returns the allocated buffer (must stay alive until after Run()).
std::vector<float> mat2blob(const cv::Mat& letterboxed, int H, int W)
{
    // letterboxed is already H×W BGR uint8
    std::vector<float> blob(1 * 3 * H * W);
    const int planeSize = H * W;

    for (int r = 0; r < H; ++r) {
        const uchar* row = letterboxed.ptr<uchar>(r);
        for (int c = 0; c < W; ++c) {
            blob[0 * planeSize + r * W + c] = row[c * 3 + 0] / 255.0f; // B
            blob[1 * planeSize + r * W + c] = row[c * 3 + 1] / 255.0f; // G
            blob[2 * planeSize + r * W + c] = row[c * 3 + 2] / 255.0f; // R
        }
    }
    return blob;
}

} // anonymous namespace

// ─── OnnxDetector implementation ─────────────────────────────────────────────

void OnnxDetector::load(const std::string& modelPath, float confThres, float nmsThres)
{
    if (modelPath.empty())
        throw std::runtime_error("OnnxDetector: model path is empty");

    if (!fs::exists(modelPath))
        throw std::runtime_error("OnnxDetector: model file not found: " + modelPath);

    confThres_ = confThres;
    nmsThres_  = nmsThres;

    loadYaml(modelPath);
    buildSession(modelPath);

    loaded_ = true;
}

void OnnxDetector::loadYaml(const std::string& modelPath)
{
    // Determine YAML path: explicit override or <basename>.yaml next to model
    std::string yamlPath = yamlPath_;
    if (yamlPath.empty()) {
        fs::path p(modelPath);
        yamlPath = (p.parent_path() / p.stem()).string() + ".yaml";
    }

    if (!fs::exists(yamlPath))
        throw std::runtime_error(
            "OnnxDetector: sidecar YAML not found: " + yamlPath +
            "\n  Call setYamlPath() before load() to specify its location.");

    try {
        YAML::Node cfg = YAML::LoadFile(yamlPath);

        if (!cfg["names"])
            throw std::runtime_error("OnnxDetector: YAML missing 'names' key: " + yamlPath);

        classNames_ = cfg["names"].as<std::vector<std::string>>();
        numClasses_ = static_cast<int>(classNames_.size());

        if (numClasses_ == 0)
            throw std::runtime_error("OnnxDetector: 'names' list is empty in: " + yamlPath);

        // Optional explicit input size
        if (cfg["img_size"]) {
            auto sz = cfg["img_size"].as<std::vector<int>>();
            if (sz.size() >= 2) {
                inputSize_.height = sz[0];
                inputSize_.width  = sz[1];
            }
        }
    }
    catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("OnnxDetector: YAML parse error in ") + yamlPath + ": " + e.what());
    }
}

void OnnxDetector::buildSession(const std::string& modelPath)
{
    sessionOptions_.SetIntraOpNumThreads(1);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    try {
        session_ = std::make_unique<Ort::Session>(
            env_,
            modelPath.c_str(),
            sessionOptions_
        );
    }
    catch (const Ort::Exception& e) {
        throw std::runtime_error(
            std::string("OnnxDetector: failed to create ORT session: ") + e.what());
    }

    // ── collect input names ────────────────────────────────────────────────
    const size_t numInputs = session_->GetInputCount();
    if (numInputs == 0)
        throw std::runtime_error("OnnxDetector: model has no inputs");

    inputNameStrs_.clear();
    inputNames_.clear();
    for (size_t i = 0; i < numInputs; ++i) {
        auto namePtr = session_->GetInputNameAllocated(i, allocator_);
        inputNameStrs_.push_back(namePtr.get());
        inputNames_.push_back(inputNameStrs_.back().c_str());
    }

    // ── collect output names ───────────────────────────────────────────────
    const size_t numOutputs = session_->GetOutputCount();
    if (numOutputs == 0)
        throw std::runtime_error("OnnxDetector: model has no outputs");

    outputNameStrs_.clear();
    outputNames_.clear();
    for (size_t i = 0; i < numOutputs; ++i) {
        auto namePtr = session_->GetOutputNameAllocated(i, allocator_);
        outputNameStrs_.push_back(namePtr.get());
        outputNames_.push_back(outputNameStrs_.back().c_str());
    }

    // ── validate and resolve input spatial size from model graph ───────────
    // Input shape is [B, 3, H, W]; B may be symbolic / fixed — we only care about H, W.
    auto inputInfo  = session_->GetInputTypeInfo(0);
    auto inputShape = inputInfo.GetTensorTypeAndShapeInfo().GetShape();

    if (inputShape.size() != 4)
        throw std::runtime_error(
            "OnnxDetector: expected 4-D input [B,3,H,W], got rank " +
            std::to_string(inputShape.size()));

    // Dimension may be -1 (dynamic); prefer YAML-provided size in that case.
    if (inputShape[2] > 0)
        inputSize_.height = static_cast<int>(inputShape[2]);
    if (inputShape[3] > 0)
        inputSize_.width  = static_cast<int>(inputShape[3]);

    // ── validate output shape ──────────────────────────────────────────────
    // Expected: [B, N, L] where L = 5 + numClasses
    auto outputInfo  = session_->GetOutputTypeInfo(0);
    auto outputShape = outputInfo.GetTensorTypeAndShapeInfo().GetShape();

    if (outputShape.size() != 3)
        throw std::runtime_error(
            "OnnxDetector: expected 3-D output [B,N,L], got rank " +
            std::to_string(outputShape.size()));

    const int64_t L = outputShape[2];
    const int expectedL = 5 + numClasses_;
    if (L > 0 && L != expectedL)
        throw std::runtime_error(
            "OnnxDetector: output last-dim=" + std::to_string(L) +
            " but expected 5+numClasses=" + std::to_string(expectedL) +
            ". Check that the YAML names match the model.");
}

std::vector<Detection> OnnxDetector::infer(const cv::Mat& frame)
{
    if (!loaded_)
        throw std::runtime_error("OnnxDetector: call load() before infer()");

    if (frame.empty())
        throw std::runtime_error("OnnxDetector: infer() called with empty frame");

    const int H = inputSize_.height;
    const int W = inputSize_.width;

    // ── pre-process ────────────────────────────────────────────────────────
    cv::Mat bgrFrame;
    if (frame.channels() == 4)
        cv::cvtColor(frame, bgrFrame, cv::COLOR_BGRA2BGR);
    else if (frame.channels() == 1)
        cv::cvtColor(frame, bgrFrame, cv::COLOR_GRAY2BGR);
    else
        bgrFrame = frame;

    detect::resizeInfo rzInfo = detect::resizeAndPad(bgrFrame, cv::Size(W, H), false, false);
    const float factor = rzInfo.factor;

    std::vector<float> blob = mat2blob(rzInfo.resized_img, H, W);

    // ── build input tensor [1, 3, H, W] ───────────────────────────────────
    std::array<int64_t, 4> inputShape{ 1, 3, H, W };
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        blob.data(),
        blob.size(),
        inputShape.data(),
        inputShape.size()
    );

    // ── run ────────────────────────────────────────────────────────────────
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(inputTensor));

    std::vector<Ort::Value> outputs;
    try {
        outputs = session_->Run(
            Ort::RunOptions{ nullptr },
            inputNames_.data(),
            inputTensors.data(),
            inputTensors.size(),
            outputNames_.data(),
            outputNames_.size()
        );
    }
    catch (const Ort::Exception& e) {
        throw std::runtime_error(
            std::string("OnnxDetector: ORT session Run failed: ") + e.what());
    }

    if (outputs.empty())
        throw std::runtime_error("OnnxDetector: session returned no output tensors");

    // ── post-process ───────────────────────────────────────────────────────
    const auto  outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    // outShape: [batchSize, numDets, arrayLen]
    const int64_t numDets  = outShape[1];
    const int64_t arrayLen = outShape[2];
    const float*  data     = outputs[0].GetTensorData<float>();

    return postProcess(data, numDets, arrayLen, factor, bgrFrame.size());
}

std::vector<Detection> OnnxDetector::postProcess(
    const float* data,
    int64_t      numDets,
    int64_t      arrayLen,
    float        factor,
    cv::Size     oriSize) const
{
    // Mirrors mnn_det::YOLO::generate_yolo_proposals + _decode_output.
    // Layout per detection: [cx, cy, w, h, obj_conf, cls0_score, cls1_score, …]

    std::vector<detect::Object> proposals;
    proposals.reserve(static_cast<size_t>(numDets / 4));

    for (int64_t i = 0; i < numDets; ++i) {
        const float* det = data + i * arrayLen;

        const float objConf = det[4];
        if (objConf < confThres_) continue;

        // Find best class
        int   bestClass = 0;
        float bestScore = det[5];
        for (int c = 1; c < numClasses_; ++c) {
            if (det[5 + c] > bestScore) {
                bestScore = det[5 + c];
                bestClass = c;
            }
        }

        const float confidence = objConf * bestScore;
        if (confidence < confThres_) continue;

        detect::Object obj;
        obj.label       = bestClass;
        obj.prob        = confidence;
        obj.rect.width  = det[2];
        obj.rect.height = det[3];
        obj.rect.x      = det[0] - obj.rect.width  * 0.5f;
        obj.rect.y      = det[1] - obj.rect.height * 0.5f;
        proposals.push_back(obj);
    }

    detect::qsort_descent_inplace(proposals);

    std::vector<int> picked;
    detect::nms_sorted_bboxes(proposals, picked, nmsThres_);

    std::vector<Detection> detections;
    detections.reserve(picked.size());

    for (int idx : picked) {
        const detect::Object& o = proposals[idx];

        // Scale back from letterboxed input to original image
        float x0 = o.rect.x      * factor;
        float y0 = o.rect.y      * factor;
        float x1 = (o.rect.x + o.rect.width)  * factor;
        float y1 = (o.rect.y + o.rect.height) * factor;

        // Clip to image bounds
        x0 = std::clamp(x0, 0.f, static_cast<float>(oriSize.width  - 1));
        y0 = std::clamp(y0, 0.f, static_cast<float>(oriSize.height - 1));
        x1 = std::clamp(x1, 0.f, static_cast<float>(oriSize.width  - 1));
        y1 = std::clamp(y1, 0.f, static_cast<float>(oriSize.height - 1));

        if (x1 <= x0 || y1 <= y0) continue;

        Detection d;
        d.rect       = { x0, y0, x1 - x0, y1 - y0 };
        d.classId    = o.label;
        d.confidence = o.prob;
        detections.push_back(d);
    }

    return detections;
}

} // namespace inference
