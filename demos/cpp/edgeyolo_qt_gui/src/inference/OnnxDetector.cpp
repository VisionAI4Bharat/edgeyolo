#include "OnnxDetector.h"
#include "../debug_log.h"

#include <opencv2/imgproc.hpp>

// EdgeYOLO shared utilities — letterbox resize, NMS, Object struct
#include "edgeyolo_bridge.h"
// EdgeYOLO platform utilities — generate_yolo_proposals, qsort, nms (vector<vector<float>>)
#include "../../../../../third_party/edgeyolo/deployment/yolo/platform/common.hpp"

#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <unordered_map>

namespace fs = std::filesystem;

#define TAG "OnnxDetector"

namespace inference {

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

    DBG_LOG(TAG, "loaded '%s' — %d classes, input %dx%d, conf=%.2f, nms=%.2f\n",
        modelPath.c_str(), numClasses_,
        inputSize_.width, inputSize_.height,
        static_cast<double>(confThres_), static_cast<double>(nmsThres_));
}

void OnnxDetector::loadYaml(const std::string& modelPath)
{
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

        YAML::Node labelsNode;
        if (cfg["class_labels"])      labelsNode = cfg["class_labels"];
        else if (cfg["names"])        labelsNode = cfg["names"];
        else throw std::runtime_error("OnnxDetector: YAML missing 'class_labels' key: " + yamlPath);

        classNames_ = labelsNode.as<std::vector<std::string>>();
        numClasses_ = static_cast<int>(classNames_.size());

        if (numClasses_ == 0)
            throw std::runtime_error("OnnxDetector: 'class_labels' list is empty in: " + yamlPath);

        if (cfg["img_size"]) {
            auto sz = cfg["img_size"].as<std::vector<int>>();
            if (sz.size() >= 2) {
                inputSize_.height = sz[0];
                inputSize_.width  = sz[1];
            }
        }

        DBG_LOG(TAG, "YAML loaded: %d classes from '%s'\n", numClasses_, yamlPath.c_str());
    }
    catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("OnnxDetector: YAML parse error in ") + yamlPath + ": " + e.what());
    }
}

void OnnxDetector::buildSession(const std::string& modelPath)
{
    // Thread counts derived from hardware — no hardcoded fallback values.
    // hardware_concurrency() returns 0 when indeterminate; passing 0 to ORT means "auto".
    const unsigned int hw          = std::thread::hardware_concurrency();
    const int nThreads             = static_cast<int>(hw);
    // Allow independent graph nodes (FPN branches, detection heads) to run concurrently.
    const int nInterOpThreads      = (hw >= 4) ? 2 : 1;

    sessionOptions_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    sessionOptions_.SetIntraOpNumThreads(nThreads);
    sessionOptions_.SetInterOpNumThreads(nInterOpThreads);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions_.AddConfigEntry("session.intra_op.allow_spinning", "1");

    // ── Try OpenVINO EP (Intel oneDNN; 2-5× speedup on Intel CPUs) ────────────
    // Requires ORT built with OpenVINO support; silently falls back to CPU EP if absent.
    std::string activeEP = "CPU";
    try {
        std::unordered_map<std::string, std::string> ovOpts;
        ovOpts["device_type"]    = "CPU_FP32";
        ovOpts["num_of_threads"] = std::to_string(nThreads > 0 ? nThreads : 0);
        sessionOptions_.AppendExecutionProvider("OpenVINO", ovOpts);
        activeEP = "OpenVINO";
    } catch (const Ort::Exception&) {
        // ORT not compiled with OpenVINO EP — fall through to default CPU EP.
    }

    try {
        session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);
    }
    catch (const Ort::Exception& e) {
        throw std::runtime_error(
            std::string("OnnxDetector: failed to create ORT session: ") + e.what());
    }

    // Input names
    const size_t numInputs = session_->GetInputCount();
    if (numInputs == 0)
        throw std::runtime_error("OnnxDetector: model has no inputs");

    inputNameStrs_.clear();
    inputNames_.clear();
    for (size_t i = 0; i < numInputs; ++i) {
        auto p = session_->GetInputNameAllocated(i, allocator_);
        inputNameStrs_.push_back(p.get());
        inputNames_.push_back(inputNameStrs_.back().c_str());
    }

    // Output names
    const size_t numOutputs = session_->GetOutputCount();
    if (numOutputs == 0)
        throw std::runtime_error("OnnxDetector: model has no outputs");

    outputNameStrs_.clear();
    outputNames_.clear();
    for (size_t i = 0; i < numOutputs; ++i) {
        auto p = session_->GetOutputNameAllocated(i, allocator_);
        outputNameStrs_.push_back(p.get());
        outputNames_.push_back(outputNameStrs_.back().c_str());
    }

    // Resolve input spatial size from model graph (may override YAML value)
    auto inputInfo  = session_->GetInputTypeInfo(0);
    auto inputShape = inputInfo.GetTensorTypeAndShapeInfo().GetShape();

    if (inputShape.size() != 4)
        throw std::runtime_error(
            "OnnxDetector: expected 4-D input [B,3,H,W], got rank " +
            std::to_string(inputShape.size()));

    if (inputShape[2] > 0) inputSize_.height = static_cast<int>(inputShape[2]);
    if (inputShape[3] > 0) inputSize_.width  = static_cast<int>(inputShape[3]);

    // Validate output shape: expected [B, N, 5+numClasses]
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

    // Pre-allocate CHW float blob, split-channel staging buffers, and fixed input tensor shape.
    const int H = inputSize_.height;
    const int W = inputSize_.width;
    blob_.assign(3 * H * W, 0.f);
    splitChannels_.resize(3);
    for (int c = 0; c < 3; ++c)
        splitChannels_[c] = cv::Mat(H, W, CV_32F, blob_.data() + c * H * W);

    inShape_ = { 1, 3, static_cast<int64_t>(H), static_cast<int64_t>(W) };

    DBG_LOG(TAG, "session built — EP=%s  input [B,3,%d,%d]  output [B,%lld,%lld]  "
        "intra=%d  inter=%d\n",
        activeEP.c_str(), H, W,
        static_cast<long long>(outputShape[1]),
        static_cast<long long>(L),
        nThreads, nInterOpThreads);
}

std::vector<Detection> OnnxDetector::infer(const cv::Mat& frame)
{
    if (!loaded_)
        throw std::runtime_error("OnnxDetector: call load() before infer()");
    if (frame.empty())
        throw std::runtime_error("OnnxDetector: infer() called with empty frame");

    const int H = inputSize_.height;
    const int W = inputSize_.width;

    // ── pre-process ─────────────────────────────────────────────────────────
    cv::Mat bgrFrame;
    if (frame.channels() == 4)
        cv::cvtColor(frame, bgrFrame, cv::COLOR_BGRA2BGR);
    else if (frame.channels() == 1)
        cv::cvtColor(frame, bgrFrame, cv::COLOR_GRAY2BGR);
    else
        bgrFrame = frame;

    // Letterbox to model input size (aspect-ratio preserving, pad with 114)
    detect::resizeInfo rzInfo = detect::resizeAndPad(bgrFrame, cv::Size(W, H), false, false);
    const float factor = rzInfo.factor;

    DBG_LOG(TAG, "preprocess: src=%dx%d → letterbox=%dx%d, factor=%.3f\n",
        bgrFrame.cols, bgrFrame.rows,
        rzInfo.resized_img.cols, rzInfo.resized_img.rows,
        static_cast<double>(factor));

    // HWC→CHW float32, BGR order, 0-255 range (no normalization).
    // cv::split+convertTo uses SIMD internally; writes sequentially into pre-alloc'd blob_.
    {
        std::vector<cv::Mat> u8planes(3);
        cv::split(rzInfo.resized_img, u8planes);
        for (int c = 0; c < 3; ++c)
            u8planes[c].convertTo(splitChannels_[c], CV_32F);
    }

    // ── build input tensor [1, 3, H, W] — reuse pre-allocated buffers ────────
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo_, blob_.data(), blob_.size(), inShape_.data(), inShape_.size());

    // ── run ─────────────────────────────────────────────────────────────────
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(inputTensor));

    std::vector<Ort::Value> outputs;
    try {
        outputs = session_->Run(
            Ort::RunOptions{ nullptr },
            inputNames_.data(),  inputTensors.data(), inputTensors.size(),
            outputNames_.data(), outputNames_.size());
    }
    catch (const Ort::Exception& e) {
        throw std::runtime_error(
            std::string("OnnxDetector: ORT Run failed: ") + e.what());
    }

    if (outputs.empty())
        throw std::runtime_error("OnnxDetector: session returned no outputs");

    // ── post-process ────────────────────────────────────────────────────────
    const auto   outShape  = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int    numDets   = static_cast<int>(outShape[1]);
    float*       data      = outputs[0].GetTensorMutableData<float>();

    // Debug: dump first few raw values so we can verify the output format
    if (Debug::isEnabled() && numDets > 0) {
        fprintf(stdout, "[OnnxDetector] raw output[0]: cx=%.3f cy=%.3f w=%.3f h=%.3f obj=%.3f cls0=%.3f",
            data[0], data[1], data[2], data[3], data[4], data[5]);
        if (numClasses_ > 1)
            fprintf(stdout, " cls1=%.3f", data[6]);
        fprintf(stdout, "\n");
        fflush(stdout);
    }

    // Use EdgeYOLO's generate_yolo_proposals from platform/common.hpp
    // Input format: [cx, cy, w, h, obj_conf, cls0_score, cls1_score, ...]
    // Output format per proposal: [x1, y1, w, h, cls_idx, confidence]
    std::vector<std::vector<float>> proposals;
    generate_yolo_proposals(numDets, data, confThres_, proposals, numClasses_);

    DBG_LOG(TAG, "proposals before NMS: %zu / %d\n", proposals.size(), numDets);

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nmsThres_);

    DBG_LOG(TAG, "detections after NMS: %zu\n", picked.size());

    // Scale coordinates from letterboxed model space back to original frame space
    std::vector<Detection> detections;
    detections.reserve(picked.size());

    const float imgW = static_cast<float>(bgrFrame.cols);
    const float imgH = static_cast<float>(bgrFrame.rows);

    for (int idx : picked) {
        const auto& p = proposals[idx];
        // p = [x1, y1, w, h, cls_idx, confidence]  (letterboxed space)
        float x0 = p[0] * factor;
        float y0 = p[1] * factor;
        float x1 = (p[0] + p[2]) * factor;
        float y1 = (p[1] + p[3]) * factor;

        x0 = std::clamp(x0, 0.f, imgW - 1.f);
        y0 = std::clamp(y0, 0.f, imgH - 1.f);
        x1 = std::clamp(x1, 0.f, imgW - 1.f);
        y1 = std::clamp(y1, 0.f, imgH - 1.f);

        if (x1 <= x0 || y1 <= y0) continue;

        Detection d;
        d.rect       = { x0, y0, x1 - x0, y1 - y0 };
        d.classId    = static_cast<int>(p[4]);
        d.confidence = p[5];
        detections.push_back(d);
    }

    return detections;
}

} // namespace inference
