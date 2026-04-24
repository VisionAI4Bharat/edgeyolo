#include "OnnxDetector.h"
#include "EdgeYoloPreProcessor.h"
#include "EdgeYoloPostProcessor.h"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <thread>
namespace fs = std::filesystem;
namespace inference {
void OnnxDetector::dsai_load(const std::string& path, float c, float n) {
    confThres_ = c; nmsThres_ = n; loadYaml(path);
    Ort::SessionOptions so; so.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    session_ = std::make_unique<Ort::Session>(env_, path.c_str(), so);

    auto typeInfo = session_->GetInputTypeInfo(0);
    auto shape = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() == 4) {
        inputSize_ = cv::Size((int)shape[3], (int)shape[2]); // NCHW
    }

    Ort::AllocatorWithDefaultOptions alloc;
    inputName_  = std::string(session_->GetInputNameAllocated(0, alloc).get());
    outputName_ = std::string(session_->GetOutputNameAllocated(0, alloc).get());

    inputBlob_.assign(3 * inputSize_.width * inputSize_.height, 0.0f);
    DBG_LOG("ONNX", "Model input: %dx%d  classes: %d  in: '%s'  out: '%s'\n",
            inputSize_.width, inputSize_.height, numClasses_, inputName_.c_str(), outputName_.c_str());
    preProcessor_  = std::make_unique<EdgeYoloPreProcessor>();
    postProcessor_ = std::make_unique<EdgeYoloPostProcessor>();
    loaded_ = true;
}
std::vector<Detection> OnnxDetector::dsai_infer(const cv::Mat& frame) {
    if (!loaded_) return {};
    PreProcessContext pctx;
    pctx.targetWidth = inputSize_.width; pctx.targetHeight = inputSize_.height;
    pctx.dstBuffer = inputBlob_.data(); pctx.dstSize = inputBlob_.size() * sizeof(float);
    pctx.outputCHW = true;
    preProcessor_->dsai_process(frame, pctx);

    auto mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> ishape = {1, 3, (int64_t)inputSize_.height, (int64_t)inputSize_.width};
    Ort::Value itensor = Ort::Value::CreateTensor<float>(mi, inputBlob_.data(), inputBlob_.size(), ishape.data(), ishape.size());
    const char* inames[] = {inputName_.c_str()};
    const char* onames[] = {outputName_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, inames, &itensor, 1, onames, 1);
    PostProcessContext ctx;
    ctx.data = outputs[0].GetTensorData<float>();
    ctx.numProposals = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    ctx.numClasses = numClasses_; ctx.classNames = classNames_;
    ctx.scaleX = (float)frame.cols / inputSize_.width;
    ctx.scaleY = (float)frame.rows / inputSize_.height;
    ctx.modelWidth = inputSize_.width; ctx.modelHeight = inputSize_.height;
    return postProcessor_->dsai_process(ctx, confThres_, nmsThres_);
}
void OnnxDetector::loadYaml(const std::string& modelPath) {
    std::string yamlPath = yamlPath_;
    if (yamlPath.empty()) { fs::path p(modelPath); yamlPath = (p.parent_path() / p.stem()).string() + ".yaml"; }
    if (!fs::exists(yamlPath)) return;
    YAML::Node cfg = YAML::LoadFile(yamlPath);
    YAML::Node labelsNode = cfg["class_labels"] ? cfg["class_labels"] : cfg["names"];
    if (labelsNode) { classNames_ = labelsNode.as<std::vector<std::string>>(); numClasses_ = (int)classNames_.size(); }
}
}
