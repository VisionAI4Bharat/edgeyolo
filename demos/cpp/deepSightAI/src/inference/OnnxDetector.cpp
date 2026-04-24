#include "OnnxDetector.h"
#include "EdgeYoloPreProcessor.h"
#include "EdgeYoloPostProcessor.h"
#include "ModelMetaLoader.h"
#include "../debug_log.h"
#include <opencv2/imgproc.hpp>
#include <thread>
namespace inference {

void OnnxDetector::dsai_load(const std::string& path, float c, float n) {
    confThres_ = c; nmsThres_ = n;

    // Load class labels and optional img_size override from sidecar YAML
    auto meta = dsai_loadModelMeta(yamlPath_, path);
    classNames_ = meta.classNames;
    numClasses_ = (int)classNames_.size();

    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    session_ = std::make_unique<Ort::Session>(env_, path.c_str(), so);

    // Input shape from model (NCHW)
    auto inShape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (inShape.size() == 4 && inShape[2] > 0 && inShape[3] > 0)
        inputSize_ = cv::Size((int)inShape[3], (int)inShape[2]);

    // Fall back to YAML img_size for dynamic/unknown dims
    if (inputSize_.width <= 0 || inputSize_.height <= 0) {
        if (meta.imgSizeOverride.width > 0)
            inputSize_ = meta.imgSizeOverride;
        else
            throw std::runtime_error("OnnxDetector: model has dynamic input shape "
                                     "and no img_size in YAML sidecar.");
    }

    // Derive numClasses from output shape [1, proposals, 5+C] if YAML had no labels
    if (numClasses_ == 0) {
        auto outShape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (outShape.size() == 3 && outShape[2] > 5)
            numClasses_ = (int)outShape[2] - 5;
    }

    // Input/output tensor names — read from model, never hardcoded
    Ort::AllocatorWithDefaultOptions alloc;
    inputName_  = std::string(session_->GetInputNameAllocated(0, alloc).get());
    outputName_ = std::string(session_->GetOutputNameAllocated(0, alloc).get());

    inputBlob_.assign(3 * inputSize_.width * inputSize_.height, 0.0f);
    preProcessor_  = std::make_unique<EdgeYoloPreProcessor>();
    postProcessor_ = std::make_unique<EdgeYoloPostProcessor>();
    loaded_ = true;

    DBG_LOG("ONNX", "input=%dx%d  classes=%d  in='%s'  out='%s'\n",
            inputSize_.width, inputSize_.height, numClasses_,
            inputName_.c_str(), outputName_.c_str());
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
    Ort::Value itensor = Ort::Value::CreateTensor<float>(
        mi, inputBlob_.data(), inputBlob_.size(), ishape.data(), ishape.size());
    const char* inames[] = {inputName_.c_str()};
    const char* onames[] = {outputName_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, inames, &itensor, 1, onames, 1);

    PostProcessContext ctx;
    ctx.data         = outputs[0].GetTensorData<float>();
    ctx.numProposals = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    ctx.numClasses   = numClasses_;
    ctx.classNames   = classNames_;
    ctx.scaleX       = (float)frame.cols / inputSize_.width;
    ctx.scaleY       = (float)frame.rows / inputSize_.height;
    ctx.modelWidth   = inputSize_.width;
    ctx.modelHeight  = inputSize_.height;
    return postProcessor_->dsai_process(ctx, confThres_, nmsThres_);
}

} // namespace inference
