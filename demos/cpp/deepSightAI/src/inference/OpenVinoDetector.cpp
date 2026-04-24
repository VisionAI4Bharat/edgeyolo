#ifdef WITH_OPENVINO
#include "OpenVinoDetector.h"
#include "EdgeYoloPreProcessor.h"
#include "EdgeYoloPostProcessor.h"
#include <opencv2/imgproc.hpp>
#include <filesystem>
namespace fs = std::filesystem;
namespace inference {
void OpenVinoDetector::dsai_load(const std::string& path, float c, float n) {
    confThres_ = c; nmsThres_ = n; loadYaml(path);
    compiledModel_ = core_.compile_model(path, "AUTO");
    inferRequest_  = compiledModel_.create_infer_request();
    auto inputShape = compiledModel_.input().get_partial_shape();
    if (inputShape.rank().is_static() && inputShape.rank().get_length() == 4) {
        inputSize_ = cv::Size((int)inputShape[3].get_length(), (int)inputShape[2].get_length());
    }
    DBG_LOG("OPENVINO", "Model input: %dx%d, Classes: %d\n", inputSize_.width, inputSize_.height, numClasses_);
    inputBlob_.assign(1 * 3 * inputSize_.width * inputSize_.height, 0.0f);
    preProcessor_  = std::make_unique<EdgeYoloPreProcessor>();
    postProcessor_ = std::make_unique<EdgeYoloPostProcessor>();
    loaded_ = true;
}
std::vector<Detection> OpenVinoDetector::dsai_infer(const cv::Mat& frame) {
    if (!loaded_) return {};
    PreProcessContext pctx;
    pctx.targetWidth = inputSize_.width; pctx.targetHeight = inputSize_.height;
    pctx.dstBuffer = inputBlob_.data(); pctx.dstSize = inputBlob_.size() * sizeof(float);
    pctx.outputCHW = true;
    preProcessor_->dsai_process(frame, pctx);

    ov::Tensor inputTensor(ov::element::f32, {1, 3, (size_t)inputSize_.height, (size_t)inputSize_.width}, inputBlob_.data());
    inferRequest_.set_input_tensor(inputTensor); inferRequest_.infer();
    
    const ov::Tensor& out = inferRequest_.get_output_tensor();
    PostProcessContext ctx;
    ctx.data = out.data<const float>();
    ctx.numProposals = out.get_shape()[1];
    ctx.numClasses = numClasses_; ctx.classNames = classNames_;
    ctx.scaleX = (float)frame.cols / inputSize_.width;
    ctx.scaleY = (float)frame.rows / inputSize_.height;
    ctx.modelWidth = inputSize_.width; ctx.modelHeight = inputSize_.height;
    return postProcessor_->dsai_process(ctx, confThres_, nmsThres_);
}
void OpenVinoDetector::loadYaml(const std::string& modelPath) {
    std::string yamlPath = yamlPath_;
    if (yamlPath.empty()) { fs::path p(modelPath); yamlPath = (p.parent_path() / p.stem()).string() + ".yaml"; }
    if (!fs::exists(yamlPath)) return;
    YAML::Node cfg = YAML::LoadFile(yamlPath);
    YAML::Node labelsNode = cfg["class_labels"] ? cfg["class_labels"] : cfg["names"];
    if (labelsNode) { classNames_ = labelsNode.as<std::vector<std::string>>(); numClasses_ = (int)classNames_.size(); }
}
}
#endif
