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
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    auto shape = tensorInfo.GetShape();
    if (shape.size() == 4) {
        // NCHW or NHWC check (EdgeYOLO is NCHW)
        inputSize_ = cv::Size((int)shape[3], (int)shape[2]);
    }

    DBG_LOG("ONNX", "Model input: %dx%d, Classes: %d\n", inputSize_.width, inputSize_.height, numClasses_);
    preProcessor_  = std::make_unique<EdgeYoloPreProcessor>();
    postProcessor_ = std::make_unique<EdgeYoloPostProcessor>();
    loaded_ = true;
}
std::vector<Detection> OnnxDetector::dsai_infer(const cv::Mat& frame) {
    if (!loaded_) return {};
    cv::Mat hwcBGR(inputSize_, CV_8UC3);
    PreProcessContext pctx; pctx.targetWidth=inputSize_.width; pctx.targetHeight=inputSize_.height;
    pctx.dstBuffer=hwcBGR.data; pctx.dstSize=hwcBGR.total()*hwcBGR.elemSize();
    preProcessor_->dsai_process(frame, pctx);
    std::vector<float> inputBlob(1*3*inputSize_.width*inputSize_.height);
    const int plane = inputSize_.width * inputSize_.height;
    for(int h=0; h<inputSize_.height; h++) {
        for(int w=0; w<inputSize_.width; w++) {
            cv::Vec3b pix = hwcBGR.at<cv::Vec3b>(h,w);
            inputBlob[0*plane + h*inputSize_.width + w] = (float)pix[0]; // B
            inputBlob[1*plane + h*inputSize_.width + w] = (float)pix[1]; // G
            inputBlob[2*plane + h*inputSize_.width + w] = (float)pix[2]; // R
        }
    }
    auto mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> ishape = {1, 3, (int64_t)inputSize_.height, (int64_t)inputSize_.width};
    Ort::Value itensor = Ort::Value::CreateTensor<float>(mi, inputBlob.data(), inputBlob.size(), ishape.data(), ishape.size());
    const char* inames[] = {"images"}; const char* onames[] = {"output"}; // Assuming fused head name is 'output'
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
