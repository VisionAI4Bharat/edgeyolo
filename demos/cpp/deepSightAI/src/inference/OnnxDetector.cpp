/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#include "OnnxDetector.h"
#include "EdgeYoloPostProcessor.h"
#include "../debug_log.h"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <thread>

namespace fs = std::filesystem;

namespace inference {

void OnnxDetector::dsai_load(const std::string& modelPath, float confThres, float nmsThres) {
    confThres_ = confThres; nmsThres_ = nmsThres;
    loadYaml(modelPath);
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions);
    postProcessor_ = std::make_unique<EdgeYoloPostProcessor>();
    loaded_ = true;
}

std::vector<Detection> OnnxDetector::dsai_infer(const cv::Mat& frame) {
    if (!loaded_) return {};
    cv::Mat res; cv::resize(frame, res, inputSize_);
    cv::cvtColor(res, res, cv::COLOR_BGR2RGB);

    std::vector<float> inputBlob(1*3*inputSize_.width*inputSize_.height);
    for(int c=0; c<3; c++)
        for(int h=0; h<inputSize_.height; h++)
            for(int w=0; w<inputSize_.width; w++)
                inputBlob[c*inputSize_.width*inputSize_.height + h*inputSize_.width + w] = res.at<cv::Vec3b>(h,w)[c] / 255.0f;

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> inputShape = {1, 3, (int64_t)inputSize_.height, (int64_t)inputSize_.width};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputBlob.data(), inputBlob.size(), inputShape.data(), inputShape.size());

    const char* inputNames[] = {"images"};
    // Adjust output names based on EdgeYOLO specific export
    const char* outputNames[] = {"dist_8", "dist_16", "dist_32", "obj_8", "obj_16", "obj_32", "cls_8", "cls_16", "cls_32"};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 9);

    PostProcessContext ctx;
    ctx.modelWidth = inputSize_.width; ctx.modelHeight = inputSize_.height;
    ctx.numClasses = numClasses_; ctx.classNames = classNames_;
    for(auto& o : outputs) ctx.outputFloats.push_back(o.GetTensorData<float>());
    return postProcessor_->dsai_process(ctx, confThres_, nmsThres_);
}

void OnnxDetector::loadYaml(const std::string& modelPath) {
    std::string yamlPath = yamlPath_;
    if (yamlPath.empty()) {
        fs::path p(modelPath);
        yamlPath = (p.parent_path() / p.stem()).string() + ".yaml";
    }
    if (!fs::exists(yamlPath)) return;
    YAML::Node cfg = YAML::LoadFile(yamlPath);
    YAML::Node labelsNode = cfg["class_labels"] ? cfg["class_labels"] : cfg["names"];
    if (labelsNode) {
        classNames_ = labelsNode.as<std::vector<std::string>>();
        numClasses_ = (int)classNames_.size();
    }
}

}
