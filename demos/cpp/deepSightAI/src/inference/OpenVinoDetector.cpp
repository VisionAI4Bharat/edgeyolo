/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#ifdef WITH_OPENVINO
#include "OpenVinoDetector.h"
#include "EdgeYoloPostProcessor.h"
#include <opencv2/imgproc.hpp>
#include <filesystem>

namespace fs = std::filesystem;

namespace inference {

void OpenVinoDetector::dsai_load(const std::string& modelPath, float confThres, float nmsThres) {
    confThres_ = confThres; nmsThres_ = nmsThres;
    loadYaml(modelPath);
    compiledModel_ = core_.compile_model(modelPath, "AUTO");
    inferRequest_  = compiledModel_.create_infer_request();
    postProcessor_ = std::make_unique<EdgeYoloPostProcessor>();
    loaded_ = true;
}

std::vector<Detection> OpenVinoDetector::dsai_infer(const cv::Mat& frame) {
    if (!loaded_) return {};
    cv::Mat res; cv::resize(frame, res, inputSize_);
    cv::cvtColor(res, res, cv::COLOR_BGR2RGB);

    float* input_data = (float*)malloc(1*3*inputSize_.width*inputSize_.height*sizeof(float));
    for(int c=0; c<3; c++)
        for(int h=0; h<inputSize_.height; h++)
            for(int w=0; w<inputSize_.width; w++)
                input_data[c*inputSize_.width*inputSize_.height + h*inputSize_.width + w] = res.at<cv::Vec3b>(h,w)[c] / 255.0f;

    ov::Tensor inputTensor(ov::element::f32, {1, 3, (size_t)inputSize_.height, (size_t)inputSize_.width}, input_data);
    inferRequest_.set_input_tensor(inputTensor);
    inferRequest_.infer();
    free(input_data);

    PostProcessContext ctx;
    ctx.modelWidth = inputSize_.width; ctx.modelHeight = inputSize_.height;
    ctx.numClasses = numClasses_; ctx.classNames = classNames_;
    for(size_t i=0; i<compiledModel_.outputs().size(); i++) {
        const ov::Tensor& out = inferRequest_.get_output_tensor(i);
        ctx.outputFloats.push_back(out.data<const float>());
    }
    return postProcessor_->dsai_process(ctx, confThres_, nmsThres_);
}

void OpenVinoDetector::loadYaml(const std::string& modelPath) {
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
#endif
