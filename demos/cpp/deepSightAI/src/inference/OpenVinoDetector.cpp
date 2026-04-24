/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#ifdef WITH_OPENVINO
#include "OpenVinoDetector.h"
#include "EdgeYoloPreProcessor.h"
#include "EdgeYoloPostProcessor.h"
#include "ModelMetaLoader.h"
#include "../debug_log.h"
#include <opencv2/imgproc.hpp>

namespace inference {

void OpenVinoDetector::dsai_load(const std::string& path, float c, float n) {
    confThres_ = c; nmsThres_ = n;

    // Load class labels and optional img_size override from sidecar YAML
    auto meta = dsai_loadModelMeta(yamlPath_, path);
    classNames_ = meta.classNames;
    numClasses_ = (int)classNames_.size();

    compiledModel_ = core_.compile_model(path, "AUTO");
    inferRequest_  = compiledModel_.create_infer_request();

    // Input shape from model (NCHW: [1, 3, H, W])
    auto inShape = compiledModel_.input().get_partial_shape();
    if (inShape.rank().is_static() && inShape.rank().get_length() == 4
        && inShape[2].is_static() && inShape[3].is_static()) {
        inputSize_ = cv::Size((int)inShape[3].get_length(), (int)inShape[2].get_length());
    }

    // Fall back to YAML img_size for dynamic/unknown dims
    if (inputSize_.width <= 0 || inputSize_.height <= 0) {
        if (meta.imgSizeOverride.width > 0)
            inputSize_ = meta.imgSizeOverride;
        else
            throw std::runtime_error("OpenVinoDetector: model has dynamic input shape "
                                     "and no img_size in YAML sidecar.");
    }

    // Derive numClasses from output shape [1, proposals, 5+C] if YAML had no labels
    if (numClasses_ == 0) {
        auto outShape = compiledModel_.output().get_partial_shape();
        if (outShape.rank().is_static() && outShape.rank().get_length() == 3
            && outShape[2].is_static() && outShape[2].get_length() > 5)
            numClasses_ = (int)outShape[2].get_length() - 5;
    }

    inputBlob_.assign(3 * inputSize_.width * inputSize_.height, 0.0f);
    preProcessor_  = std::make_unique<EdgeYoloPreProcessor>();
    postProcessor_ = std::make_unique<EdgeYoloPostProcessor>();
    loaded_ = true;

    DBG_LOG("OPENVINO", "input=%dx%d  classes=%d\n",
            inputSize_.width, inputSize_.height, numClasses_);
}

std::vector<Detection> OpenVinoDetector::dsai_infer(const cv::Mat& frame) {
    if (!loaded_) return {};
    PreProcessContext pctx;
    pctx.targetWidth = inputSize_.width; pctx.targetHeight = inputSize_.height;
    pctx.dstBuffer = inputBlob_.data(); pctx.dstSize = inputBlob_.size() * sizeof(float);
    pctx.outputCHW = true;
    preProcessor_->dsai_process(frame, pctx);

    ov::Tensor inputTensor(ov::element::f32,
        {1, 3, (size_t)inputSize_.height, (size_t)inputSize_.width}, inputBlob_.data());
    inferRequest_.set_input_tensor(inputTensor);
    inferRequest_.infer();

    const ov::Tensor& out = inferRequest_.get_output_tensor();
    PostProcessContext ctx;
    ctx.data         = out.data<const float>();
    ctx.numProposals = out.get_shape()[1];
    ctx.numClasses   = numClasses_;
    ctx.classNames   = classNames_;
    ctx.scaleX       = (float)frame.cols / inputSize_.width;
    ctx.scaleY       = (float)frame.rows / inputSize_.height;
    ctx.modelWidth   = inputSize_.width;
    ctx.modelHeight  = inputSize_.height;
    return postProcessor_->dsai_process(ctx, confThres_, nmsThres_);
}

} // namespace inference
#endif
