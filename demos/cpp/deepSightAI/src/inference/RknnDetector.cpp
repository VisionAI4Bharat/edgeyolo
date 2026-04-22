/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#ifdef WITH_RKNN
#include "RknnDetector.h"
#include "EdgeYoloPostProcessor.h"
#include <rknn_api.h>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#ifdef WITH_RGA
#include "im2d.h"
#include "rga.h"
#endif

namespace inference {

struct RknnDetector::Impl {
    rknn_context ctx = 0;
    rknn_input_output_num ioNum = {};
    rknn_tensor_attr* inputAttrs = nullptr;
    rknn_tensor_attr* outputAttrs = nullptr;
    rknn_tensor_mem* inputMems[1] = {};
    rknn_tensor_mem* outputMems[16] = {};
    int modelWidth = 0; int modelHeight = 0; int numClasses = 80;
    float confThres = 0.25f; float nmsThres = 0.45f;
    bool loaded = false; std::vector<std::string> classNames;
    void dsai_release() {
        if(!ctx) return;
        for(uint32_t i=0; i<ioNum.n_input; i++) if(inputMems[i]) rknn_destroy_mem(ctx, inputMems[i]);
        for(uint32_t i=0; i<ioNum.n_output; i++) if(outputMems[i]) rknn_destroy_mem(ctx, outputMems[i]);
        free(inputAttrs); free(outputAttrs); rknn_destroy(ctx); ctx=0; loaded=false;
    }
};

RknnDetector::RknnDetector() : pImpl_(std::make_unique<Impl>()) {}
RknnDetector::~RknnDetector() { pImpl_->dsai_release(); }

void RknnDetector::dsai_load(const std::string& path, float c, float n) {
    if(pImpl_->loaded) pImpl_->dsai_release();
    pImpl_->confThres = c; pImpl_->nmsThres = n;
    FILE* f = fopen(path.c_str(), "rb"); if(!f) throw std::runtime_error("Open fail");
    fseek(f,0,SEEK_END); size_t s=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<unsigned char> d(s); fread(data(d),1,s,f); fclose(f);
    if(rknn_init(&pImpl_->ctx, d.data(), s, 0, nullptr)<0) throw std::runtime_error("Init fail");
    rknn_query(pImpl_->ctx, RKNN_QUERY_IN_OUT_NUM, &pImpl_->ioNum, sizeof(pImpl_->ioNum));
    pImpl_->inputAttrs = (rknn_tensor_attr*)calloc(pImpl_->ioNum.n_input, sizeof(rknn_tensor_attr));
    pImpl_->outputAttrs = (rknn_tensor_attr*)calloc(pImpl_->ioNum.n_output, sizeof(rknn_tensor_attr));
    for(uint32_t i=0; i<pImpl_->ioNum.n_input; i++) {
        pImpl_->inputAttrs[i].index=i; rknn_query(pImpl_->ctx, RKNN_QUERY_INPUT_ATTR, &pImpl_->inputAttrs[i], sizeof(rknn_tensor_attr));
    }
    pImpl_->modelWidth = pImpl_->inputAttrs[0].dims[2]; pImpl_->modelHeight = pImpl_->inputAttrs[0].dims[1];
    for(uint32_t i=0; i<pImpl_->ioNum.n_output; i++) {
        pImpl_->outputAttrs[i].index=i; rknn_query(pImpl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &pImpl_->outputAttrs[i], sizeof(rknn_tensor_attr));
    }
    for(uint32_t i=0; i<pImpl_->ioNum.n_input; i++) pImpl_->inputMems[i]=rknn_create_mem(pImpl_->ctx, pImpl_->inputAttrs[i].size_with_stride);
    for(uint32_t i=0; i<pImpl_->ioNum.n_output; i++) pImpl_->outputMems[i]=rknn_create_mem(pImpl_->ctx, pImpl_->outputAttrs[i].size_with_stride);
    rknn_set_io_mem(pImpl_->ctx, pImpl_->inputMems[0], &pImpl_->inputAttrs[0]);
    for(uint32_t i=0; i<pImpl_->ioNum.n_output; i++) rknn_set_io_mem(pImpl_->ctx, pImpl_->outputMems[i], &pImpl_->outputAttrs[i]);
    postProcessor_ = std::make_unique<EdgeYoloPostProcessor>();
    pImpl_->loaded = true;
}

std::vector<Detection> RknnDetector::dsai_infer(const cv::Mat& f) {
    if(!pImpl_->loaded) return {};
    bool ok=false;
#ifdef WITH_RGA
    try {
        rga_buffer_t s = wrapbuffer_virtualaddr((void*)f.data, f.cols, f.rows, RK_FORMAT_BGR_888);
        rga_buffer_t d = wrapbuffer_virtualaddr(pImpl_->inputMems[0]->virt_addr, pImpl_->modelWidth, pImpl_->modelHeight, RK_FORMAT_RGB_888);
        if(imresize(s, d)==IM_STATUS_SUCCESS) ok=true;
    } catch(...) { ok=false; }
#endif
    if(!ok) {
        cv::Mat r; cv::resize(f, r, cv::Size(pImpl_->modelWidth, pImpl_->modelHeight));
        cv::cvtColor(r, r, cv::COLOR_BGR2RGB); memcpy(pImpl_->inputMems[0]->virt_addr, r.data, pImpl_->inputAttrs[0].size);
    }
    if(rknn_run(pImpl_->ctx, nullptr)<0) return {};
    PostProcessContext ctx; ctx.modelWidth=pImpl_->modelWidth; ctx.modelHeight=pImpl_->modelHeight;
    ctx.numClasses=pImpl_->numClasses; ctx.classNames=pImpl_->classNames;
    for(uint32_t i=0; i<pImpl_->ioNum.n_output; i++) {
        ctx.outputInt8s.push_back((int8_t*)pImpl_->outputMems[i]->virt_addr);
        ctx.outputScales.push_back(pImpl_->outputAttrs[i].scale); ctx.outputZps.push_back(pImpl_->outputAttrs[i].zp);
    }
    return postProcessor_->dsai_process(ctx, pImpl_->confThres, pImpl_->nmsThres);
}

const std::vector<std::string>& RknnDetector::dsai_classNames() const { return pImpl_->classNames; }
void RknnDetector::dsai_setClassLabels(const std::vector<std::string>& l) { pImpl_->classNames=l; pImpl_->numClasses=l.size(); }
cv::Size RknnDetector::dsai_inputSize() const { return {pImpl_->modelWidth, pImpl_->modelHeight}; }
bool RknnDetector::dsai_isLoaded() const { return pImpl_->loaded; }
void RknnDetector::dsai_setYamlPath(const std::string&) {}

}
#endif
