/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#ifdef WITH_RKNN
#include "RknnDetector.h"
#include "EdgeYoloPreProcessor.h"
#include "EdgeYoloPostProcessor.h"
#include <rknn_api.h>
#include <stdexcept>

namespace inference {

struct RknnDetector::Impl {
    rknn_context ctx=0; rknn_input_output_num ioNum={}; rknn_tensor_attr *inAt=nullptr, *outAt=nullptr;
    rknn_tensor_mem *inMe[1]={}, *outMe[1]={}; int mW=0, mH=0, nC=80; float cT=0.25f, nT=0.45f;
    bool l=false; std::vector<std::string> cN;
    void dsai_release() {
        if(!ctx) return; for(uint32_t i=0; i<ioNum.n_input; i++) if(inMe[i]) rknn_destroy_mem(ctx, inMe[i]);
        for(uint32_t i=0; i<ioNum.n_output; i++) if(outMe[i]) rknn_destroy_mem(ctx, outMe[i]);
        free(inAt); free(outAt); rknn_destroy(ctx); ctx=0; l=false;
    }
};

RknnDetector::RknnDetector() : pImpl_(std::make_unique<Impl>()) {}
RknnDetector::~RknnDetector() { pImpl_->dsai_release(); }

void RknnDetector::dsai_load(const std::string& p, float c, float n) {
    if(pImpl_->l) pImpl_->dsai_release(); pImpl_->cT=c; pImpl_->nT=n;
    FILE* f=fopen(p.c_str(),"rb"); fseek(f,0,SEEK_END); size_t s=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<unsigned char> d(s); fread(d.data(),1,s,f); fclose(f);
    if(rknn_init(&pImpl_->ctx, d.data(), s, 0, nullptr)<0) throw std::runtime_error("init fail");
    rknn_query(pImpl_->ctx, RKNN_QUERY_IN_OUT_NUM, &pImpl_->ioNum, sizeof(pImpl_->ioNum));
    pImpl_->inAt=(rknn_tensor_attr*)calloc(pImpl_->ioNum.n_input, sizeof(rknn_tensor_attr));
    pImpl_->outAt=(rknn_tensor_attr*)calloc(pImpl_->ioNum.n_output, sizeof(rknn_tensor_attr));
    for(uint32_t i=0; i<pImpl_->ioNum.n_input; i++) { pImpl_->inAt[i].index=i; rknn_query(pImpl_->ctx, RKNN_QUERY_INPUT_ATTR, &pImpl_->inAt[i], sizeof(rknn_tensor_attr)); }
    for(uint32_t i=0; i<pImpl_->ioNum.n_output; i++) { pImpl_->outAt[i].index=i; rknn_query(pImpl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &pImpl_->outAt[i], sizeof(rknn_tensor_attr)); }
    pImpl_->mW=pImpl_->inAt[0].dims[2]; pImpl_->mH=pImpl_->inAt[0].dims[1];
    for(uint32_t i=0; i<pImpl_->ioNum.n_input; i++) pImpl_->inMe[i]=rknn_create_mem(pImpl_->ctx, pImpl_->inAt[i].size_with_stride);
    for(uint32_t i=0; i<pImpl_->ioNum.n_output; i++) pImpl_->outMe[i]=rknn_create_mem(pImpl_->ctx, pImpl_->outAt[i].size_with_stride);
    rknn_set_io_mem(pImpl_->ctx, pImpl_->inMe[0], &pImpl_->inAt[0]);
    for(uint32_t i=0; i<pImpl_->ioNum.n_output; i++) rknn_set_io_mem(pImpl_->ctx, pImpl_->outMe[i], &pImpl_->outAt[i]);
    preProcessor_=std::make_unique<EdgeYoloPreProcessor>();
    postProcessor_=std::make_unique<EdgeYoloPostProcessor>();
    pImpl_->l=true;
}

std::vector<Detection> RknnDetector::dsai_infer(const cv::Mat& f) {
    if(!pImpl_->l) return {};
    PreProcessContext pc; pc.targetWidth=pImpl_->mW; pc.targetHeight=pImpl_->mH;
    pc.dstBuffer=pImpl_->inMe[0]->virt_addr; pc.dstSize=pImpl_->inAt[0].size;
    preProcessor_->dsai_process(f, pc);
    if(rknn_run(pImpl_->ctx, nullptr)<0) return {};
    PostProcessContext ctx; ctx.numClasses=pImpl_->nC; ctx.data=(const float*)pImpl_->outMe[0]->virt_addr;
    ctx.numProposals=pImpl_->outAt[0].dims[1];
    ctx.scaleX=(float)f.cols/pImpl_->mW; ctx.scaleY=(float)f.rows/pImpl_->mH;
    return postProcessor_->dsai_process(ctx, pImpl_->cT, pImpl_->nT);
}
const std::vector<std::string>& RknnDetector::dsai_classNames() const { return pImpl_->cN; }
void RknnDetector::dsai_setClassLabels(const std::vector<std::string>& l) { pImpl_->cN=l; pImpl_->nC=l.size(); }
cv::Size RknnDetector::dsai_inputSize() const { return {pImpl_->mW, pImpl_->mH}; }
bool RknnDetector::dsai_isLoaded() const { return pImpl_->l; }
void RknnDetector::dsai_setYamlPath(const std::string&) {}
}
#endif
