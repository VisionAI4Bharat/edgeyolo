/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 *
 * This software is dual-licensed:
 * 1. GNU General Public License v3.0 (GPLv3)
 * 2. A proprietary license for commercial use.
 *
 * You may use this software under the terms of the GPLv3 if you are using it
 * for non-commercial purposes. For commercial usage, a separate commercial 
 * license must be obtained from swatah.ai (info@swatah.ai).
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
 * for more details.
 *
 * Trademarks: All trademarks, service marks, and logos are the property of 
 * their respective owners.
 */

#ifdef WITH_RKNN

#include "RknnDetector.h"

// Direct RKNN runtime API — no intermediate wrapper
#include <rknn_api.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace inference {

// ─── constants ────────────────────────────────────────────────────────────────

// EdgeYOLO decoupled head: 3 strides × 3 tensor types (dist, obj, cls)
static constexpr int kNumScales  = 3;
static constexpr int kNumOutputs = 9;    // must match model export
static constexpr int kStrides[3] = {8, 16, 32};

// ─── private implementation struct ────────────────────────────────────────────

struct RknnDetector::Impl {
    rknn_context          ctx          = 0;
    rknn_input_output_num ioNum        = {};
    rknn_tensor_attr*     inputAttrs   = nullptr;
    rknn_tensor_attr*     outputAttrs  = nullptr;

    // Zero-copy DMA-backed tensor memory — allocated once at load, reused every frame
    rknn_tensor_mem* inputMems [1]           = {};
    rknn_tensor_mem* outputMems[kNumOutputs] = {};

    int   modelWidth  = 416;
    int   modelHeight = 416;
    int   numClasses  = 80;
    int   grids[kNumScales] = {};   // grid size per stride, e.g. 52/26/13 for 416

    float confThres = 0.25f;
    float nmsThres  = 0.45f;
    bool  loaded    = false;

    std::vector<std::string> classNames;

    void release();
};

// ─── Impl::release ────────────────────────────────────────────────────────────

void RknnDetector::Impl::release()
{
    if (!ctx) return;

    for (uint32_t i = 0; i < ioNum.n_input; ++i) {
        if (inputMems[i]) {
            rknn_destroy_mem(ctx, inputMems[i]);
            inputMems[i] = nullptr;
        }
    }
    for (int i = 0; i < kNumOutputs; ++i) {
        if (outputMems[i]) {
            rknn_destroy_mem(ctx, outputMems[i]);
            outputMems[i] = nullptr;
        }
    }

    free(inputAttrs);   inputAttrs  = nullptr;
    free(outputAttrs);  outputAttrs = nullptr;

    rknn_destroy(ctx);
    ctx    = 0;
    loaded = false;
    printf("[RknnDetector] released\n");
}

// ─── letterbox helper ─────────────────────────────────────────────────────────

struct LBInfo { float scale; int padL; int padT; };

static LBInfo letterbox(const cv::Mat& src, cv::Mat& dst, int W, int H)
{
    const float sx = static_cast<float>(W) / src.cols;
    const float sy = static_cast<float>(H) / src.rows;
    const float s  = std::min(sx, sy);

    const int nw = static_cast<int>(std::round(src.cols * s));
    const int nh = static_cast<int>(std::round(src.rows * s));
    const int pl = (W - nw) / 2;
    const int pt = (H - nh) / 2;

    cv::Mat resized;
    cv::resize(src, resized, {nw, nh}, 0, 0, cv::INTER_LINEAR);

    dst.create(H, W, CV_8UC3);
    dst.setTo(cv::Scalar(114, 114, 114));   // grey padding (RKNN export default)
    resized.copyTo(dst(cv::Rect(pl, pt, nw, nh)));

    return {s, pl, pt};
}

// ─── NMS ──────────────────────────────────────────────────────────────────────

struct Candidate {
    cv::Rect2f box;
    float      score;
    int        cls;
};

static float iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width,  b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float w  = std::max(0.f, x2 - x1);
    const float h  = std::max(0.f, y2 - y1);
    const float i  = w * h;
    return i / (a.width * a.height + b.width * b.height - i + 1e-6f);
}

static std::vector<Candidate> nmsClassWise(std::vector<Candidate>& v, float thr)
{
    std::sort(v.begin(), v.end(),
              [](const Candidate& a, const Candidate& b){ return a.score > b.score; });

    std::vector<bool> dropped(v.size(), false);
    std::vector<Candidate> out;
    out.reserve(v.size());

    for (size_t i = 0; i < v.size(); ++i) {
        if (dropped[i]) continue;
        out.push_back(v[i]);
        for (size_t j = i + 1; j < v.size(); ++j) {
            if (!dropped[j] && v[i].cls == v[j].cls &&
                iou(v[i].box, v[j].box) > thr)
                dropped[j] = true;
        }
    }
    return out;
}

// ─── EdgeYOLO 9-tensor post-processor ────────────────────────────────────────
//
// Layout (RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, NHWC):
//   output[0..2] : dist — shape [1, g, g, 4]     dl/dt/dr/db in model pixels
//   output[3..5] : obj  — shape [1, g, g, 1]     raw logit
//   output[6..8] : cls  — shape [1, g, g, C]     raw class logits
//
// All tensors are INT8 with per-tensor (scale, zp) from rknn_tensor_attr.
// Sigmoid is applied to obj and cls; no sigmoid on dist (it's a regression).

static std::vector<Candidate> decodeHead(const RknnDetector::Impl& p, float confThr)
{
    const rknn_tensor_attr* a = p.outputAttrs;
    rknn_tensor_mem* const* m = p.outputMems;

    std::vector<Candidate> cands;
    cands.reserve(256);

    for (int s = 0; s < kNumScales; ++s) {
        const int   g   = p.grids[s];
        const int   st  = kStrides[s];
        const int   gSq = g * g;

        // Tensor data pointers (INT8)
        const auto* distQ = static_cast<const int8_t*>(m[s + 0]->virt_addr);
        const auto* objQ  = static_cast<const int8_t*>(m[s + 3]->virt_addr);
        const auto* clsQ  = static_cast<const int8_t*>(m[s + 6]->virt_addr);

        // Per-tensor dequantisation params
        const float dSc = a[s + 0].scale;  const int dZp = a[s + 0].zp;
        const float oSc = a[s + 3].scale;  const int oZp = a[s + 3].zp;
        const float cSc = a[s + 6].scale;  const int cZp = a[s + 6].zp;

        for (int gy = 0; gy < g; ++gy)
        for (int gx = 0; gx < g; ++gx)
        {
            const int idx = gy * g + gx;

            // Objectness — early reject before scanning all C classes
            const float rawObj = (static_cast<int>(objQ[idx]) - oZp) * oSc;
            const float obj    = 1.f / (1.f + std::exp(-rawObj));
            if (obj < confThr * 0.5f) continue;

            // Best class
            const int8_t* qc = &clsQ[idx * p.numClasses];
            int   bestCls  = 0;
            float bestRaw  = -1e9f;
            for (int c = 0; c < p.numClasses; ++c) {
                const float v = (static_cast<int>(qc[c]) - cZp) * cSc;
                if (v > bestRaw) { bestRaw = v; bestCls = c; }
            }
            const float clsP = 1.f / (1.f + std::exp(-bestRaw));
            const float conf = obj * clsP;
            if (conf < confThr) continue;

            // Box decode: centre (cx, cy) ± distance offsets in model-pixel space
            const int8_t* q4 = &distQ[idx * 4];
            const float dl = (static_cast<int>(q4[0]) - dZp) * dSc;
            const float dt = (static_cast<int>(q4[1]) - dZp) * dSc;
            const float dr = (static_cast<int>(q4[2]) - dZp) * dSc;
            const float db = (static_cast<int>(q4[3]) - dZp) * dSc;

            const float cx = (gx + 0.5f) * st;
            const float cy = (gy + 0.5f) * st;

            const float mW = static_cast<float>(p.modelWidth  - 1);
            const float mH = static_cast<float>(p.modelHeight - 1);
            const float x1 = std::max(0.f, cx - dl);
            const float y1 = std::max(0.f, cy - dt);
            const float x2 = std::min(mW,  cx + dr);
            const float y2 = std::min(mH,  cy + db);
            if (x2 <= x1 || y2 <= y1) continue;

            cands.push_back({cv::Rect2f{x1, y1, x2 - x1, y2 - y1}, conf, bestCls});
        }
        (void)gSq;  // suppress unused warning when asserts disabled
    }
    return cands;
}

// ─── RknnDetector public API ──────────────────────────────────────────────────

RknnDetector::RknnDetector()
    : pImpl_(std::make_unique<Impl>()) {}

RknnDetector::~RknnDetector()
{
    if (pImpl_) pImpl_->release();
}

void RknnDetector::load(const std::string& modelPath, float confThres, float nmsThres)
{
    if (pImpl_->loaded) pImpl_->release();   // allow reload

    auto& p = *pImpl_;
    p.confThres = confThres;
    p.nmsThres  = nmsThres;

    // ── validate paths ────────────────────────────────────────────────────
    if (modelPath.empty())
        throw std::runtime_error("RknnDetector: model path is empty");
    if (!fs::exists(modelPath))
        throw std::runtime_error("RknnDetector: model not found: " + modelPath);

    const fs::path mp(modelPath);
    const std::string yamlPath = (mp.parent_path() / mp.stem()).string() + ".yaml";
    if (!fs::exists(yamlPath))
        throw std::runtime_error(
            "RknnDetector: sidecar YAML not found: " + yamlPath +
            "\n  Place a .yaml with the same base name next to the .rknn file.");

    // ── parse sidecar YAML ────────────────────────────────────────────────
    try {
        const YAML::Node cfg = YAML::LoadFile(yamlPath);

        YAML::Node lbl;
        if      (cfg["class_labels"]) lbl = cfg["class_labels"];
        else if (cfg["names"])        lbl = cfg["names"];
        else throw std::runtime_error("YAML missing 'class_labels' key");

        p.classNames = lbl.as<std::vector<std::string>>();
        if (p.classNames.empty())
            throw std::runtime_error("'class_labels' list is empty");

        if (cfg["img_size"]) {
            const auto sz = cfg["img_size"].as<std::vector<int>>();
            if (sz.size() >= 2) { p.modelHeight = sz[0]; p.modelWidth = sz[1]; }
        }
    }
    catch (const YAML::Exception& e) {
        throw std::runtime_error(
            "RknnDetector: YAML parse error in '" + yamlPath + "': " +
            std::string(e.what()));
    }

    // ── rknn_init ─────────────────────────────────────────────────────────
    int ret = rknn_init(&p.ctx, (char*)modelPath.c_str(), 0, 0, nullptr);
    if (ret < 0)
        throw std::runtime_error(
            "RknnDetector: rknn_init failed (" + std::to_string(ret) +
            ") for: " + modelPath);

    // Log SDK / driver versions for diagnostics
    {
        rknn_sdk_version ver{};
        if (rknn_query(p.ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == RKNN_SUCC)
            printf("[RknnDetector] SDK %s  driver %s\n",
                   ver.api_version, ver.drv_version);
    }

    // RV1106 has a single NPU core — rknn_set_core_mask is absent from
    // librknnmrt. On multi-core targets (RK3588) this call pins inference to
    // core 0 for deterministic latency; enable it by linking against the
    // appropriate librknnrt.so that exports the symbol.

    // ── query I/O counts ──────────────────────────────────────────────────
    ret = rknn_query(p.ctx, RKNN_QUERY_IN_OUT_NUM, &p.ioNum, sizeof(p.ioNum));
    if (ret != RKNN_SUCC)
        throw std::runtime_error("RknnDetector: rknn_query IN_OUT_NUM failed");

    if (p.ioNum.n_output != kNumOutputs)
        throw std::runtime_error(
            "RknnDetector: expected " + std::to_string(kNumOutputs) +
            " outputs (EdgeYOLO 3-scale decoupled head), got " +
            std::to_string(p.ioNum.n_output) +
            ".  Verify the model is an EdgeYOLO INT8 export.");

    // ── input tensor attrs ────────────────────────────────────────────────
    p.inputAttrs = static_cast<rknn_tensor_attr*>(
        calloc(p.ioNum.n_input, sizeof(rknn_tensor_attr)));
    for (uint32_t i = 0; i < p.ioNum.n_input; ++i) {
        p.inputAttrs[i].index = i;
        ret = rknn_query(p.ctx, RKNN_QUERY_NATIVE_INPUT_ATTR,
                         &p.inputAttrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
            throw std::runtime_error("RknnDetector: input attr query failed");
    }

    // ── output tensor attrs (native NHWC for zero-copy) ───────────────────
    p.outputAttrs = static_cast<rknn_tensor_attr*>(
        calloc(kNumOutputs, sizeof(rknn_tensor_attr)));
    for (int i = 0; i < kNumOutputs; ++i) {
        p.outputAttrs[i].index = i;
        ret = rknn_query(p.ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR,
                         &p.outputAttrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
            throw std::runtime_error("RknnDetector: output attr query failed i=" +
                                     std::to_string(i));
    }

    // ── derive model dims from input attrs ────────────────────────────────
    const auto& a0 = p.inputAttrs[0];
    if (a0.fmt == RKNN_TENSOR_NHWC) {
        // dims: [N, H, W, C]
        p.modelHeight = static_cast<int>(a0.dims[1]);
        p.modelWidth  = static_cast<int>(a0.dims[2]);
    } else {
        // NCHW: [N, C, H, W]
        p.modelHeight = static_cast<int>(a0.dims[2]);
        p.modelWidth  = static_cast<int>(a0.dims[3]);
    }

    // ── derive grid sizes from dist tensor dims ───────────────────────────
    // dist tensors are output[0..2], NHWC shape [1, g, g, 4]
    for (int s = 0; s < kNumScales; ++s)
        p.grids[s] = static_cast<int>(p.outputAttrs[s].dims[1]);

    // ── derive numClasses from cls tensor — authoritative over YAML count ──
    // cls tensors are output[6..8], NHWC shape [1, g, g, C]
    const int modelClasses = static_cast<int>(p.outputAttrs[6].dims[3]);
    if (modelClasses != static_cast<int>(p.classNames.size()))
        fprintf(stderr,
                "[RknnDetector] WARN: model has %d classes but YAML lists %d labels. "
                "Using model class count; labels may be misaligned.\n",
                modelClasses,
                static_cast<int>(p.classNames.size()));
    p.numClasses = modelClasses;

    // ── allocate zero-copy input buffer ───────────────────────────────────
    // Set UINT8 NHWC so the NPU fuses quantisation internally.
    // memcpy(input_mems[0]->virt_addr, …) at inference time is the only copy.
    p.inputAttrs[0].type = RKNN_TENSOR_UINT8;
    p.inputAttrs[0].fmt  = RKNN_TENSOR_NHWC;

    p.inputMems[0] = rknn_create_mem(p.ctx, p.inputAttrs[0].size_with_stride);
    if (!p.inputMems[0])
        throw std::runtime_error("RknnDetector: rknn_create_mem input failed");

    ret = rknn_set_io_mem(p.ctx, p.inputMems[0], &p.inputAttrs[0]);
    if (ret < 0)
        throw std::runtime_error(
            "RknnDetector: rknn_set_io_mem input failed: " + std::to_string(ret));

    // ── allocate zero-copy output buffers ─────────────────────────────────
    for (int i = 0; i < kNumOutputs; ++i) {
        p.outputMems[i] = rknn_create_mem(p.ctx, p.outputAttrs[i].size_with_stride);
        if (!p.outputMems[i])
            throw std::runtime_error(
                "RknnDetector: rknn_create_mem output " + std::to_string(i));

        ret = rknn_set_io_mem(p.ctx, p.outputMems[i], &p.outputAttrs[i]);
        if (ret < 0)
            throw std::runtime_error(
                "RknnDetector: rknn_set_io_mem output " + std::to_string(i) +
                " failed: " + std::to_string(ret));
    }

    printf("[RknnDetector] ready  model=%dx%d  classes=%d  "
           "grids=%d/%d/%d  conf=%.2f  nms=%.2f\n",
           p.modelWidth, p.modelHeight, p.numClasses,
           p.grids[0], p.grids[1], p.grids[2],
           confThres, nmsThres);

    p.loaded = true;
}

std::vector<Detection> RknnDetector::infer(const cv::Mat& bgrFrame)
{
    if (!pImpl_->loaded)
        throw std::runtime_error("RknnDetector: call load() before infer()");
    if (bgrFrame.empty())
        throw std::runtime_error("RknnDetector: empty frame passed to infer()");

    auto& p = *pImpl_;

    // ── ensure 3-channel BGR ──────────────────────────────────────────────
    cv::Mat bgr;
    if (bgrFrame.channels() == 4)
        cv::cvtColor(bgrFrame, bgr, cv::COLOR_BGRA2BGR);
    else if (bgrFrame.channels() == 1)
        cv::cvtColor(bgrFrame, bgr, cv::COLOR_GRAY2BGR);
    else
        bgr = bgrFrame;

    // ── letterbox resize → model input size ──────────────────────────────
    cv::Mat lb;
    const LBInfo lbi = letterbox(bgr, lb, p.modelWidth, p.modelHeight);

    // ── BGR → RGB ─────────────────────────────────────────────────────────
    // EdgeYOLO was trained with RGB input (PyTorch/torchvision convention).
    // export_rknn.py does not set reorder_channel in rknn.config(), so the
    // NPU passes channels through as-is and expects RGB byte order.
    cv::Mat rgb;
    cv::cvtColor(lb, rgb, cv::COLOR_BGR2RGB);

    // ── zero-copy write into NPU DMA buffer ───────────────────────────────
    // Normalisation (÷255) is fused into the NPU via UINT8 input type +
    // mean=0 / std=255 baked at export — no CPU divide needed.
    std::memcpy(p.inputMems[0]->virt_addr, rgb.data,
                static_cast<size_t>(p.modelWidth * p.modelHeight * 3));

    // ── run NPU inference ─────────────────────────────────────────────────
    const int ret = rknn_run(p.ctx, nullptr);
    if (ret < 0)
        throw std::runtime_error("RknnDetector: rknn_run failed: " + std::to_string(ret));

    // ── decode 9-tensor EdgeYOLO head ─────────────────────────────────────
    auto cands = decodeHead(p, p.confThres);

    // ── class-wise NMS ────────────────────────────────────────────────────
    auto kept = nmsClassWise(cands, p.nmsThres);

    // ── map from letterbox space back to original frame coordinates ───────
    const float inv  = 1.f / lbi.scale;
    const float maxW = static_cast<float>(bgr.cols);
    const float maxH = static_cast<float>(bgr.rows);

    std::vector<Detection> result;
    result.reserve(kept.size());

    for (const auto& k : kept) {
        const float x1 = std::clamp((k.box.x - lbi.padL) * inv, 0.f, maxW - 1.f);
        const float y1 = std::clamp((k.box.y - lbi.padT) * inv, 0.f, maxH - 1.f);
        const float x2 = std::clamp((k.box.x + k.box.width  - lbi.padL) * inv,
                                    0.f, maxW - 1.f);
        const float y2 = std::clamp((k.box.y + k.box.height - lbi.padT) * inv,
                                    0.f, maxH - 1.f);
        if (x2 <= x1 || y2 <= y1) continue;

        Detection d;
        d.rect       = cv::Rect{static_cast<int>(x1), static_cast<int>(y1),
                                static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)};
        d.classId    = k.cls;
        d.confidence = k.score;
        result.push_back(d);
    }

    return result;
}

const std::vector<std::string>& RknnDetector::classNames() const
{
    return pImpl_->classNames;
}

void RknnDetector::setClassLabels(const std::vector<std::string>& labels)
{
    pImpl_->classNames = labels;
    // numClasses stays at the model's actual output count — labels are display only
}

cv::Size RknnDetector::inputSize() const
{
    return {pImpl_->modelWidth, pImpl_->modelHeight};
}

bool RknnDetector::isLoaded() const
{
    return pImpl_->loaded;
}

}  // namespace inference

#endif  // WITH_RKNN
