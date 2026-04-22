/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once
#include "IPostProcessor.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace inference {

class EdgeYoloPostProcessor : public IPostProcessor {
public:
    std::vector<Detection> dsai_process(const PostProcessContext& ctx, float confThr, float nmsThr) override {
        std::vector<Detection> candidates;
        const int strides[3] = {8, 16, 32};
        bool isInt8 = !ctx.outputInt8s.empty();
        
        // Final safety check
        if (isInt8) {
            if (ctx.outputInt8s.size() < 9) return {};
        } else {
            if (ctx.outputFloats.size() < 9) return {};
        }

        for (int s = 0; s < 3; ++s) {
            int gridW = ctx.modelWidth / strides[s];
            int gridH = ctx.modelHeight / strides[s];
            int gridSize = gridW * gridH;

            for (int i = 0; i < gridSize; i++) {
                float score = 0;
                if (isInt8) {
                    float oSc = ctx.outputScales[s+3]; int oZp = ctx.outputZps[s+3];
                    float rawObj = (ctx.outputInt8s[s+3][i] - oZp) * oSc;
                    score = 1.0f / (1.0f + expf(-rawObj));
                } else {
                    score = 1.0f / (1.0f + expf(-ctx.outputFloats[s+3][i]));
                }
                if (score < confThr) continue;

                int bestCls = 0; float bestClsScore = -1e9f;
                for (int c = 0; c < ctx.numClasses; c++) {
                    float val = 0;
                    if (isInt8) {
                        float cSc = ctx.outputScales[s+6]; int cZp = ctx.outputZps[s+6];
                        val = (ctx.outputInt8s[s+6][i * ctx.numClasses + c] - cZp) * cSc;
                    } else {
                        val = ctx.outputFloats[s+6][i * ctx.numClasses + c];
                    }
                    if (val > bestClsScore) { bestClsScore = val; bestCls = c; }
                }
                float finalScore = score * (1.0f / (1.0f + expf(-bestClsScore)));
                if (finalScore < confThr) continue;

                float l, t, r, b;
                if (isInt8) {
                    float dSc = ctx.outputScales[s]; int dZp = ctx.outputZps[s];
                    l = (ctx.outputInt8s[s][i*4+0] - dZp) * dSc;
                    t = (ctx.outputInt8s[s][i*4+1] - dZp) * dSc;
                    r = (ctx.outputInt8s[s][i*4+2] - dZp) * dSc;
                    b = (ctx.outputInt8s[s][i*4+3] - dZp) * dSc;
                } else {
                    l = ctx.outputFloats[s][i*4+0]; t = ctx.outputFloats[s][i*4+1];
                    r = ctx.outputFloats[s][i*4+2]; b = ctx.outputFloats[s][i*4+3];
                }

                float cx = ((i % gridW) + 0.5f) * strides[s];
                float cy = ((i / gridW) + 0.5f) * strides[s];

                Detection d;
                d.rect.x = cx - l * strides[s]; d.rect.y = cy - t * strides[s];
                d.rect.width = (l + r) * strides[s]; d.rect.height = (t + b) * strides[s];
                d.confidence = finalScore; d.classId = bestCls;
                candidates.push_back(d);
            }
        }

        if (candidates.empty()) return {};
        std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

        const int numCandidates = static_cast<int>(candidates.size());
        std::vector<float> areas(numCandidates);
        for (int i = 0; i < numCandidates; ++i) areas[i] = candidates[i].rect.width * candidates[i].rect.height;

        std::vector<Detection> result;
        std::vector<bool> suppressed(numCandidates, false);
        for (int i = 0; i < numCandidates; ++i) {
            if (suppressed[i]) continue;
            result.push_back(candidates[i]);
            const auto& r1 = candidates[i].rect;
            const float a1 = areas[i];
            for (int j = i + 1; j < numCandidates; ++j) {
                if (suppressed[j]) continue;
                const auto& r2 = candidates[j].rect;
                float xx1 = std::max(r1.x, r2.x);
                float yy1 = std::max(r1.y, r2.y);
                float xx2 = std::min(r1.x + r1.width,  r2.x + r2.width);
                float yy2 = std::min(r1.y + r1.height, r2.y + r2.height);
                float w = std::max(0.0f, xx2 - xx1);
                float h = std::max(0.0f, yy2 - yy1);
                float inter = w * h;
                if (inter > 0) {
                    float iou = inter / (a1 + areas[j] - inter);
                    if (iou > nmsThr) suppressed[j] = true;
                }
            }
        }
        return result;
    }
};

}
