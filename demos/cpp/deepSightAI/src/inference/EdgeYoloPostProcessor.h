/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once
#include "IPostProcessor.h"
#include <cmath>
#include <algorithm>

namespace inference {

class EdgeYoloPostProcessor : public IPostProcessor {
public:
    std::vector<Detection> dsai_process(const PostProcessContext& ctx, float confThr, float nmsThr) override {
        std::vector<Detection> candidates;
        const int strides[3] = {8, 16, 32};
        bool isInt8 = !ctx.outputInt8s.empty();

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

        std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

        std::vector<Detection> result;
        std::vector<bool> suppressed(candidates.size(), false);
        for (size_t i = 0; i < candidates.size(); i++) {
            if (suppressed[i]) continue;
            result.push_back(candidates[i]);
            for (size_t j = i + 1; j < candidates.size(); j++) {
                if (suppressed[j]) continue;
                float inter = (candidates[i].rect & candidates[j].rect).area();
                float iou = inter / (candidates[i].rect.area() + candidates[j].rect.area() - inter + 1e-6f);
                if (iou > nmsThr) suppressed[j] = true;
            }
        }
        return result;
    }
};

}
