/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once
#include "IPostProcessor.h"
#include "../debug_log.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace inference {

class EdgeYoloPostProcessor : public IPostProcessor {
public:
    std::vector<Detection> dsai_process(const PostProcessContext& ctx, float confThr, float nmsThr) override {
        std::vector<Detection> candidates;
        const int stride = 5 + ctx.numClasses;

        for (size_t i = 0; i < ctx.numProposals; i++) {
            const float* det = ctx.data + i * stride;
            float objConf = det[4];
            if (objConf < confThr) continue;

            float maxClsConf = 0; int bestCls = 0;
            for (int c = 0; c < ctx.numClasses; c++) {
                if (det[5 + c] > maxClsConf) { maxClsConf = det[5 + c]; bestCls = c; }
            }
            float finalConf = objConf * maxClsConf;
            if (finalConf < confThr) continue;

            Detection d;
            d.confidence = finalConf; d.classId = bestCls;
            float w = det[2]; float h = det[3];
            float x = det[0] - w * 0.5f; float y = det[1] - h * 0.5f;
            
            d.rect.x = x * ctx.scaleX; d.rect.y = y * ctx.scaleY;
            d.rect.width = w * ctx.scaleX; d.rect.height = h * ctx.scaleY;
            candidates.push_back(d);
        }

        if (candidates.empty()) return {};
        std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
        
        std::vector<float> areas(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) areas[i] = candidates[i].rect.width * candidates[i].rect.height;
        
        std::vector<Detection> result;
        std::vector<bool> suppressed(candidates.size(), false);
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (suppressed[i]) continue;
            result.push_back(candidates[i]);
            for (size_t j = i + 1; j < candidates.size(); ++j) {
                if (suppressed[j]) continue;
                float xx1 = std::max(candidates[i].rect.x, candidates[j].rect.x);
                float yy1 = std::max(candidates[i].rect.y, candidates[j].rect.y);
                float xx2 = std::min(candidates[i].rect.x + candidates[i].rect.width, candidates[j].rect.x + candidates[j].rect.width);
                float yy2 = std::min(candidates[i].rect.y + candidates[i].rect.height, candidates[j].rect.y + candidates[j].rect.height);
                float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
                if (inter > 0 && (inter / (areas[i] + areas[j] - inter)) > nmsThr) suppressed[j] = true;
            }
        }
        return result;
    }
};

}
