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

#ifndef RKNNDETECTOR_H
#define RKNNDETECTOR_H

#ifdef WITH_RKNN

#include "IDetector.h"
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace inference {

/**
 * EdgeYOLO RKNN inference backend — direct RKNN API, zero-copy memory.
 *
 * Implementation strategy
 * =======================
 * The Rockchip RV1106 NPU exposes a zero-copy DMA memory API.  At load time
 * we allocate DMA-backed rknn_tensor_mem buffers for both input and all
 * outputs via rknn_create_mem() + rknn_set_io_mem().  At inference time the
 * letterboxed frame is memcpy'd once into the pre-mapped input buffer
 * (input_mems[0]->virt_addr) and the NPU reads it directly — no per-frame
 * malloc, no double copy.
 *
 * EdgeYOLO decoupled head (9 outputs)
 * ====================================
 * EdgeYOLO exports 9 INT8 tensors per inference pass:
 *   [0..2]  dist(s)  — per-grid box distances dl,dt,dr,db  (4 channels)
 *   [3..5]  obj(s)   — per-grid objectness score            (1 channel)
 *   [6..8]  cls(s)   — per-grid class logits                (C channels)
 * for s in {stride-8, stride-16, stride-32}.
 * Grid sizes are read from output tensor attrs at load time, so the decoder
 * is agnostic to model input resolution (416, 640, …).
 *
 * Preprocessing
 * =============
 * Input type is set to RKNN_TENSOR_UINT8 NHWC so the NPU fuses quantisation
 * internally — raw uint8 BGR pixels are fed directly, no CPU normalisation.
 * Letterbox resize (grey pad 114) preserves aspect ratio; coordinates are
 * remapped to original frame space after NMS.
 *
 * Sidecar YAML (must sit next to the .rknn file, same base name):
 *   class_labels: [classA, classB, …]
 *   img_size: [H, W]       # optional — overridden by queried tensor attrs
 *
 * Thread safety: one instance per thread.
 */
class RknnDetector : public IDetector {
public:
    RknnDetector();
    ~RknnDetector() override;

    RknnDetector(const RknnDetector&)            = delete;
    RknnDetector& operator=(const RknnDetector&) = delete;

    // ── IDetector ─────────────────────────────────────────────────────────
    void dsai_load(const std::string& modelPath,
              float confThres = 0.25f,
              float nmsThres  = 0.45f) override;

    std::vector<Detection> dsai_infer(const cv::Mat& bgrFrame) override;

    const std::vector<std::string>& dsai_classNames() const override;
    void dsai_setClassLabels(const std::vector<std::string>& labels) override;
    cv::Size dsai_inputSize() const override;
    bool     dsai_isLoaded()  const override;

    // Opaque implementation — defined only in RknnDetector.cpp.
    // Public so static helpers in .cpp can name it (strict GCC 8.x access rules).
    struct Impl;

private:
    std::unique_ptr<Impl> pImpl_;
};

}  // namespace inference

#endif  // WITH_RKNN
#endif  // RKNNDETECTOR_H
