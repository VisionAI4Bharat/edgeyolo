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

#ifndef IDETECTOR_H
#define IDETECTOR_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <stdexcept>

namespace inference {

/**
 * A single detection result in original image coordinates.
 */
struct Detection {
    cv::Rect_<float> rect;   // x, y, w, h in original image pixels
    int              classId;
    float            confidence;
};

/**
 * Pure abstract inference backend.
 *
 * Lifecycle:
 *   1. Construct the concrete subclass.
 *   2. Call load() — throws std::runtime_error on failure.
 *   3. Call infer() per frame — throws std::runtime_error on fatal backend error.
 *   4. Destroy (RAII cleanup).
 *
 * All implementations must be thread-safe for concurrent infer() calls IF
 * the underlying runtime supports it; otherwise document the restriction.
 */
class IDetector {
public:
    virtual ~IDetector() = default;

    /**
     * Load and initialise the model.
     * @param modelPath  Absolute path to the model file (.onnx / .rknn).
     * @param confThres  Objectness × class confidence threshold (0–1).
     * @param nmsThres   IoU threshold for NMS (0–1).
     * @throws std::runtime_error with a descriptive message on any failure.
     */
    virtual void load(const std::string& modelPath,
                      float confThres = 0.25f,
                      float nmsThres  = 0.45f) = 0;

    /**
     * Run inference on a single BGR frame.
     * The frame is letterbox-resized internally; no pre-processing needed by caller.
     * @param frame  BGR cv::Mat, any size.
     * @return       Detections in original frame coordinates.
     * @throws std::runtime_error on fatal backend error.
     */
    virtual std::vector<Detection> infer(const cv::Mat& frame) = 0;

    /**
     * Class labels in index order, populated after load().
     */
    virtual const std::vector<std::string>& classNames() const = 0;

    /**
     * Override class labels after load (e.g. from user-edited config).
     * Replaces the labels read from YAML; numClasses is updated accordingly.
     */
    virtual void setClassLabels(const std::vector<std::string>& labels) = 0;

    /**
     * Model input spatial size (width × height), populated after load().
     */
    virtual cv::Size inputSize() const = 0;

    /**
     * True after a successful load().
     */
    virtual bool isLoaded() const = 0;
};

} // namespace inference

#endif // IDETECTOR_H
