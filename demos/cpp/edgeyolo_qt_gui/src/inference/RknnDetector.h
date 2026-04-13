#ifndef RKNNDETECTOR_H
#define RKNNDETECTOR_H

#ifdef WITH_RKNN

#include "IDetector.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>

// Forward-declare RKNN::YOLO to avoid pulling in rknn.h / rknn_api.h in every TU
namespace RKNN { class YOLO; }

namespace inference {

/**
 * EdgeYOLO RKNN backend for Rockchip RV1106 (single NPU core).
 *
 * Wraps RKNN::YOLO from
 *   third_party/edgeyolo/cpp/rknn/include/image_utils/rknn.h
 *
 * The RKNN model uses a decoupled 9-output INT8 head; all dequantisation and
 * NMS is handled internally by RKNN::YOLO — this class only translates the
 * result into the common Detection type.
 *
 * Model path: .rknn file.
 * A sidecar .yaml (same basename) MUST exist alongside the .rknn file and
 * must contain:
 *   names: [classA, classB, …]
 *   img_size: [H, W]
 * (RKNN::YOLO reads it internally, but we also read it for classNames().)
 *
 * Guarded by WITH_RKNN compile flag.
 * Thread safety: NOT thread-safe.
 */
class RknnDetector : public IDetector {
public:
    explicit RknnDetector() = default;
    ~RknnDetector() override;

    // ── IDetector ─────────────────────────────────────────────────────────
    void load(const std::string& modelPath,
              float confThres = 0.25f,
              float nmsThres  = 0.45f) override;

    std::vector<Detection> infer(const cv::Mat& frame) override;

    const std::vector<std::string>& classNames() const override { return classNames_; }
    cv::Size inputSize() const override { return inputSize_; }
    bool     isLoaded()  const override { return loaded_; }

private:
    std::unique_ptr<RKNN::YOLO> yolo_;

    cv::Size                  inputSize_{ 416, 416 };
    std::vector<std::string>  classNames_;

    bool loaded_{ false };
};

} // namespace inference

#endif // WITH_RKNN
#endif // RKNNDETECTOR_H
