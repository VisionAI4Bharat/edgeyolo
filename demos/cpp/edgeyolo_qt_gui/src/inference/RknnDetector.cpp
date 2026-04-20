#ifdef WITH_RKNN

#include "RknnDetector.h"

// Pull in the full RKNN::YOLO definition only in this translation unit
#include "../../../../../third_party/edgeyolo/cpp/rknn/include/image_utils/rknn.h"

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace inference {

RknnDetector::~RknnDetector() = default;

void RknnDetector::load(const std::string& modelPath, float confThres, float nmsThres)
{
    if (modelPath.empty())
        throw std::runtime_error("RknnDetector: model path is empty");

    if (!fs::exists(modelPath))
        throw std::runtime_error("RknnDetector: model file not found: " + modelPath);

    // RKNN::YOLO reads the sidecar .yaml (same base name, .yaml extension) internally.
    // Verify it exists before handing off so we can give a clear error.
    fs::path p(modelPath);
    const std::string yamlPath = (p.parent_path() / p.stem()).string() + ".yaml";
    if (!fs::exists(yamlPath))
        throw std::runtime_error(
            "RknnDetector: sidecar YAML not found: " + yamlPath +
            "\n  The .yaml must sit next to the .rknn file with the same base name.");

    // Read class names and input size for our own classNames() / inputSize() API.
    try {
        YAML::Node cfg = YAML::LoadFile(yamlPath);

        YAML::Node labelsNode;
        if (cfg["class_labels"])      labelsNode = cfg["class_labels"];
        else if (cfg["names"])        labelsNode = cfg["names"];
        else throw std::runtime_error("RknnDetector: YAML missing 'class_labels' key: " + yamlPath);
        classNames_ = labelsNode.as<std::vector<std::string>>();

        if (classNames_.empty())
            throw std::runtime_error("RknnDetector: 'class_labels' list is empty in: " + yamlPath);

        if (cfg["img_size"]) {
            auto sz = cfg["img_size"].as<std::vector<int>>();
            if (sz.size() >= 2) {
                inputSize_.height = sz[0];
                inputSize_.width  = sz[1];
            }
        }
    }
    catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("RknnDetector: YAML parse error: ") + e.what());
    }

    // Construct and load the RKNN::YOLO instance.
    // Constructor also reads the sidecar YAML; confThres/nmsThres are passed here.
    yolo_ = std::make_unique<RKNN::YOLO>(modelPath, confThres, nmsThres);

    // RV1106 has a single NPU core — force CORE_0 to avoid driver errors.
    yolo_->set_running_core_mode(RKNN_NPU_CORE_0);

    const int ret = yolo_->load_model();
    if (ret < 0)
        throw std::runtime_error(
            "RknnDetector: rknn_init failed with error code " + std::to_string(ret) +
            " for model: " + modelPath);

    loaded_ = true;
}

std::vector<Detection> RknnDetector::infer(const cv::Mat& frame)
{
    if (!loaded_)
        throw std::runtime_error("RknnDetector: call load() before infer()");

    if (frame.empty())
        throw std::runtime_error("RknnDetector: infer() called with empty frame");

    cv::Mat bgrFrame;
    if (frame.channels() == 4)
        cv::cvtColor(frame, bgrFrame, cv::COLOR_BGRA2BGR);
    else if (frame.channels() == 1)
        cv::cvtColor(frame, bgrFrame, cv::COLOR_GRAY2BGR);
    else
        bgrFrame = frame;

    // RKNN::YOLO::infer handles letterbox, INT8 dequant, conf filter, and NMS internally.
    std::vector<detect::Object> objects;
    try {
        objects = yolo_->infer(bgrFrame);
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("RknnDetector: RKNN::YOLO::infer threw: ") + e.what());
    }

    // Translate detect::Object -> Detection
    std::vector<Detection> detections;
    detections.reserve(objects.size());

    for (const detect::Object& o : objects) {
        if (o.rect.width <= 0 || o.rect.height <= 0) continue;

        Detection d;
        d.rect       = o.rect;
        d.classId    = o.label;
        d.confidence = o.prob;
        detections.push_back(d);
    }

    return detections;
}

} // namespace inference

#endif // WITH_RKNN
