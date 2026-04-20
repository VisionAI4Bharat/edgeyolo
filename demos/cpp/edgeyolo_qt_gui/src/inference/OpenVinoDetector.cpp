#ifdef WITH_OPENVINO

#include "OpenVinoDetector.h"

#include <opencv2/imgproc.hpp>

#include "edgeyolo_bridge.h"

#include <filesystem>
#include <stdexcept>
#include <algorithm>

namespace fs = std::filesystem;

namespace inference {

namespace {

std::vector<float> mat2blob(const cv::Mat& letterboxed, int H, int W)
{
    std::vector<float> blob(1 * 3 * H * W);
    const int plane = H * W;
    for (int r = 0; r < H; ++r) {
        const uchar* row = letterboxed.ptr<uchar>(r);
        for (int c = 0; c < W; ++c) {
            blob[0 * plane + r * W + c] = static_cast<float>(row[c * 3 + 2]); // R
            blob[1 * plane + r * W + c] = static_cast<float>(row[c * 3 + 1]); // G
            blob[2 * plane + r * W + c] = static_cast<float>(row[c * 3 + 0]); // B
        }
    }
    return blob;
}

} // anonymous namespace

// ─── load ────────────────────────────────────────────────────────────────────

void OpenVinoDetector::load(const std::string& modelPath, float confThres, float nmsThres)
{
    if (modelPath.empty())
        throw std::runtime_error("OpenVinoDetector: model path is empty");

    if (!fs::exists(modelPath))
        throw std::runtime_error("OpenVinoDetector: model file not found: " + modelPath);

    confThres_ = confThres;
    nmsThres_  = nmsThres;

    loadYaml(modelPath);

    try {
        compiledModel_ = core_.compile_model(modelPath, "AUTO");
        inferRequest_  = compiledModel_.create_infer_request();
    }
    catch (const ov::Exception& e) {
        throw std::runtime_error(
            std::string("OpenVinoDetector: failed to compile model: ") + e.what());
    }

    // Resolve spatial size from compiled model's input shape if dynamic axes are free
    auto inputTensorShape = compiledModel_.input().get_partial_shape();
    if (inputTensorShape.rank().is_static() && inputTensorShape.rank().get_length() == 4) {
        auto h = inputTensorShape[2];
        auto w = inputTensorShape[3];
        if (h.is_static() && static_cast<int>(h.get_length()) > 0)
            inputSize_.height = static_cast<int>(h.get_length());
        if (w.is_static() && static_cast<int>(w.get_length()) > 0)
            inputSize_.width = static_cast<int>(w.get_length());
    }

    // Validate output shape [B, N, L]
    auto outShape = compiledModel_.output().get_partial_shape();
    if (outShape.rank().is_static() && outShape.rank().get_length() == 3) {
        auto L = outShape[2];
        if (L.is_static()) {
            const int expectedL = 5 + numClasses_;
            if (static_cast<int>(L.get_length()) != expectedL)
                throw std::runtime_error(
                    "OpenVinoDetector: output last-dim=" +
                    std::to_string(L.get_length()) +
                    " but expected 5+numClasses=" + std::to_string(expectedL));
        }
    }

    loaded_ = true;
}

void OpenVinoDetector::loadYaml(const std::string& modelPath)
{
    std::string yamlPath = yamlPath_;
    if (yamlPath.empty()) {
        fs::path p(modelPath);
        yamlPath = (p.parent_path() / p.stem()).string() + ".yaml";
    }

    if (!fs::exists(yamlPath))
        throw std::runtime_error(
            "OpenVinoDetector: sidecar YAML not found: " + yamlPath +
            "\n  Call setYamlPath() to specify its location.");

    try {
        YAML::Node cfg = YAML::LoadFile(yamlPath);
        YAML::Node labelsNode;
        if (cfg["class_labels"])      labelsNode = cfg["class_labels"];
        else if (cfg["names"])        labelsNode = cfg["names"];
        else throw std::runtime_error("OpenVinoDetector: YAML missing 'class_labels' key: " + yamlPath);

        classNames_ = labelsNode.as<std::vector<std::string>>();
        numClasses_ = static_cast<int>(classNames_.size());

        if (numClasses_ == 0)
            throw std::runtime_error("OpenVinoDetector: 'class_labels' list is empty in: " + yamlPath);

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
            std::string("OpenVinoDetector: YAML parse error: ") + e.what());
    }
}

// ─── infer ───────────────────────────────────────────────────────────────────

std::vector<Detection> OpenVinoDetector::infer(const cv::Mat& frame)
{
    if (!loaded_)
        throw std::runtime_error("OpenVinoDetector: call load() before infer()");

    if (frame.empty())
        throw std::runtime_error("OpenVinoDetector: infer() called with empty frame");

    cv::Mat bgrFrame;
    if (frame.channels() == 4)
        cv::cvtColor(frame, bgrFrame, cv::COLOR_BGRA2BGR);
    else if (frame.channels() == 1)
        cv::cvtColor(frame, bgrFrame, cv::COLOR_GRAY2BGR);
    else
        bgrFrame = frame;

    const int H = inputSize_.height;
    const int W = inputSize_.width;

    detect::resizeInfo rzInfo = detect::resizeAndPad(bgrFrame, cv::Size(W, H), false, false);
    const float factor = rzInfo.factor;

    std::vector<float> blob = mat2blob(rzInfo.resized_img, H, W);

    // Set input tensor
    ov::Tensor inputTensor(ov::element::f32, ov::Shape{1, 3,
        static_cast<size_t>(H), static_cast<size_t>(W)}, blob.data());
    inferRequest_.set_input_tensor(inputTensor);

    try {
        inferRequest_.infer();
    }
    catch (const ov::Exception& e) {
        throw std::runtime_error(
            std::string("OpenVinoDetector: infer request failed: ") + e.what());
    }

    const ov::Tensor& outputTensor = inferRequest_.get_output_tensor();
    const auto outShape = outputTensor.get_shape(); // [1, N, L]

    if (outShape.size() != 3)
        throw std::runtime_error("OpenVinoDetector: unexpected output rank");

    const size_t numDets  = outShape[1];
    const size_t arrayLen = outShape[2];
    const float* data     = outputTensor.data<const float>();

    return postProcess(data, numDets, arrayLen, factor, bgrFrame.size());
}

// ─── postProcess ─────────────────────────────────────────────────────────────

std::vector<Detection> OpenVinoDetector::postProcess(
    const float* data,
    size_t       numDets,
    size_t       arrayLen,
    float        factor,
    cv::Size     oriSize) const
{
    std::vector<detect::Object> proposals;
    proposals.reserve(numDets / 4);

    for (size_t i = 0; i < numDets; ++i) {
        const float* det = data + i * arrayLen;
        const float objConf = det[4];
        if (objConf < confThres_) continue;

        int   bestClass = 0;
        float bestScore = det[5];
        for (int c = 1; c < numClasses_; ++c) {
            if (det[5 + c] > bestScore) {
                bestScore = det[5 + c];
                bestClass = c;
            }
        }

        const float confidence = objConf * bestScore;
        if (confidence < confThres_) continue;

        detect::Object obj;
        obj.label       = bestClass;
        obj.prob        = confidence;
        obj.rect.width  = det[2];
        obj.rect.height = det[3];
        obj.rect.x      = det[0] - obj.rect.width  * 0.5f;
        obj.rect.y      = det[1] - obj.rect.height * 0.5f;
        proposals.push_back(obj);
    }

    detect::qsort_descent_inplace(proposals);
    std::vector<int> picked;
    detect::nms_sorted_bboxes(proposals, picked, nmsThres_);

    std::vector<Detection> detections;
    detections.reserve(picked.size());

    for (int idx : picked) {
        const detect::Object& o = proposals[idx];
        float x0 = std::clamp(o.rect.x                      * factor, 0.f, (float)(oriSize.width  - 1));
        float y0 = std::clamp(o.rect.y                      * factor, 0.f, (float)(oriSize.height - 1));
        float x1 = std::clamp((o.rect.x + o.rect.width)     * factor, 0.f, (float)(oriSize.width  - 1));
        float y1 = std::clamp((o.rect.y + o.rect.height)    * factor, 0.f, (float)(oriSize.height - 1));

        if (x1 <= x0 || y1 <= y0) continue;

        Detection d;
        d.rect       = { x0, y0, x1 - x0, y1 - y0 };
        d.classId    = o.label;
        d.confidence = o.prob;
        detections.push_back(d);
    }

    return detections;
}

} // namespace inference

#endif // WITH_OPENVINO
