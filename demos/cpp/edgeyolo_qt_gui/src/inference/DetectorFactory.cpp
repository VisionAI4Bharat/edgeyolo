#include "DetectorFactory.h"
#include "OnnxDetector.h"

#ifdef WITH_OPENVINO
#  include "OpenVinoDetector.h"
#endif

#ifdef WITH_RKNN
#  include "RknnDetector.h"
#endif

namespace inference {

const char* DetectorFactory::name(Backend backend) noexcept
{
    switch (backend) {
        case Backend::ONNX:     return "ONNX Runtime";
        case Backend::OPENVINO: return "OpenVINO";
        case Backend::RKNN:     return "RKNN";
    }
    return "Unknown";
}

bool DetectorFactory::isAvailable(Backend backend) noexcept
{
    switch (backend) {
        case Backend::ONNX:
            return true;  // always compiled in
        case Backend::OPENVINO:
#ifdef WITH_OPENVINO
            return true;
#else
            return false;
#endif
        case Backend::RKNN:
#ifdef WITH_RKNN
            return true;
#else
            return false;
#endif
    }
    return false;
}

std::unique_ptr<IDetector> DetectorFactory::create(Backend            backend,
                                                    const std::string& modelPath,
                                                    const std::string& yamlPath,
                                                    float              confThres,
                                                    float              nmsThres)
{
    if (!isAvailable(backend))
        throw std::runtime_error(
            std::string("DetectorFactory: backend '") + name(backend) +
            "' was not compiled into this binary.");

    std::unique_ptr<IDetector> detector;

    switch (backend) {
        case Backend::ONNX: {
            auto d = std::make_unique<OnnxDetector>();
            if (!yamlPath.empty())
                d->setYamlPath(yamlPath);
            detector = std::move(d);
            break;
        }
#ifdef WITH_OPENVINO
        case Backend::OPENVINO: {
            auto d = std::make_unique<OpenVinoDetector>();
            if (!yamlPath.empty())
                d->setYamlPath(yamlPath);
            detector = std::move(d);
            break;
        }
#endif
#ifdef WITH_RKNN
        case Backend::RKNN: {
            // RKNN::YOLO reads its own sidecar YAML internally based on modelPath;
            // the yamlPath parameter is unused for RKNN (it must sit next to the .rknn).
            detector = std::make_unique<RknnDetector>();
            break;
        }
#endif
        default:
            throw std::runtime_error(
                std::string("DetectorFactory: unhandled backend '") + name(backend) + "'.");
    }

    // load() throws std::runtime_error on failure — propagate directly to caller.
    detector->load(modelPath, confThres, nmsThres);

    return detector;
}

} // namespace inference
