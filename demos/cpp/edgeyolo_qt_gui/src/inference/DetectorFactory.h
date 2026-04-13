#ifndef DETECTORFACTORY_H
#define DETECTORFACTORY_H

#include "IDetector.h"
#include <memory>
#include <string>
#include <stdexcept>

namespace inference {

enum class Backend {
    ONNX,
    OPENVINO,
    RKNN
};

/**
 * Creates and initialises the requested IDetector backend.
 *
 * Usage:
 *   auto det = DetectorFactory::create(Backend::ONNX, "/path/to/model.onnx", 0.25f, 0.45f);
 *   // det is ready — load() has already been called.
 *
 * Throws std::runtime_error if:
 *   - The requested backend was not compiled in.
 *   - load() fails for any reason.
 */
class DetectorFactory {
public:
    DetectorFactory() = delete;

    /**
     * @param backend     Which inference backend to use.
     * @param modelPath   Absolute path to the model file (.onnx / .rknn).
     * @param yamlPath    Path to the sidecar YAML (class names, img_size).
     *                    If empty, each backend looks for <modelBaseName>.yaml.
     * @param confThres   Confidence threshold.
     * @param nmsThres    NMS IoU threshold.
     * @return            Loaded, ready-to-use IDetector.
     * @throws std::runtime_error on any failure.
     */
    static std::unique_ptr<IDetector> create(Backend            backend,
                                             const std::string& modelPath,
                                             const std::string& yamlPath  = {},
                                             float              confThres = 0.25f,
                                             float              nmsThres  = 0.45f);

    /** Returns true if the backend was compiled into this binary. */
    static bool isAvailable(Backend backend) noexcept;

    /** Human-readable name for a backend enum value. */
    static const char* name(Backend backend) noexcept;
};

} // namespace inference

#endif // DETECTORFACTORY_H
