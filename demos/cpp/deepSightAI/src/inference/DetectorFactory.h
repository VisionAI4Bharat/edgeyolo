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
 *   auto det = DetectorFactory::dsai_create(Backend::ONNX, "/path/to/model.onnx", 0.25f, 0.45f);
 *   // det is ready — dsai_load() has already been called.
 *
 * Throws std::runtime_error if:
 *   - The requested backend was not compiled in.
 *   - dsai_load() fails for any reason.
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
    static std::unique_ptr<IDetector> dsai_create(Backend            backend,
                                             const std::string& modelPath,
                                             const std::string& yamlPath  = {},
                                             float              confThres = 0.25f,
                                             float              nmsThres  = 0.45f);

    /** Returns true if the backend was compiled into this binary. */
    static bool dsai_isAvailable(Backend backend) noexcept;

    /** Human-readable name for a backend enum value. */
    static const char* dsai_name(Backend backend) noexcept;

    /**
     * Validates that modelPath's extension is compatible with backend.
     * Throws std::runtime_error with a descriptive message if not.
     * Called internally by dsai_create; also available for pre-flight checks.
     */
    static void dsai_validateModelExtension(Backend backend, const std::string& modelPath);
};

} // namespace inference

#endif // DETECTORFACTORY_H
