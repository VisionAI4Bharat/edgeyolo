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

#include "DetectorFactory.h"

#ifdef WITH_ONNXRT
#  include "OnnxDetector.h"
#endif

#ifdef WITH_OPENVINO
#  include "OpenVinoDetector.h"
#endif

#ifdef WITH_RKNN
#  include "RknnDetector.h"
#endif

namespace inference {

const char* DetectorFactory::dsai_name(Backend backend) noexcept
{
    switch (backend) {
        case Backend::ONNX:     return "ONNX Runtime";
        case Backend::OPENVINO: return "OpenVINO";
        case Backend::RKNN:     return "RKNN";
    }
    return "Unknown";
}

bool DetectorFactory::dsai_isAvailable(Backend backend) noexcept
{
    switch (backend) {
        case Backend::ONNX:
#ifdef WITH_ONNXRT
            return true;
#else
            return false;
#endif
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

std::unique_ptr<IDetector> DetectorFactory::dsai_create(Backend            backend,
                                                    const std::string& modelPath,
                                                    const std::string& yamlPath,
                                                    float              confThres,
                                                    float              nmsThres)
{
    if (!dsai_isAvailable(backend))
        throw std::runtime_error(
            std::string("DetectorFactory: backend '") + dsai_name(backend) +
            "' was not compiled into this binary.");

    std::unique_ptr<IDetector> detector;

    switch (backend) {
#ifdef WITH_ONNXRT
        case Backend::ONNX: {
            auto d = std::make_unique<OnnxDetector>();
            if (!yamlPath.empty())
                d->dsai_setYamlPath(yamlPath);
            detector = std::move(d);
            break;
        }
#endif
#ifdef WITH_OPENVINO
        case Backend::OPENVINO: {
            auto d = std::make_unique<OpenVinoDetector>();
            if (!yamlPath.empty())
                d->dsai_setYamlPath(yamlPath);
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
                std::string("DetectorFactory: unhandled backend '") + dsai_name(backend) + "'.");
    }

    // dsai_load() throws std::runtime_error on failure — propagate directly to caller.
    detector->dsai_load(modelPath, confThres, nmsThres);

    return detector;
}

} // namespace inference
