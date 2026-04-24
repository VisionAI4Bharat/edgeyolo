/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 *
 * Loads model metadata from a YAML sidecar file.
 * Used by all inference backends (detection, classification, or any future
 * task) to read class labels and optional spatial size overrides without
 * duplicating YAML parsing logic per-backend.
 */

#pragma once
#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <string>
#include <vector>

namespace inference {

struct ModelMeta {
    std::vector<std::string> classNames;
    cv::Size imgSizeOverride{0, 0};  // 0,0 = absent from YAML
};

// Parse the YAML sidecar for a model.
// If yamlPath is non-empty it is used directly; otherwise the loader looks
// for <modelPath-without-extension>.yaml next to the model file.
inline ModelMeta dsai_loadModelMeta(const std::string& yamlPath,
                                    const std::string& modelPath)
{
    namespace fs = std::filesystem;
    ModelMeta out;

    std::string path = yamlPath;
    if (path.empty()) {
        fs::path p(modelPath);
        path = (p.parent_path() / p.stem()).string() + ".yaml";
    }
    if (!fs::exists(path)) return out;

    YAML::Node cfg = YAML::LoadFile(path);

    // Accept both "class_labels" and "names" for backwards compatibility
    YAML::Node labels = cfg["class_labels"] ? cfg["class_labels"] : cfg["names"];
    if (labels && labels.IsSequence())
        out.classNames = labels.as<std::vector<std::string>>();

    // Optional spatial override for models exported with dynamic input dims.
    // YAML format:  img_size: [H, W]
    if (cfg["img_size"] && cfg["img_size"].IsSequence()) {
        auto sz = cfg["img_size"].as<std::vector<int>>();
        if (sz.size() == 2 && sz[0] > 0 && sz[1] > 0)
            out.imgSizeOverride = cv::Size(sz[1], sz[0]);  // [H,W] → cv::Size(W,H)
    }

    return out;
}

} // namespace inference
