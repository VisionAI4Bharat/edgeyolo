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

#include "app_config.h"
#include "debug_log.h"

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
namespace fs = std::filesystem;

// ─── comment-preserving YAML patch ───────────────────────────────────────────
//
// Walks the original file text line by line and replaces only the value side
// of key:value lines using a YAML::Node as the source of truth. Comments,
// blank lines, and unknown keys are copied verbatim. Keys absent from the
// original are appended at the end. Completely key-name agnostic — if new
// fields are added to dsai_saveToFile they are automatically handled.

static std::string scalarStr(const YAML::Node& n) {
    YAML::Emitter em; em << n; return em.c_str();
}

// Replace the value portion of a YAML scalar line, preserving key and any
// trailing inline comment (e.g. "  backend: 1  # 0=ONNX").
static std::string patchValue(const std::string& line, const std::string& newVal) {
    size_t colon = line.find(':');
    if (colon == std::string::npos) return line;
    size_t vs = colon + 1;
    while (vs < line.size() && line[vs] == ' ') vs++;
    std::string prefix = line.substr(0, vs);
    bool inQ = false;
    size_t cpos = std::string::npos;
    for (size_t i = vs; i < line.size(); i++) {
        if (line[i] == '"') inQ = !inQ;
        if (!inQ && line[i] == '#' && i > 0 && line[i-1] == ' ') { cpos = i; break; }
    }
    return cpos != std::string::npos
        ? prefix + newVal + "  " + line.substr(cpos)
        : prefix + newVal;
}

static std::string patchYamlText(const std::string& orig, const YAML::Node& desired) {
    auto keyOf = [](const std::string& ln) -> std::string {
        size_t s = ln.find_first_not_of(' ');
        if (s == std::string::npos || ln[s] == '#' || ln[s] == '-') return {};
        size_t c = ln.find(':', s);
        if (c == std::string::npos) return {};
        std::string k = ln.substr(s, c - s);
        while (!k.empty() && k.back() == ' ') k.pop_back();
        return k;
    };
    auto indentOf = [](const std::string& ln) {
        int i = 0; while (i < (int)ln.size() && ln[i] == ' ') i++; return i;
    };
    auto blankOrComment = [](const std::string& ln) {
        size_t s = ln.find_first_not_of(' ');
        return s == std::string::npos || ln[s] == '#';
    };
    auto isSeqItem = [](const std::string& ln) {
        size_t s = ln.find_first_not_of(' ');
        return s != std::string::npos && ln[s] == '-';
    };
    auto emitSeq = [&](std::ostringstream& out, const std::string& k) {
        out << k << ":\n";
        for (const auto& item : desired[k]) out << "  - " << scalarStr(item) << "\n";
    };

    std::set<std::string> seen;
    enum class St { Normal, InMap, SkipSeq };
    St state = St::Normal;
    std::string seqKey, mapKey;
    std::ostringstream out;
    std::istringstream iss(orig);
    std::string line;

    while (std::getline(iss, line)) {
        if (state == St::SkipSeq) {
            if (isSeqItem(line)) continue;
            emitSeq(out, seqKey);
            state = St::Normal;
        }

        if (state == St::Normal) {
            if (blankOrComment(line)) { out << line << "\n"; continue; }
            std::string k = keyOf(line);
            if (k.empty()) { out << line << "\n"; continue; }
            if (indentOf(line) == 0) {
                if (!desired[k]) {
                    out << line << "\n";
                } else if (desired[k].IsScalar()) {
                    seen.insert(k);
                    out << patchValue(line, scalarStr(desired[k])) << "\n";
                } else if (desired[k].IsSequence()) {
                    seen.insert(k); seqKey = k; state = St::SkipSeq;
                } else if (desired[k].IsMap()) {
                    seen.insert(k); mapKey = k; state = St::InMap;
                    out << line << "\n";
                } else {
                    out << line << "\n";
                }
            } else {
                out << line << "\n";
            }
        } else if (state == St::InMap) {
            if (blankOrComment(line)) { out << line << "\n"; continue; }
            std::string k = keyOf(line);
            if (k.empty()) { out << line << "\n"; continue; }
            if (indentOf(line) == 0) {
                state = St::Normal;
                if (!desired[k]) {
                    out << line << "\n";
                } else if (desired[k].IsScalar()) {
                    seen.insert(k);
                    out << patchValue(line, scalarStr(desired[k])) << "\n";
                } else if (desired[k].IsSequence()) {
                    seen.insert(k); seqKey = k; state = St::SkipSeq;
                } else if (desired[k].IsMap()) {
                    seen.insert(k); mapKey = k; state = St::InMap;
                    out << line << "\n";
                } else {
                    out << line << "\n";
                }
            } else {
                if (desired[mapKey] && desired[mapKey][k]) {
                    out << patchValue(line, scalarStr(desired[mapKey][k])) << "\n";
                } else {
                    out << line << "\n";
                }
            }
        }
    }
    if (state == St::SkipSeq) emitSeq(out, seqKey);

    // Append top-level keys from desired not found in the original
    for (auto it = desired.begin(); it != desired.end(); ++it) {
        std::string k = it->first.as<std::string>();
        if (seen.count(k)) continue;
        if (it->second.IsScalar()) {
            out << k << ": " << scalarStr(it->second) << "\n";
        } else if (it->second.IsSequence()) {
            emitSeq(out, k);
        } else if (it->second.IsMap()) {
            out << k << ":\n";
            for (auto jt = it->second.begin(); jt != it->second.end(); ++jt)
                out << "  " << jt->first.as<std::string>() << ": " << scalarStr(jt->second) << "\n";
        }
    }
    return out.str();
}


std::string AppConfig::dsai_defaultPath() {
#ifdef __arm__
    if (fs::exists("/etc/deepSightAI/config.yaml"))
        return "/etc/deepSightAI/config.yaml";
#endif
    const char* xdg = std::getenv("XDG_CONFIG_HOME");
    std::string base = xdg ? xdg : (std::string(std::getenv("HOME") ? std::getenv("HOME") : "/root") + "/.config");
    std::string dir = base + "/deepSightAI";
    if (!fs::exists(dir)) fs::create_directories(dir);
    return dir + "/config.yaml";
}

AppConfig AppConfig::dsai_loadFromFile(const std::string& path) {
    AppConfig cfg;
    YAML::Node n;
    try {
        n = YAML::LoadFile(path);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("AppConfig: cannot read '") + path + "': " + e.what());
    }

    auto get = [&](const char* k, auto& out) {
        if (n[k]) out = n[k].as<std::decay_t<decltype(out)>>();
    };

    int backendInt  = static_cast<int>(cfg.backend);
    int sourceInt   = static_cast<int>(cfg.source);
    get("backend",          backendInt);
    get("source",           sourceInt);
    cfg.backend = static_cast<Backend>(backendInt);
    cfg.source  = static_cast<SourceType>(sourceInt);

    get("model_file",       cfg.modelFile);
    get("yaml_file",        cfg.yamlFile);
    get("conf_threshold",   cfg.confThreshold);
    get("nms_threshold",    cfg.nmsThreshold);
    get("camera_device_id", cfg.cameraDeviceId);
    get("video_file",       cfg.videoFile);
    get("rtsp_url",         cfg.rtspUrl);
    get("iq_dir",           cfg.iqDir);
    get("capture_width",    cfg.captureWidth);
    get("capture_height",   cfg.captureHeight);
    get("capture_fps",      cfg.captureFps);
    get("gain",             cfg.gain);
    get("gamma",            cfg.gamma);
    get("brightness",       cfg.brightness);
    get("rockchip_hw",      cfg.rockchipHw);
    get("roi_enabled",      cfg.roiEnabled);
    get("web_port",         cfg.webPort);
    get("debug_logging",    cfg.debugLogging);
    get("rtsp_port",         cfg.rtspPort);

    if (n["roi"]) {
        YAML::Node r = n["roi"];
        if (r["x"])      cfg.roi.x      = r["x"].as<int>();
        if (r["y"])      cfg.roi.y      = r["y"].as<int>();
        if (r["width"])  cfg.roi.width  = r["width"].as<int>();
        if (r["height"]) cfg.roi.height = r["height"].as<int>();
    }

    YAML::Node labelsNode = n["class_labels"] ? n["class_labels"] : n["names"];
    if (labelsNode && labelsNode.IsSequence()) {
        cfg.classLabels.clear();
        for (const auto& item : labelsNode)
            cfg.classLabels.push_back(item.as<std::string>());
    }

    if (n["hidden_class_ids"] && n["hidden_class_ids"].IsSequence()) {
        cfg.hiddenClassIds.clear();
        for (const auto& item : n["hidden_class_ids"])
            cfg.hiddenClassIds.push_back(item.as<int>());
    }

    return cfg;
}

void AppConfig::dsai_saveToFile(const std::string& path) const {
    // Ensure parent directory exists
    std::string dir = path.substr(0, path.rfind('/'));
    if (!dir.empty())
        fs::create_directories(dir);

    YAML::Node n;
    n["backend"]          = static_cast<int>(backend);
    n["model_file"]       = modelFile;
    n["yaml_file"]        = yamlFile;
    n["source"]           = static_cast<int>(source);
    n["camera_device_id"] = cameraDeviceId;
    n["video_file"]       = videoFile;
    n["rtsp_url"]         = rtspUrl;
    n["iq_dir"]           = iqDir;
    n["capture_width"]    = captureWidth;
    n["capture_height"]   = captureHeight;
    n["capture_fps"]      = captureFps;
    n["gain"]             = gain;
    n["gamma"]            = gamma;
    n["brightness"]       = brightness;
    n["conf_threshold"]   = confThreshold;
    n["nms_threshold"]    = nmsThreshold;
    n["roi_enabled"]      = roiEnabled;
    n["rockchip_hw"]      = rockchipHw;
    n["web_port"]         = webPort;
    n["debug_logging"]    = debugLogging;
    n["rtsp_port"]        = rtspPort;

    YAML::Node roiNode;
    roiNode["x"]      = roi.x;
    roiNode["y"]      = roi.y;
    roiNode["width"]  = roi.width;
    roiNode["height"] = roi.height;
    n["roi"] = roiNode;

    YAML::Node labels;
    for (const auto& l : classLabels) labels.push_back(l);
    n["class_labels"] = labels;

    YAML::Node hidden;
    for (int id : hiddenClassIds) hidden.push_back(id);
    n["hidden_class_ids"] = hidden;

    // If the file already exists, patch it in-place to preserve comments.
    // Fall back to a fresh write when no prior file is present.
    if (fs::exists(path)) {
        std::string orig;
        { std::ifstream ifs(path); std::ostringstream ss; ss << ifs.rdbuf(); orig = ss.str(); }
        std::ofstream ofs(path);
        if (!ofs) throw std::runtime_error("AppConfig: cannot write '" + path + "'");
        ofs << patchYamlText(orig, n);
    } else {
        std::ofstream ofs(path);
        if (!ofs) throw std::runtime_error("AppConfig: cannot write '" + path + "'");
        ofs << n;
    }
}

std::string AppConfig::dsai_logConfigToString() const {
    static const char* kBackendNames[] = {"ONNX Runtime", "OpenVINO", "RKNN"};
    static const char* kSourceNames[]  = {"Camera", "VideoFile", "RTSP"};

    int bi = static_cast<int>(backend);
    int si = static_cast<int>(source);
    const char* backendStr = (bi >= 0 && bi < 3) ? kBackendNames[bi] : "Unknown";
    const char* sourceStr  = (si >= 0 && si < 3) ? kSourceNames[si]  : "Unknown";

    std::ostringstream o;
    char buf[128];

    o << "--- active configuration ---\n";
    o << "  Backend    : " << backendStr << "\n";
    o << "  Model      : " << (modelFile.empty() ? "(none)"  : modelFile) << "\n";
    o << "  YAML       : " << (yamlFile.empty()  ? "(auto)"  : yamlFile)  << "\n";

    snprintf(buf, sizeof(buf), "conf=%.2f  nms=%.2f", confThreshold, nmsThreshold);
    o << "  Thresholds : " << buf << "\n";

    if (!classLabels.empty()) {
        o << "  Classes    : [";
        for (size_t i = 0; i < classLabels.size(); ++i) {
            if (i) o << ", ";
            o << classLabels[i];
        }
        o << "]\n";
    }

    o << "  Source     : " << sourceStr;
    if (source == SourceType::Camera) {
        snprintf(buf, sizeof(buf), "  device=%d  %dx%d @ %dfps",
                 cameraDeviceId, captureWidth, captureHeight, captureFps);
        o << buf;
    } else if (source == SourceType::VideoFile) {
        o << "  " << videoFile;
    } else if (source == SourceType::Rtsp) {
        o << "  " << rtspUrl;
    }
    o << "\n";

    if (roiEnabled) {
        snprintf(buf, sizeof(buf), "x=%d y=%d w=%d h=%d",
                 roi.x, roi.y, roi.width, roi.height);
        o << "  ROI        : " << buf << "\n";
    } else {
        o << "  ROI        : disabled\n";
    }

    o << "  Web port   : " << webPort << "\n";
    o << "---";
    return o.str();
}

void AppConfig::dsai_logConfig() const {
    DBG_LOG("CONFIG", "\n%s\n", dsai_logConfigToString().c_str());
}
