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

#pragma once

#include "../app_config.h"
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <opencv2/core.hpp>

namespace inference { class IDetector; }

class HeadlessApp {
public:
    explicit HeadlessApp(const std::string& configPath = "");
    ~HeadlessApp();

    int  run();
    void dsai_stop();
    void dsai_requestRestart();

    AppConfig&       dsai_config()       { return cfg_; }
    const AppConfig& dsai_config() const { return cfg_; }
    const std::string& configPath() const { return configPath_; }

    void dsai_applyAndRestart(const AppConfig& newCfg);

    // Frame sharing for web stream
    void dsai_pushFrame(const cv::Mat& frame);
    cv::Mat dsai_latestFrame();

private:
    void dsai_runInferenceLoop();

    AppConfig   cfg_;
    std::string configPath_;

    std::atomic<bool> running_  { false };
    std::atomic<bool> restart_  { false };

    std::mutex frameMutex_;
    cv::Mat    latestFrame_;
};
