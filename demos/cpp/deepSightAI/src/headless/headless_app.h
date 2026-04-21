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

namespace inference { class IDetector; }

class HeadlessApp {
public:
    explicit HeadlessApp(const std::string& configPath = "");
    ~HeadlessApp();

    // Run capture→infer loop in foreground until dsai_stop() is called.
    // Returns exit code: 0 = clean exit, 1 = restart requested.
    int  run();
    void dsai_stop();

    // Reload config from disk and restart the inference loop.
    void dsai_requestRestart();

    // Config access (used by web server)
    AppConfig&       dsai_config()       { return cfg_; }
    const AppConfig& dsai_config() const { return cfg_; }
    const std::string& configPath() const { return configPath_; }

    // Save current config and signal restart
    void dsai_applyAndRestart(const AppConfig& newCfg);

private:
    void dsai_runInferenceLoop();

    AppConfig   cfg_;
    std::string configPath_;

    std::atomic<bool> running_  { false };
    std::atomic<bool> restart_  { false };
};
