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

#include <atomic>
#include <string>
#include <thread>

class HeadlessApp;

class WebConfigServer {
public:
    explicit WebConfigServer(HeadlessApp& app);
    ~WebConfigServer();

    // Bind to app.dsai_config().webPort and start serving in a background thread.
    // Returns false if the port cannot be bound.
    bool dsai_start();
    void dsai_stop();

private:
    void        serveLoop();
    void        handleClient(int fd);
    std::string dispatch(const std::string& method,
                         const std::string& path,
                         const std::string& body);

    // GET handlers
    std::string dsai_jsonConfigResp();

    // POST handlers (update in-memory config + save to disk, no restart)
    std::string dsai_applyModel(const std::string& body);
    std::string dsai_applySource(const std::string& body);
    std::string applyDetection(const std::string& body);
    std::string dsai_applyRoi(const std::string& body);
    std::string applySystem(const std::string& body);

    // POST /api/restart — signal inference loop to stop; main() will re-exec
    std::string dsai_triggerRestart();

    // GET /stream — MJPEG stream
    std::string streamResp();

    HeadlessApp&      app_;
    int               serverFd_ = -1;
    std::atomic<bool> running_  { false };
    std::thread       thread_;
};
