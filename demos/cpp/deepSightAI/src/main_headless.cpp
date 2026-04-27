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

#include "headless/headless_app.h"
#include "headless/web_config.h"

#include <csignal>
#include <cstdio>
#include <string>
#include <unistd.h>
#include <limits.h>

extern char** environ;

static HeadlessApp* g_app = nullptr;

static void onSignal(int sig) {
    fprintf(stderr, "\n[main] Signal %d — stopping.\n", sig);
    if (g_app) g_app->dsai_stop();
}

static std::string resolveExe(const char* argv0) {
    char buf[PATH_MAX] = {};
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) return std::string(buf, static_cast<size_t>(len));
    return argv0;
}

static void printUsage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [--config <path.yaml>] [path.yaml]\n"
        "\n"
        "  --config / -c <path>   Config YAML file\n"
        "  --help    / -h         Show this help\n"
        "\n"
        "The web dashboard is served on the port in the config (default: 8080).\n"
        "It stays live even when the inference pipeline fails — fix the config\n"
        "and hit /api/restart to recover without rebooting.\n",
        prog);
}

int main(int argc, char* argv[]) {
    std::string cfgPath;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--config" || a == "-c") && i + 1 < argc) {
            cfgPath = argv[++i];
        } else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (!a.empty() && a[0] != '-') {
            const size_t dot = a.rfind('.');
            if (dot != std::string::npos) {
                const std::string ext = a.substr(dot);
                if (ext == ".yaml" || ext == ".yml") cfgPath = a;
            }
        }
    }

    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);
    std::signal(SIGPIPE, SIG_IGN);

    const std::string exePath = resolveExe(argv[0]);

    // HeadlessApp and WebConfigServer are created once and live for the
    // entire process lifetime.  The inference pipeline inside HeadlessApp
    // can fail without taking the web server down — the operator can always
    // reconfigure and restart via the dashboard.
    HeadlessApp app(cfgPath);
    g_app = &app;

    WebConfigServer web(app);
    if (!web.dsai_start()) {
        fprintf(stderr, "[main] Web server could not bind port %d — continuing without it.\n",
                app.dsai_config().webPort);
    }

    for (;;) {
        // run() blocks until inference stops cleanly (returns 0) or a restart
        // is requested via the web dashboard (returns 1).  On inference error,
        // run() itself waits for an explicit action so the web server stays live.
        const int code = app.run();
        web.dsai_stop();
        g_app = nullptr;

        if (code != 1) break;  // 0 = clean stop via SIGINT/SIGTERM

        // Restart requested: re-exec for a fully clean hardware state (ISP, RKNN, VI).
        fprintf(stdout, "[main] Restart requested — re-execing.\n");
        fflush(stdout);
        execve(exePath.c_str(), argv, environ);
        // execve only returns on failure — nothing we can do without a clean process
        perror("[main] execve failed — cannot restart");
        break;
    }

    fprintf(stdout, "[main] Exited cleanly.\n");
    return 0;
}
