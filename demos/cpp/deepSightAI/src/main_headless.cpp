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
#include <cstring>
#include <string>
#include <unistd.h>
#include <limits.h>

extern char** environ;  // POSIX — re-exec support

static HeadlessApp* g_app = nullptr;

static void onSignal(int sig) {
    fprintf(stderr, "\n[main] Signal %d — stopping.\n", sig);
    if (g_app) g_app->dsai_stop();
}

// Resolve the absolute path of the running executable via /proc/self/exe.
// Falls back to argv[0] if the symlink cannot be read.
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
        "  --config / -c <path>   Config YAML file (default: ~/.config/deepSightAI/config.yaml)\n"
        "  --help    / -h         Show this help\n"
        "\n"
        "The web configuration dashboard is served on the port specified in the config\n"
        "(default: 8080). Open http://<device-ip>:8080 to configure the pipeline.\n",
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
            // Bare positional argument: accept any .yaml/.yml file
            const size_t dot = a.rfind('.');
            if (dot != std::string::npos) {
                const std::string ext = a.substr(dot);
                if (ext == ".yaml" || ext == ".yml") cfgPath = a;
            }
        }
    }

    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);
    std::signal(SIGPIPE, SIG_IGN);  // ignore broken pipe from web clients

    const std::string exePath = resolveExe(argv[0]);

    for (;;) {
        HeadlessApp app(cfgPath);
        g_app = &app;

        WebConfigServer web(app);
        if (!web.dsai_start()) {
            fprintf(stderr,
                "[main] Web server could not bind port %d — continuing without it.\n",
                app.dsai_config().webPort);
        }

        const int code = app.run();
        web.dsai_stop();
        g_app = nullptr;

        if (code != 1) break;  // 0 = clean stop via signal

        // Restart requested (code == 1): re-exec for a fully clean process state.
        printf("[main] Restarting…\n");
        fflush(stdout);
        execve(exePath.c_str(), argv, environ);

        // execve only returns on failure — fall back to in-process restart
        perror("[main] execve failed; restarting in-process");
        // loop continues with a freshly constructed HeadlessApp
    }

    printf("[main] Exited cleanly.\n");
    return 0;
}
