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
#include <cstdio>

/**
 * Lightweight debug logging controlled by a single global flag.
 *
 * Enable at runtime via Debug::dsai_setEnabled(true) — toggled from the
 * "Debug Logging" checkbox in the Configure dialog.
 *
 * DBG_LOG(tag, fmt, ...)  — stdout, only when enabled
 * ERR_LOG(tag, fmt, ...)  — stderr, always
 */
namespace Debug {
    extern std::atomic<bool> enabled;
    inline void dsai_setEnabled(bool on) { enabled.store(on); }
    inline bool isEnabled()         { return enabled.load(); }
}

#define DBG_LOG(tag, ...)                                               \
    do {                                                                \
        if (Debug::isEnabled()) {                                       \
            fprintf(stdout, "[" tag "] " __VA_ARGS__);                  \
            fflush(stdout);                                             \
        }                                                               \
    } while (0)

#define ERR_LOG(tag, ...)                                               \
    do {                                                                \
        fprintf(stderr, "[" tag " ERROR] " __VA_ARGS__);               \
        fflush(stderr);                                                 \
    } while (0)
