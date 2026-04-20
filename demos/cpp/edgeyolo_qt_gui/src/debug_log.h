#pragma once

#include <atomic>
#include <cstdio>

/**
 * Lightweight debug logging controlled by a single global flag.
 *
 * Enable at runtime via Debug::setEnabled(true) — toggled from the
 * "Debug Logging" checkbox in the Configure dialog.
 *
 * DBG_LOG(tag, fmt, ...)  — stdout, only when enabled
 * ERR_LOG(tag, fmt, ...)  — stderr, always
 */
namespace Debug {
    extern std::atomic<bool> enabled;
    inline void setEnabled(bool on) { enabled.store(on); }
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
