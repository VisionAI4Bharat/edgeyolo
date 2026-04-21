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

/**
 * RockchipCapture — hardware camera / RTSP capture for Rockchip SoCs.
 *
 * Zero Qt / UI dependency: pure C++17 + OpenCV.
 * Compiled as libdeepSightAI_capture.so — link from any application.
 *
 * Two modes
 * =========
 * dsai_openCamera()  — MIPI / ISP camera via Rockchip RK MPI VI pipeline.
 *                 Full ISP pipeline (RKAIQ) + YUV420SP → BGR.
 *                 Requires WITH_RKNN build flag + LUCKFOX_SDK_ROOT.
 *
 * dsai_openRtsp()    — Network RTSP / RTMP stream via OpenCV / ffmpeg.
 *                 Works on any platform.  Hardware VDEC (RK_MPI_VDEC) can
 *                 be substituted here in a future revision without changing
 *                 the public API.
 *
 * Fallback
 * ========
 * When built without WITH_RKNN, dsai_openCamera() falls back to cv::VideoCapture
 * (V4L2) so the same source tree compiles on a desktop for development.
 *
 * Thread safety
 * =============
 * dsai_read() is safe to call from a single dedicated capture thread.
 * All other methods must be called from the owning thread.
 */
#pragma once

#include <memory>
#include <string>
#include <opencv2/core.hpp>

namespace deepSightAI {

class RockchipCapture {
public:
    // ── Camera (MIPI / ISP) configuration ────────────────────────────────────
    struct CameraConfig {
        int         width   = 1920;
        int         height  = 1080;
        double      fps     = 30.0;
        std::string iqDir   = "/etc/iqfiles";  // RKAIQ ISP tuning files
        int         devId   = 0;               // VI device index (usually 0)
        int         chnId   = 0;               // VI channel index (usually 0)
    };

    // ── RTSP / network source configuration ──────────────────────────────────
    struct RtspConfig {
        std::string url;
        int         openTimeoutMs = 5000;   // ffmpeg connect timeout
        int         readTimeoutMs = 10000;  // ffmpeg read timeout
    };

    RockchipCapture();
    ~RockchipCapture();

    // Non-copyable (owns hardware resources)
    RockchipCapture(const RockchipCapture&)            = delete;
    RockchipCapture& operator=(const RockchipCapture&) = delete;

    // Movable
    RockchipCapture(RockchipCapture&&) noexcept;
    RockchipCapture& operator=(RockchipCapture&&) noexcept;

    /**
     * Open a MIPI / CSI camera via the RK MPI VI pipeline.
     * Initialises RKAIQ ISP, RK MPI system, VI device and VI channel.
     * @return true on success.
     */
    bool dsai_openCamera(const CameraConfig& cfg);

    /**
     * Open a network RTSP / RTMP stream.
     * @return true on success.
     */
    bool dsai_openRtsp(const RtspConfig& cfg);

    bool dsai_isOpened() const noexcept;

    /**
     * Capture the next frame.
     *
     * Camera path (WITH_RKNN): blocks until the VI hardware delivers a new
     * YUV420SP frame, converts in-place to BGR, then deep-copies to bgrFrame.
     *
     * RTSP / fallback path: delegates to cv::VideoCapture::dsai_read().
     *
     * @param bgrFrame  Receives a BGR CV_8UC3 Mat.
     * @return true on success; false on error or end-of-stream.
     */
    bool dsai_read(cv::Mat& bgrFrame);

    /** Release hardware resources and close the source. */
    void dsai_release();

    int    dsai_captureWidth()  const noexcept;
    int    dsai_captureHeight() const noexcept;
    double dsai_captureFps()    const noexcept;

    /** Human-readable description of the last error, or empty string. */
    std::string dsai_lastError() const;

    // Opaque implementation — defined only in rockchip_capture.cpp.
    // Public so static helper functions in the .cpp can name it without
    // triggering private-access errors on strict compilers (GCC 8.x).
    struct Impl;

private:
    std::unique_ptr<Impl> pImpl_;
};

}  // namespace deepSightAI
