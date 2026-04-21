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

#include "headless_app.h"
#include "../app_config.h"

#include "capture/rockchip_capture.h"
#include "inference/IDetector.h"
#include "inference/DetectorFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv_modules.hpp>

#include <cstdio>
#include <cstring>
#include <chrono>

// ─────────────────────────────────────────────────────────────────────────────

HeadlessApp::HeadlessApp(const std::string& configPath)
    : configPath_(configPath.empty() ? AppConfig::defaultPath() : configPath)
{
    try {
        cfg_ = AppConfig::dsai_loadFromFile(configPath_);
    } catch (const std::exception& e) {
        fprintf(stderr, "[HeadlessApp] Config load failed (%s), using defaults.\n", e.what());
    }
}

HeadlessApp::~HeadlessApp() { dsai_stop(); }

void HeadlessApp::dsai_stop()    { running_ = false; }
void HeadlessApp::dsai_requestRestart() { restart_ = true; running_ = false; }

void HeadlessApp::dsai_applyAndRestart(const AppConfig& newCfg) {
    cfg_ = newCfg;
    try { cfg_.saveToFile(configPath_); }
    catch (const std::exception& e) {
        fprintf(stderr, "[HeadlessApp] Config save failed: %s\n", e.what());
    }
    dsai_requestRestart();
}

// ─── inference loop ───────────────────────────────────────────────────────────

int HeadlessApp::run() {
    running_ = true;
    restart_ = false;

    dsai_runInferenceLoop();

    return restart_.load() ? 1 : 0;
}

void HeadlessApp::dsai_runInferenceLoop() {
    // ── open capture source ──────────────────────────────────────────────
    deepSightAI::RockchipCapture cap;
    bool opened = false;

    if (cfg_.source == SourceType::Rtsp && !cfg_.rtspUrl.empty()) {
        deepSightAI::RockchipCapture::RtspConfig rc;
        rc.url           = cfg_.rtspUrl;
        rc.openTimeoutMs = 8000;
        rc.readTimeoutMs = 15000;
        opened = cap.dsai_openRtsp(rc);
        if (!opened)
            fprintf(stderr, "[HeadlessApp] RTSP open failed: %s\n", cap.dsai_lastError().c_str());
    }

    if (!opened) {
        deepSightAI::RockchipCapture::CameraConfig cc;
        cc.devId  = cfg_.cameraDeviceId;
        cc.width  = cfg_.width();
        cc.height = cfg_.height();
        cc.fps    = static_cast<double>(cfg_ .fps());
        cc.iqDir  = cfg_.iqDir;
        opened = cap.dsai_openCamera(cc);
        if (!opened) {
            fprintf(stderr, "[HeadlessApp] Camera open failed: %s\n", cap.dsai_lastError().c_str());
            return;
        }
    }

    printf("[HeadlessApp] Capture open: %dx%d @ %.0ffps\n",
           cap.dsai_captureWidth(), cap.dsai_captureHeight(), cap.dsai_captureFps());

    // ── load detector ────────────────────────────────────────────────────
    if (cfg_.modelFile.empty()) {
        fprintf(stderr, "[HeadlessApp] No model file configured. Set model_file in config.\n");
        return;
    }

    std::unique_ptr<inference::IDetector> detector;
    try {
        inference::Backend detBackend;
        switch (cfg_.backend) {
            case Backend::RKNN:    detBackend = inference::Backend::RKNN;    break;
            case Backend::OpenVINO: detBackend = inference::Backend::OPENVINO; break;
            default:               detBackend = inference::Backend::ONNX;    break;
        }
        detector = inference::DetectorFactory::dsai_create(
            detBackend, cfg_.modelFile, cfg_.yamlFile,
            cfg_.confThreshold, cfg_.nmsThreshold);
    } catch (const std::exception& e) {
        fprintf(stderr, "[HeadlessApp] Detector load failed: %s\n", e.what());
        return;
    }

    if (!cfg_.classLabels.empty())
        detector->dsai_setClassLabels(cfg_.classLabels);

    printf("[HeadlessApp] Model loaded: %s  input=%dx%d\n",
           cfg_.modelFile.c_str(),
           detector->dsai_inputSize().width, detector->dsai_inputSize().height);

    // ── ROI helper ───────────────────────────────────────────────────────
    auto applyRoi = [&](const cv::Mat& frame) -> cv::Mat {
        if (!cfg_.roiEnabled || cfg_.roi.width <= 0 || cfg_.roi.height <= 0)
            return frame;
        cv::Rect r(cfg_.roi.x, cfg_.roi.y, cfg_.roi.width, cfg_.roi.height);
        r &= cv::Rect(0, 0, frame.cols, frame.rows);
        return frame(r).clone();
    };

    // ── main loop ────────────────────────────────────────────────────────
    using Clock = std::chrono::steady_clock;
    uint64_t frameN = 0;

    while (running_) {
        cv::Mat frame;
        if (!cap.dsai_read(frame) || frame.empty()) {
            fprintf(stderr, "[HeadlessApp] Frame read failed — stopping.\n");
            break;
        }

        cv::Mat roi = dsai_applyRoi(frame);

        std::vector<inference::Detection> detections;
        try {
            detections = detector->dsai_infer(roi);
        } catch (const std::exception& e) {
            fprintf(stderr, "[HeadlessApp] infer error: %s\n", e.what());
            continue;
        }

        ++frameN;

        // Print detections to stdout (one line per frame with hits)
        if (!detections.empty() && cfg_.debugLogging) {
            const auto& names = detector->dsai_classNames();
            printf("[frame %llu]", static_cast<unsigned long long>(frameN));
            for (const auto& d : detections) {
                const char* label = (d.classId >= 0 && d.classId < (int)names.size())
                    ? names[d.classId].c_str() : "?";
                printf("  %s %.2f", label, d.confidence);
            }
            printf("\n");
        }

        // Periodic heartbeat
        if (frameN % 300 == 0)
            printf("[HeadlessApp] heartbeat frame %llu  detections=%zu\n",
                   static_cast<unsigned long long>(frameN), detections.size());
    }

    cap.dsai_release();
    printf("[HeadlessApp] Loop exited after %llu frames.\n",
           static_cast<unsigned long long>(frameN));
}
