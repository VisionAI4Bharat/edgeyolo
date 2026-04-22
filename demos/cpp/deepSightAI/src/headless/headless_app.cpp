/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#include "headless_app.h"
#include "../app_config.h"
#include "../debug_log.h"

#include "capture/CaptureFactory.h"
#include "inference/IDetector.h"
#include "inference/DetectorFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv_modules.hpp>

#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>

HeadlessApp::HeadlessApp(const std::string& configPath)
    : configPath_(configPath.empty() ? AppConfig::dsai_defaultPath() : configPath)
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
    try { cfg_.dsai_saveToFile(configPath_); }
    catch (const std::exception& e) {
        fprintf(stderr, "[HeadlessApp] Config save failed: %s\n", e.what());
    }
    dsai_requestRestart();
}

void HeadlessApp::dsai_setHiddenClasses(const std::vector<int>& ids) {
    cfg_.hiddenClassIds = ids;
    try { cfg_.dsai_saveToFile(configPath_); } catch(...) {}
}

int HeadlessApp::run() {
    running_ = true;
    restart_ = false;
    dsai_runInferenceLoop();
    return restart_.load() ? 1 : 0;
}

void HeadlessApp::dsai_pushFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frameMutex_);
    latestFrame_ = frame.clone();
}

cv::Mat HeadlessApp::dsai_latestFrame() {
    std::lock_guard<std::mutex> lock(frameMutex_);
    return latestFrame_.clone();
}

void HeadlessApp::dsai_runInferenceLoop() {
    Debug::dsai_setEnabled(cfg_.debugLogging);

    auto cap_ptr = deepSightAI::CaptureFactory::dsai_create();
    deepSightAI::ICapture& cap = *cap_ptr;
    cap.dsai_setAppConfig(cfg_);

    if (cfg_.modelFile.empty()) {
        fprintf(stderr, "[HeadlessApp] No model file configured.\n");
        return;
    }

    std::unique_ptr<inference::IDetector> detector;
    try {
        inference::Backend detBackend = static_cast<inference::Backend>(cfg_.backend);
        detector = inference::DetectorFactory::dsai_create(detBackend, cfg_.modelFile, cfg_.yamlFile, cfg_.confThreshold, cfg_.nmsThreshold);
        if (!cfg_.classLabels.empty()) detector->dsai_setClassLabels(cfg_.classLabels);
        
        cv::Size inputSize = detector->dsai_inputSize();
        cap.dsai_setModelInputSize(inputSize.width, inputSize.height);
    } catch (const std::exception& e) {
        fprintf(stderr, "[HeadlessApp] Detector init failed: %s\n", e.what());
        return;
    }

    bool opened = false;
    if (cfg_.source == SourceType::VideoFile) {
        printf("[HeadlessApp] Opening video file: %s\n", cfg_.videoFile.c_str());
        opened = cap.dsai_openSource(cfg_.videoFile);
    } else if (cfg_.source == SourceType::Rtsp) {
        printf("[HeadlessApp] Opening RTSP stream: %s\n", cfg_.rtspUrl.c_str());
        opened = cap.dsai_openSource(cfg_.rtspUrl);
    } else {
        printf("[HeadlessApp] Opening camera: device %d, %dx%d @ %dfps\n",
               cfg_.cameraDeviceId, cfg_.dsai_width(), cfg_.dsai_height(), cfg_.dsai_fps());
        opened = cap.dsai_openCamera(cfg_.cameraDeviceId, cfg_.dsai_width(), cfg_.dsai_height(), static_cast<double>(cfg_.dsai_fps()));
    }

    if (!opened) {
        fprintf(stderr, "[HeadlessApp] Capture open failed: %s\n", cap.dsai_lastError().c_str());
        return;
    }

    printf("[HeadlessApp] Capture open: %dx%d @ %.1ffps\n",
           cap.dsai_captureWidth(), cap.dsai_captureHeight(), cap.dsai_captureFps());

    auto applyRoi = [&](const cv::Mat& frame) -> cv::Mat {
        if (!cfg_.roiEnabled || cfg_.roi.width <= 0 || cfg_.roi.height <= 0) return frame;
        cv::Rect r(cfg_.roi.x, cfg_.roi.y, cfg_.roi.width, cfg_.roi.height);
        r &= cv::Rect(0, 0, frame.cols, frame.rows);
        return frame(r);
    };

    unsigned long long frameN = 0;
    cv::Mat frame;
    while (running_) {
        if (!cap.dsai_read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        cv::Mat roi = applyRoi(frame);
        auto detections = detector->dsai_infer(roi);
        
        // Filter detections
        std::vector<inference::Detection> filtered;
        for (const auto& d : detections) {
            bool hidden = false;
            for (int id : cfg_.hiddenClassIds) { if (id == d.classId) { hidden = true; break; } }
            if (!hidden) filtered.push_back(d);
        }

        cap.dsai_setOSD(filtered);

        // Draw for web stream preview (Headless Vision)
        for (const auto& d : filtered) {
            cv::rectangle(frame, d.rect, cv::Scalar(0, 255, 0), 2);
            std::string label = (d.classId < (int)cfg_.classLabels.size() ? cfg_.classLabels[d.classId] : std::to_string(d.classId));
            cv::putText(frame, label, cv::Point(d.rect.x, d.rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        dsai_pushFrame(frame);

        frameN++;
        DBG_LOG("DETECTOR", "frame %llu: %zu detections\n", frameN, filtered.size());

        if (frameN % 100 == 0) {
            printf("[HeadlessApp] Frame %llu, detections=%zu\n", frameN, filtered.size());
        }
    }

    cap.dsai_release();
}


