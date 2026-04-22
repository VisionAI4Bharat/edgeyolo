/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#pragma once

#include <QWidget>
#include <QMutex>
#include <QImage>
#include <QPixmap>
#include <QThread>
#include <QWaitCondition>
#include <atomic>
#include <opencv2/core.hpp>
#include "../inference/IDetector.h"
#include "../capture/ICapture.h"
#include "../capture/CaptureFactory.h"

class VideoWidget : public QWidget {
    Q_OBJECT
public:
    explicit VideoWidget(QWidget* parent = nullptr);
    ~VideoWidget();

    void dsai_setCameraDevice(int devId);
    void dsai_setVideoSource(const QString& path);
    void dsai_setRockchipHardware(bool enabled);
    void dsai_setAppConfig(const AppConfig& cfg);
    void dsai_setModelInputSize(const cv::Size& size);
    void dsai_setClassNames(const QStringList& names);
    void dsai_setDetectionResults(const std::vector<inference::Detection>& results);
    void dsai_loadRoiFromConfig(const QString& path);
    void dsai_saveRoiToConfig(const QString& path);
    void dsai_setEditMode(bool edit);
    void dsai_clearRoi();

    void dsai_startCaptureThread();
    void dsai_stopCaptureThread();

signals:
    void frameReady(const cv::Mat& frame);
    void boundingBoxChanged(const QRect& rect);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    void dsai_updateFrame();
    QPointF dsai_widgetToImageCoordinates(const QPointF& widgetPos) const;

    std::unique_ptr<deepSightAI::ICapture> capture_;
    QMutex captureMutex_;
    QThread* captureThread_ = nullptr;
    std::atomic<bool> running_{false};

    QMutex displayMutex_;
    cv::Mat currentFrame_;
    cv::Mat displayFrame_;
    QImage qtImage_;
    QPixmap qtPixmap_;
    cv::Size lastFrameSize_;
    
    QMutex resultsMutex_;
    std::vector<inference::Detection> detections_;
    
    bool isEditingRoi_ = false;
    bool isDrawingBox_ = false;
    QPoint boxStartPoint_;
    QPoint boxCurrentPoint_;
    QRect boundingBox_;
    
    QStringList classNames_;
    bool isRockchip_ = false;
    QPixmap logoPixmap_;
    
    AppConfig appConfig_;
    cv::Size modelInputSize_{0, 0};

    int cameraDeviceId_ = 0;
    QString videoSourcePath_;
};
