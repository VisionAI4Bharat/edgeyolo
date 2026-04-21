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

#ifndef VIDEOWIDGET_H
#define VIDEOWIDGET_H

#include <QWidget>
#include <QImage>
#include <QPixmap>
#include <QMouseEvent>
#include <QPainter>
#include <QRect>
#include <QVector>
#include <QString>
#include <QMutex>
#include <QThread>
#include <QWaitCondition>
#include <atomic>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#ifdef HAVE_OPENCV_VIDEOIO
#  include <opencv2/videoio.hpp>
#endif

class VideoWidget : public QWidget {
    Q_OBJECT
public:
    explicit VideoWidget(QWidget *parent = nullptr);
    ~VideoWidget();

    void dsai_setCameraDevice(int deviceId);
    void dsai_setVideoSource(const QString& path);  // video file or RTSP URL
    void dsai_setEditMode(bool editMode);
    void dsai_setBoundingBox(const QRect& box);
    QRect dsai_getBoundingBox() const;
    void dsai_setDetectionResults(const QVector<QRect>& boxes,
                            const QVector<int>& classIds,
                            const QVector<float>& confidences);

    /**
     * Load ROI bounding box (and optionally class names) from a YAML file.
     * The roi key stores pixel coordinates as integers:
     *   roi: {x: 10, y: 20, width: 300, height: 200}
     * @param configPath  Path to the YAML file.  No-op if empty or file absent.
     */
    void dsai_loadRoiFromConfig(const QString& configPath);

    /**
     * Save the current ROI bounding box to a YAML file.
     * Merges with existing YAML content if the file already exists.
     * Removes the roi key if no box is set.
     * @param configPath  Destination YAML path.
     * @throws std::runtime_error on write failure.
     */
    void dsai_saveRoiToConfig(const QString& configPath);

    void dsai_stopCaptureThread();
    void dsai_setClassNames(const QStringList& names);
    void dsai_setRockchipHardware(bool enabled) { rockchipHw_ = enabled; }

    /** Enable/disable verbose debug logging (forwards to the global Debug flag). */
    void dsai_setDebugLogging(bool enabled);

signals:
    void boundingBoxChanged(const QRect& box);
    /** Emitted for every decoded frame, carrying the raw BGR cv::Mat. */
    void frameReady(const cv::Mat& frame);

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    void dsai_updateFrame();
    void dsai_startCaptureThread();
    QPointF dsai_widgetToImageCoordinates(const QPointF& widgetPos);

    // Video capture (unavailable when opencv-mobile excludes videoio)
#ifdef HAVE_OPENCV_VIDEOIO
    cv::VideoCapture  capture_;
#endif
    std::atomic<bool> running_;
    QThread*          captureThread_ = nullptr;  // owned; QThread::create() returns QThread*
    mutable QMutex    frameMutex_;
    QWaitCondition    frameCondition_;
    cv::Mat currentFrame_;
    cv::Mat displayFrame_;
    int     cameraDeviceId_  = 0;
    QString videoSourcePath_; // non-empty = video file or RTSP, overrides camera

    // Display properties — qtImage_ is written from capture thread (protected by
    // displayMutex_); qtPixmap_ and lastFrameSize_ are GUI-thread-only (set in paintEvent).
    QImage qtImage_;
    QPixmap qtPixmap_;
    mutable QMutex displayMutex_;   // guards qtImage_
    QSize  lastFrameSize_;  // original frame size, cached in paintEvent for coord conversion

    // Bounding box for annotation
    QRect boundingBox_;
    std::atomic<bool> isEditMode_;
    std::atomic<bool> isDrawingBox_;
    QPoint boxStartPoint_;
    QPoint boxCurrentPoint_;
    static constexpr int BOX_COLOR_R = 0;
    static constexpr int BOX_COLOR_G = 0;
    static constexpr int BOX_COLOR_B = 255; // Blue
    static constexpr int BOX_WIDTH = 2;

    // Detection results
    QVector<QRect> detectionBoxes_;
    QVector<int> detectionClassIds_;
    QVector<float> detectionConfidences_;
    QMutex resultsMutex_;

    // Configuration
    QString configFilePath_;
    float metersToPixelsRatio_; // For 1m box calibration
    int referenceWidth_; // Reference width in pixels for calibration

    // Class names for object selection (populated from config)
    QStringList classNames_;

    // Rockchip hardware mode
    bool rockchipHw_ = false;

    // Logo overlay
    QPixmap logoPixmap_;
};

#endif // VIDEOWIDGET_H