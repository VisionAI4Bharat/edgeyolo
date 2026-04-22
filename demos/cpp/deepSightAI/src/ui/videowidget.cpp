#include "videowidget.h"
#include "../debug_log.h"
#include <QPainter>
#include <QMouseEvent>
#include <QMutexLocker>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <chrono>

#define TAG "VideoWidget"

VideoWidget::VideoWidget(QWidget* parent) : QWidget(parent) {
    logoPixmap_.load(":/assets/logo.png");
}

VideoWidget::~VideoWidget() { dsai_stopCaptureThread(); }

void VideoWidget::dsai_setCameraDevice(int devId) { cameraDeviceId_ = devId; videoSourcePath_.clear(); }
void VideoWidget::dsai_setVideoSource(const QString& path) { videoSourcePath_ = path; }
void VideoWidget::dsai_setRockchipHardware(bool enabled) { isRockchip_ = enabled; }
void VideoWidget::dsai_setAppConfig(const AppConfig& cfg) { appConfig_ = cfg; }
void VideoWidget::dsai_setModelInputSize(const cv::Size& size) { modelInputSize_ = size; }
void VideoWidget::dsai_setClassNames(const QStringList& names) { classNames_ = names; }

void VideoWidget::dsai_setDetectionResults(const std::vector<inference::Detection>& results) {
    { std::lock_guard<QMutex> lock(resultsMutex_); detections_ = results; }
    if (isRockchip_ && capture_) capture_->dsai_setOSD(results);
}

void VideoWidget::dsai_loadRoiFromConfig(const QString& path) {
    if (path.isEmpty()) return;
    try {
        YAML::Node cfg = YAML::LoadFile(path.toStdString());
        if (cfg["roi"]) {
            auto r = cfg["roi"];
            boundingBox_ = QRect(r["x"].as<int>(), r["y"].as<int>(), r["width"].as<int>(), r["height"].as<int>());
            update();
        }
    } catch(...) {}
}

void VideoWidget::dsai_saveRoiToConfig(const QString& path) {
    if (path.isEmpty()) return;
    YAML::Node cfg;
    cfg["roi"]["x"] = boundingBox_.x();
    cfg["roi"]["y"] = boundingBox_.y();
    cfg["roi"]["width"] = boundingBox_.width();
    cfg["roi"]["height"] = boundingBox_.height();
    std::ofstream fout(path.toStdString());
    fout << cfg;
}

void VideoWidget::dsai_setEditMode(bool edit) { isEditingRoi_ = edit; isDrawingBox_ = false; update(); }
void VideoWidget::dsai_clearRoi() { boundingBox_ = QRect(); update(); }

void VideoWidget::dsai_startCaptureThread() {
    dsai_stopCaptureThread();
    running_ = true;
    const int devId = cameraDeviceId_;
    const QString path = videoSourcePath_;
    const AppConfig cfg = appConfig_;
    const cv::Size modelSize = modelInputSize_;

    captureThread_ = QThread::create([this, devId, path, cfg, modelSize]() {
        auto cap = deepSightAI::CaptureFactory::dsai_create();
        cap->dsai_setAppConfig(cfg);
        cap->dsai_setModelInputSize(modelSize.width, modelSize.height);

        bool opened = false;
        if (!path.isEmpty()) {
            DBG_LOG(TAG, "Opening source: %s\n", path.toStdString().c_str());
            opened = cap->dsai_openSource(path.toStdString());
        } else {
            DBG_LOG(TAG, "Opening camera: %d (%dx%d)\n", devId, cfg.dsai_width(), cfg.dsai_height());
            opened = cap->dsai_openCamera(devId, cfg.dsai_width(), cfg.dsai_height(), (double)cfg.dsai_fps());
        }
        if (!opened) { fprintf(stderr, "[VideoWidget] Failed to open source: %s\n", cap->dsai_lastError().c_str()); return; }

        // Pacing logic
        double srcFps = cap->dsai_captureFps();
        int frameDelayMs = (srcFps > 0 && srcFps < 300) ? (int)(1000.0 / srcFps) : 33;

        { QMutexLocker lock(&captureMutex_); capture_ = std::move(cap); }
        while(running_) {
            auto start = std::chrono::steady_clock::now();
            cv::Mat frame;
            if (capture_ && capture_->dsai_read(frame)) {
                { std::lock_guard<QMutex> lock(displayMutex_); currentFrame_ = frame.clone(); }
                emit frameReady(frame);
                QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);
            }
            auto end = std::chrono::steady_clock::now();
            int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            int sleep = frameDelayMs - elapsed;
            if (sleep > 0) QThread::msleep(sleep);
            else QThread::msleep(1);
        }
        if (capture_) capture_->dsai_release();
    });
    captureThread_->start();
}

void VideoWidget::dsai_stopCaptureThread() {
    running_ = false;
    if (captureThread_) {
        captureThread_->wait();
        delete captureThread_;
        captureThread_ = nullptr;
    }
    { std::lock_guard<QMutex> lock(displayMutex_); currentFrame_ = cv::Mat(); }
    update();
}

void VideoWidget::dsai_updateFrame() {}

void VideoWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

    cv::Mat frame;
    { std::lock_guard<QMutex> lock(displayMutex_); if (currentFrame_.empty()) return; frame = currentFrame_; }

    QImage img((const uchar*)frame.data, frame.cols, frame.rows, frame.step, QImage::Format_BGR888);
    qtPixmap_ = QPixmap::fromImage(img).scaled(size(), Qt::KeepAspectRatio, Qt::FastTransformation);
    lastFrameSize_ = cv::Size(frame.cols, frame.rows);

    int xOff = (width() - qtPixmap_.width()) / 2;
    int yOff = (height() - qtPixmap_.height()) / 2;
    painter.drawPixmap(xOff, yOff, qtPixmap_);

    float scaleX = (float)qtPixmap_.width() / frame.cols;
    float scaleY = (float)qtPixmap_.height() / frame.rows;

    // Overlays
    static const cv::Scalar kColors[] = {{0,255,0}, {0,0,255}, {255,0,0}, {0,255,255}, {255,0,255}, {255,128,0}};
    {
        std::lock_guard<QMutex> lock(resultsMutex_);
        for (const auto& det : detections_) {
            QColor color(kColors[det.classId % 6][2], kColors[det.classId % 6][1], kColors[det.classId % 6][0]);
            painter.setPen(QPen(color, 2));
            QRect r(xOff + det.rect.x * scaleX, yOff + det.rect.y * scaleY, det.rect.width * scaleX, det.rect.height * scaleY);
            painter.drawRect(r);
            QString name = (det.classId < (int)classNames_.size() ? classNames_[det.classId] : QString("ID:%1").arg(det.classId));
            painter.drawText(r.topLeft() - QPoint(0, 5), QString("%1 %2").arg(name).arg(det.confidence, 0, 'f', 2));
        }
    }
    if (!boundingBox_.isNull()) {
        painter.setPen(QPen(Qt::blue, 2));
        painter.drawRect(xOff + boundingBox_.x() * scaleX, yOff + boundingBox_.y() * scaleY, boundingBox_.width() * scaleX, boundingBox_.height() * scaleY);
    }
    if (!logoPixmap_.isNull() && qtPixmap_.width() > 0) {
        painter.setOpacity(0.7);
        painter.drawPixmap(xOff + qtPixmap_.width() - logoPixmap_.width() - 10, yOff + qtPixmap_.height() - logoPixmap_.height() - 10, logoPixmap_);
    }
}

void VideoWidget::mousePressEvent(QMouseEvent* e) {
    if (!isEditingRoi_ || e->button() != Qt::LeftButton) return;
    isDrawingBox_ = true;
    boxStartPoint_ = dsai_widgetToImageCoordinates(e->position()).toPoint();
}

void VideoWidget::mouseMoveEvent(QMouseEvent* e) {
    if (isDrawingBox_) { boxCurrentPoint_ = dsai_widgetToImageCoordinates(e->position()).toPoint(); boundingBox_ = QRect(boxStartPoint_, boxCurrentPoint_).normalized(); update(); }
}

void VideoWidget::mouseReleaseEvent(QMouseEvent* e) {
    if (isDrawingBox_) { isDrawingBox_ = false; emit boundingBoxChanged(boundingBox_); }
}

QPointF VideoWidget::dsai_widgetToImageCoordinates(const QPointF& widgetPos) const {
    QMutexLocker lock(const_cast<QMutex*>(&displayMutex_));
    if (currentFrame_.empty() || qtPixmap_.isNull()) return widgetPos;
    int xOff = (width() - qtPixmap_.width()) / 2;
    int yOff = (height() - qtPixmap_.height()) / 2;
    return QPointF((widgetPos.x() - xOff) * (float)currentFrame_.cols / qtPixmap_.width(), (widgetPos.y() - yOff) * (float)currentFrame_.rows / qtPixmap_.height());
}
