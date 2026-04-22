#include "videowidget.h"
#include "../debug_log.h"
#include <QPainter>
#include <QMouseEvent>
#include <QMutexLocker>
#include <yaml-cpp/yaml.h>
#include <fstream>

#define TAG "VideoWidget"

VideoWidget::VideoWidget(QWidget* parent) : QWidget(parent) {
    logoPixmap_.load(":/assets/logo.png");
}

VideoWidget::~VideoWidget() { dsai_stopCaptureThread(); }

void VideoWidget::dsai_setCameraDevice(int devId) {
    dsai_stopCaptureThread();
    dsai_startCaptureThread();
}

void VideoWidget::dsai_setVideoSource(const QString& path) {
    dsai_stopCaptureThread();
    dsai_startCaptureThread();
}

void VideoWidget::dsai_setRockchipHardware(bool enabled) { isRockchip_ = enabled; }

void VideoWidget::dsai_setClassNames(const QStringList& names) { classNames_ = names; }

void VideoWidget::dsai_setDetectionResults(const std::vector<inference::Detection>& results) {
    std::lock_guard<QMutex> lock(displayMutex_);
    detections_ = results;
    if (isRockchip_ && capture_) capture_->dsai_setOSD(results);
    update();
}

void VideoWidget::dsai_loadRoiFromConfig(const QString& path) {
    try {
        YAML::Node cfg = YAML::LoadFile(path.toStdString());
        if (cfg["roi"]) {
            auto r = cfg["roi"];
            currentRoi_ = QRect(r["x"].as<int>(), r["y"].as<int>(), r["width"].as<int>(), r["height"].as<int>());
            update();
        }
    } catch(...) {}
}

void VideoWidget::dsai_saveRoiToConfig(const QString& path) {
    YAML::Node cfg;
    cfg["roi"]["x"] = currentRoi_.x();
    cfg["roi"]["y"] = currentRoi_.y();
    cfg["roi"]["width"] = currentRoi_.width();
    cfg["roi"]["height"] = currentRoi_.height();
    std::ofstream fout(path.toStdString());
    fout << cfg;
}

void VideoWidget::dsai_setEditMode(bool edit) { isEditingRoi_ = edit; isDrawingBox_ = false; update(); }
void VideoWidget::dsai_clearRoi() { currentRoi_ = QRect(); update(); }

void VideoWidget::dsai_startCaptureThread() {
    captureThread_ = QThread::create([this]() {
        auto cap = deepSightAI::CaptureFactory::dsai_create();
        if (!cap->dsai_openCamera(0, 640, 480, 30.0)) return;
        { QMutexLocker lock(&captureMutex_); capture_ = std::move(cap); }
        while(!QThread::currentThread()->isInterruptionRequested()) {
            cv::Mat frame;
            if (capture_->dsai_read(frame)) {
                { QMutexLocker lock(&displayMutex_); currentFrame_ = frame.clone(); }
                emit frameReady(frame);
            }
            QThread::msleep(10);
        }
    });
    captureThread_->start();
}

void VideoWidget::dsai_stopCaptureThread() {
    if (captureThread_) {
        captureThread_->requestInterruption();
        captureThread_->wait();
        delete captureThread_;
        captureThread_ = nullptr;
    }
}

void VideoWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    std::lock_guard<QMutex> lock(displayMutex_);
    if (currentFrame_.empty()) return;
    
    QImage img((const uchar*)currentFrame_.data, currentFrame_.cols, currentFrame_.rows, currentFrame_.step, QImage::Format_BGR888);
    qtPixmap_ = QPixmap::fromImage(img).scaled(size(), Qt::KeepAspectRatio);
    lastFrameSize_ = currentFrame_.size();
    
    int x = (width() - qtPixmap_.width()) / 2;
    int y = (height() - qtPixmap_.height()) / 2;
    painter.drawPixmap(x, y, qtPixmap_);

    painter.setPen(Qt::red);
    if (!currentRoi_.isNull()) painter.drawRect(currentRoi_); // Simplified mapping
}

void VideoWidget::mousePressEvent(QMouseEvent* e) {
    if (!isEditingRoi_ || e->button() != Qt::LeftButton) return;
    isDrawingBox_ = true;
    boxStartPoint_ = e->pos();
}

void VideoWidget::mouseMoveEvent(QMouseEvent* e) {
    if (isDrawingBox_) { boxCurrentPoint_ = e->pos(); update(); }
}

void VideoWidget::mouseReleaseEvent(QMouseEvent* e) {
    if (isDrawingBox_) {
        isDrawingBox_ = false;
        currentRoi_ = QRect(boxStartPoint_, e->pos()).normalized();
        emit boundingBoxChanged(currentRoi_);
    }
}

QPointF VideoWidget::dsai_widgetToImageCoordinates(const QPointF& p) const { return p; } // Placeholder
