#include "videowidget.h"
#include "../debug_log.h"
#include <QPainter>
#include <QMouseEvent>
#include <QApplication>
#include <QScreen>
#include <QDateTime>
#include <QDebug>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Component tag used with the shared DBG_LOG / ERR_LOG macros.
#define VW_TAG "VideoWidget"

void VideoWidget::setDebugLogging(bool enabled) { Debug::setEnabled(enabled); }

VideoWidget::VideoWidget(QWidget *parent) : QWidget(parent),
    running_(false),
    cameraDeviceId_(0),
    isEditMode_(false),
    isDrawingBox_(false),
    metersToPixelsRatio_(100.0f), // Default: 100 pixels per meter (will be calibrated)
    referenceWidth_(640) // Reference width for calibration
{
    setMinimumSize(640, 480);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    logoPixmap_.load(":/assets/logo.png");

    // Start video capture thread
    startCaptureThread();
}

VideoWidget::~VideoWidget() {
    stopCaptureThread();
}

void VideoWidget::setCameraDevice(int deviceId) {
    stopCaptureThread();
    videoSourcePath_.clear();
    cameraDeviceId_ = deviceId;
    startCaptureThread();
}

void VideoWidget::setVideoSource(const QString& path) {
    stopCaptureThread();
    videoSourcePath_ = path;
    startCaptureThread();
}

void VideoWidget::setEditMode(bool editMode) {
    isEditMode_ = editMode;
    if (!isEditMode_)
        isDrawingBox_ = false;  // stop in-progress draw; preserve completed ROI
    update();
}

void VideoWidget::setBoundingBox(const QRect& box) {
    {
        QMutexLocker locker(&frameMutex_);
        boundingBox_ = box;
    }
    update(); // Trigger repaint
}

QRect VideoWidget::getBoundingBox() const {
    QMutexLocker locker(&frameMutex_);
    return boundingBox_;
}

void VideoWidget::setDetectionResults(const QVector<QRect>& boxes,
                                     const QVector<int>& classIds,
                                     const QVector<float>& confidences) {
    {
        QMutexLocker locker(&resultsMutex_);
        detectionBoxes_ = boxes;
        detectionClassIds_ = classIds;
        detectionConfidences_ = confidences;
    }
    update(); // Trigger repaint
}

void VideoWidget::setClassNames(const QStringList& names)
{
    QMutexLocker locker(&resultsMutex_);
    classNames_ = names;
}

void VideoWidget::loadRoiFromConfig(const QString& configPath)
{
    if (configPath.isEmpty()) return;

    const std::string path = configPath.toStdString();

    // File may not exist yet (first run) — silently ignore
    {
        std::ifstream probe(path);
        if (!probe.good()) return;
    }

    try {
        const YAML::Node cfg = YAML::LoadFile(path);

        if (cfg["roi"]) {
            const YAML::Node& roi = cfg["roi"];
            // Stored as integer pixel coordinates
            const int x = roi["x"].as<int>(0);
            const int y = roi["y"].as<int>(0);
            const int w = roi["width"].as<int>(0);
            const int h = roi["height"].as<int>(0);

            if (w > 0 && h > 0) {
                const QRect box(x, y, w, h);
                DBG_LOG(VW_TAG, "loaded ROI from '%s': (%d,%d) %dx%d\n",
                    path.c_str(), x, y, w, h);
                {
                    QMutexLocker locker(&frameMutex_);
                    boundingBox_ = box;
                }
                emit boundingBoxChanged(box);
                update();
            } else {
                DBG_LOG(VW_TAG, "roi key present in '%s' but w/h invalid (%dx%d) — skipped\n",
                    path.c_str(), w, h);
            }
        } else {
            DBG_LOG(VW_TAG, "no roi key in '%s'\n", path.c_str());
        }

        // Optionally refresh class names list from the same file
        if (cfg["names"]) {
            classNames_.clear();
            for (const auto& n : cfg["names"])
                classNames_.append(QString::fromStdString(n.as<std::string>()));
            DBG_LOG(VW_TAG, "loaded %lld class names from '%s'\n",
                static_cast<long long>(classNames_.size()), path.c_str());
        }
    }
    catch (const YAML::Exception& e) {
        ERR_LOG(VW_TAG, "loadRoiFromConfig YAML error in '%s': %s\n", path.c_str(), e.what());
    }
}

void VideoWidget::saveRoiToConfig(const QString& configPath)
{
    if (configPath.isEmpty())
        throw std::runtime_error("VideoWidget::saveRoiToConfig: config path is empty");

    QRect box;
    {
        QMutexLocker locker(&frameMutex_);
        box = boundingBox_;
    }

    const std::string path = configPath.toStdString();

    // Load existing YAML so we only update the roi key and preserve everything else
    YAML::Node cfg;
    {
        std::ifstream probe(path);
        if (probe.good()) {
            try {
                cfg = YAML::LoadFile(path);
            }
            catch (const YAML::Exception& e) {
                throw std::runtime_error(
                    std::string("VideoWidget::saveRoiToConfig: failed to parse existing YAML: ")
                    + e.what());
            }
        }
    }

    if (box.isNull() || box.isEmpty()) {
        // No ROI — remove the key if it exists
        cfg.remove("roi");
    } else {
        // Store pixel coordinates as plain integers — no normalisation needed;
        // coordinates are in the original capture frame space, not widget space.
        cfg["roi"]["x"]      = box.x();
        cfg["roi"]["y"]      = box.y();
        cfg["roi"]["width"]  = box.width();
        cfg["roi"]["height"] = box.height();
    }

    std::ofstream fout(path);
    if (!fout.is_open())
        throw std::runtime_error(
            "VideoWidget::saveRoiToConfig: cannot open file for writing: " + path);

    fout << cfg;
    if (!fout)
        throw std::runtime_error(
            "VideoWidget::saveRoiToConfig: write error to: " + path);

    DBG_LOG(VW_TAG, "saved ROI (%d,%d %dx%d) to '%s'\n",
        box.x(), box.y(), box.width(), box.height(), path.c_str());
}

void VideoWidget::updateFrame() {
    cv::Mat frame;
    {
        QMutexLocker locker(&frameMutex_);
        if (!currentFrame_.empty())
            frame = currentFrame_.clone();
    }

    if (!frame.empty())
        currentFrame_ = frame;

    if (currentFrame_.empty())
        return;

    // ── BGR → RGB conversion ──────────────────────────────────────────────────
    cv::cvtColor(currentFrame_, displayFrame_, cv::COLOR_BGR2RGB);

    // ── draw overlays ─────────────────────────────────────────────────────────
    {
        QMutexLocker locker(&resultsMutex_);

        // Stable per-class BGR colours (cycles for > 8 classes)
        static const cv::Scalar kColors[] = {
            {  0, 255,   0}, {  0,   0, 255}, {255,   0,   0},
            {  0, 255, 255}, {255,   0, 255}, {255, 128,   0},
            {128,   0, 255}, {  0, 128, 255},
        };
        constexpr int kNC = static_cast<int>(sizeof(kColors) / sizeof(kColors[0]));

        for (int i = 0; i < static_cast<int>(detectionBoxes_.size()); ++i) {
            const QRect&  box        = detectionBoxes_[i];
            const int     classId    = detectionClassIds_[i];
            const float   confidence = detectionConfidences_[i];
            const cv::Scalar color   = kColors[(classId >= 0 ? classId : 0) % kNC];

            cv::rectangle(displayFrame_,
                          cv::Point(box.x(), box.y()),
                          cv::Point(box.x() + box.width(), box.y() + box.height()),
                          color, 2);

            std::string label;
            if (classId >= 0 && classId < static_cast<int>(classNames_.size()))
                label = classNames_[classId].toStdString() + ": " +
                        std::to_string(confidence).substr(0, 4);
            else
                label = "id" + std::to_string(classId) + ": " +
                        std::to_string(confidence).substr(0, 4);

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(displayFrame_,
                          cv::Point(box.x(), box.y() - textSize.height - baseline),
                          cv::Point(box.x() + textSize.width, box.y()),
                          color, cv::FILLED);
            cv::putText(displayFrame_, label,
                        cv::Point(box.x(), box.y() - baseline),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }

    // ── ROI bounding box — always visible when set (edit and run modes) ───────
    {
        QRect box;
        {
            QMutexLocker fl(&frameMutex_);
            box = boundingBox_;
        }
        if (!box.isEmpty())
            cv::rectangle(displayFrame_,
                          cv::Point(box.x(), box.y()),
                          cv::Point(box.x() + box.width(), box.y() + box.height()),
                          cv::Scalar(BOX_COLOR_B, BOX_COLOR_G, BOX_COLOR_R),
                          BOX_WIDTH);
    }

    // ── in-progress draw preview (edit mode only) ─────────────────────────────
    if (isEditMode_ && isDrawingBox_) {
        const QPoint start   = boxStartPoint_;
        const QPoint current = boxCurrentPoint_;
        const int x = std::min(start.x(), current.x());
        const int y = std::min(start.y(), current.y());
        const int w = std::abs(start.x() - current.x());
        const int h = std::abs(start.y() - current.y());
        if (w > 0 && h > 0)
            cv::rectangle(displayFrame_,
                          cv::Point(x, y), cv::Point(x + w, y + h),
                          cv::Scalar(BOX_COLOR_B, BOX_COLOR_G, BOX_COLOR_R),
                          BOX_WIDTH);
    }

    // ── store QImage (thread-safe) ────────────────────────────────────────────
    // .copy() detaches the QImage from displayFrame_'s data buffer so it can be
    // safely read on the GUI thread after displayFrame_ is overwritten next frame.
    {
        QMutexLocker displayLock(&displayMutex_);
        qtImage_ = QImage(displayFrame_.data,
                          displayFrame_.cols,
                          displayFrame_.rows,
                          static_cast<int>(displayFrame_.step),
                          QImage::Format_RGB888).copy();
    }

    // ── request repaint on the GUI thread ─────────────────────────────────────
    // Never call QWidget::update() directly from a non-GUI thread.
    QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);
}

void VideoWidget::startCaptureThread() {
    running_ = true;
    const int     camId  = cameraDeviceId_;
    const QString vsPath = videoSourcePath_;

    const bool rknn = rockchipHw_;

    captureThread_ = QThread::create([this, camId, vsPath, rknn]() {
        if (vsPath.isEmpty()) {
            if (rknn) {
                // ── Rockchip camera path (RV1106 / MIPI / V4L2 NPU-aware) ──────
                // TODO: initialise Rockchip camera pipeline here
            } else {
                // ── Standard camera path ─────────────────────────────────────
                capture_.open(camId, cv::CAP_V4L2);
                if (!capture_.isOpened())
                    capture_.open(camId);  // fallback to default backend
            }
        } else {
            if (rknn) {
                // ── Rockchip RTSP path (hardware-decoded RTSP on RV1106) ─────
                // TODO: initialise Rockchip RTSP pipeline here
            } else {
                // ── Standard RTSP / video file path ──────────────────────────
                capture_.open(vsPath.toStdString());
            }
        }

        if (!capture_.isOpened()) {
            ERR_LOG(VW_TAG, "failed to open source: %s\n",
                vsPath.isEmpty()
                    ? QString("camera %1").arg(camId).toStdString().c_str()
                    : vsPath.toStdString().c_str());
            return;
        }

        // Apply camera properties only for real camera sources
        if (vsPath.isEmpty()) {
            capture_.set(cv::CAP_PROP_FRAME_WIDTH,  640);
            capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            capture_.set(cv::CAP_PROP_FPS,          30);
        }

        // Determine per-frame target delay.
        // For cameras, read() blocks in V4L2 at the hardware rate — use a minimal
        // yield sleep so we don't busy-spin but don't add extra delay.
        // For video files / RTSP, honour the container FPS so playback is real-time.
        int frameDelayMs = 1;   // cameras: near-zero extra sleep
        if (!vsPath.isEmpty()) {
            const double srcFps = capture_.get(cv::CAP_PROP_FPS);
            if (srcFps > 0.0 && srcFps <= 240.0)
                frameDelayMs = static_cast<int>(1000.0 / srcFps);
            else
                frameDelayMs = 33;  // fallback ~30 fps
            DBG_LOG(VW_TAG, "opened '%s'  reported FPS=%.2f  frameDelayMs=%d\n",
                vsPath.toStdString().c_str(), srcFps, frameDelayMs);
        } else {
            const double srcFps = capture_.get(cv::CAP_PROP_FPS);
            const int    w      = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
            const int    h      = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
            DBG_LOG(VW_TAG, "opened camera %d  %dx%d  reported FPS=%.2f\n", camId, w, h, srcFps);
        }

        using Clock     = std::chrono::steady_clock;
        using Ms        = std::chrono::milliseconds;
        uint64_t frameN = 0;

        while (running_) {
            const auto frameStart = Clock::now();

            cv::Mat frame;
            if (!capture_.read(frame) || frame.empty()) {
                if (!vsPath.isEmpty()) {
                    // End of video file — loop back
                    DBG_LOG(VW_TAG, "end of file, looping back (frame %llu)\n",
                        static_cast<unsigned long long>(frameN));
                    capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
                } else {
                    ERR_LOG(VW_TAG, "failed to read frame from camera %d\n", camId);
                }
                QThread::msleep(10);
                continue;
            }

            ++frameN;

            {
                QMutexLocker locker(&frameMutex_);
                currentFrame_ = frame.clone();
            }

            // Emit raw frame for the detection worker (BGR, original resolution)
            emit frameReady(frame);

            updateFrame();

            if (frameN % 90 == 0) {
                DBG_LOG(VW_TAG, "capture heartbeat: frame %llu\n",
                    static_cast<unsigned long long>(frameN));
            }

            // Sleep for the remainder of the target frame period
            const auto elapsed = std::chrono::duration_cast<Ms>(Clock::now() - frameStart).count();
            const int  sleepMs = frameDelayMs - static_cast<int>(elapsed);
            if (sleepMs > 0)
                QThread::msleep(static_cast<unsigned long>(sleepMs));
        }

        capture_.release();
        DBG_LOG(VW_TAG, "capture thread exited after %llu frames\n",
            static_cast<unsigned long long>(frameN));
    });
    captureThread_->start();
}

void VideoWidget::stopCaptureThread() {
    running_ = false;
    frameCondition_.wakeOne();
    if (captureThread_) {
        captureThread_->wait();
        delete captureThread_;
        captureThread_ = nullptr;
    }
}

void VideoWidget::paintEvent(QPaintEvent *event) {
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

    // COW copy under lock — detaches from the capture thread's buffer.
    // QPixmap::fromImage() and scaled() must run on the GUI thread.
    QImage img;
    {
        QMutexLocker displayLock(&displayMutex_);
        img = qtImage_;
    }

    if (!img.isNull()) {
        qtPixmap_      = QPixmap::fromImage(img).scaled(
                             size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        lastFrameSize_ = img.size();   // cache for coordinate conversion in mouse events

        const int x = (width()  - qtPixmap_.width())  / 2;
        const int y = (height() - qtPixmap_.height()) / 2;
        painter.drawPixmap(x, y, qtPixmap_);

        // Logo overlay — bottom-right corner, 70% opacity
        if (!logoPixmap_.isNull()) {
            constexpr int kPadding = 10;
            const int lx = x + qtPixmap_.width()  - logoPixmap_.width()  - kPadding;
            const int ly = y + qtPixmap_.height() - logoPixmap_.height() - kPadding;
            painter.setOpacity(0.70);
            painter.drawPixmap(lx, ly, logoPixmap_);
            painter.setOpacity(1.0);
        }
    } else {
        painter.setPen(Qt::white);
        painter.setFont(QFont("Arial", 16));
        painter.drawText(rect(), Qt::AlignCenter, "No Video Signal");
    }
}

void VideoWidget::mousePressEvent(QMouseEvent *event) {
    if (!isEditMode_ || event->button() != Qt::LeftButton) {
        QWidget::mousePressEvent(event);
        return;
    }

    // Convert widget coordinates to image coordinates
    QPointF widgetPos = event->position();
    QPointF imagePos = widgetToImageCoordinates(widgetPos);

    isDrawingBox_ = true;
    boxStartPoint_   = imagePos.toPoint();
    boxCurrentPoint_ = boxStartPoint_;
    {
        QMutexLocker locker(&frameMutex_);
        boundingBox_ = QRect(boxStartPoint_, QSize());
    }
    emit boundingBoxChanged(QRect(boxStartPoint_, QSize()));
    update();
}

void VideoWidget::mouseMoveEvent(QMouseEvent *event) {
    if (!isEditMode_ || !isDrawingBox_) {
        QWidget::mouseMoveEvent(event);
        return;
    }

    QPointF widgetPos = event->position();
    QPointF imagePos = widgetToImageCoordinates(widgetPos);
    boxCurrentPoint_ = imagePos.toPoint();

    const int x = std::min(boxStartPoint_.x(), boxCurrentPoint_.x());
    const int y = std::min(boxStartPoint_.y(), boxCurrentPoint_.y());
    const int w = std::abs(boxStartPoint_.x() - boxCurrentPoint_.x());
    const int h = std::abs(boxStartPoint_.y() - boxCurrentPoint_.y());
    const QRect newBox(x, y, w, h);
    {
        QMutexLocker locker(&frameMutex_);
        boundingBox_ = newBox;
    }
    emit boundingBoxChanged(newBox);
    update();
}

void VideoWidget::mouseReleaseEvent(QMouseEvent *event) {
    if (!isEditMode_ || event->button() != Qt::LeftButton || !isDrawingBox_) {
        QWidget::mouseReleaseEvent(event);
        return;
    }

    isDrawingBox_ = false;

    // Finalise bounding box
    const QPointF widgetPos = event->position();
    const QPointF imagePos  = widgetToImageCoordinates(widgetPos);
    boxCurrentPoint_ = imagePos.toPoint();

    const int x = std::min(boxStartPoint_.x(), boxCurrentPoint_.x());
    const int y = std::min(boxStartPoint_.y(), boxCurrentPoint_.y());
    const int w = std::abs(boxStartPoint_.x() - boxCurrentPoint_.x());
    const int h = std::abs(boxStartPoint_.y() - boxCurrentPoint_.y());

    QRect finalBox;
    if (w > 10 && h > 10)
        finalBox = QRect(x, y, w, h);   // accepted; else stays null (clear)

    {
        QMutexLocker locker(&frameMutex_);
        boundingBox_ = finalBox;
    }
    emit boundingBoxChanged(finalBox);
    update();
}

QPointF VideoWidget::widgetToImageCoordinates(const QPointF& widgetPos) {
    // qtPixmap_ and lastFrameSize_ are only written in paintEvent() — GUI thread — so
    // reading them here (also GUI thread, mouse events) needs no extra locking.
    if (qtPixmap_.isNull() || lastFrameSize_.isEmpty())
        return widgetPos;

    if (qtPixmap_.width() == 0 || qtPixmap_.height() == 0)
        return widgetPos;

    const int   pixmapX   = (width()  - qtPixmap_.width())  / 2;
    const int   pixmapY   = (height() - qtPixmap_.height()) / 2;
    const qreal adjustedX = widgetPos.x() - pixmapX;
    const qreal adjustedY = widgetPos.y() - pixmapY;

    // lastFrameSize_ is the original frame resolution; qtPixmap_ is the scaled version.
    const qreal scaleX = static_cast<qreal>(lastFrameSize_.width())  / qtPixmap_.width();
    const qreal scaleY = static_cast<qreal>(lastFrameSize_.height()) / qtPixmap_.height();

    const qreal imageX = std::max(0.0, std::min(adjustedX * scaleX,
                                                 static_cast<qreal>(lastFrameSize_.width())));
    const qreal imageY = std::max(0.0, std::min(adjustedY * scaleY,
                                                 static_cast<qreal>(lastFrameSize_.height())));
    return QPointF(imageX, imageY);
}