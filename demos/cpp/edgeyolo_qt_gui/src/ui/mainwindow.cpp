#include "mainwindow.h"

#include <QDateTime>
#include <QFileInfo>
#include <QMessageBox>
#include <QDebug>

#include <chrono>

// ─── DetectionWorker ──────────────────────────────────────────────────────────

DetectionWorker::DetectionWorker(QObject* parent)
    : QThread(parent)
{}

DetectionWorker::~DetectionWorker()
{
    stop();
    wait();
}

void DetectionWorker::setDetector(inference::IDetector* detector)
{
    QMutexLocker lock(&mutex_);
    detector_ = detector;
}

void DetectionWorker::pushFrame(const cv::Mat& frame)
{
    QMutexLocker lock(&mutex_);
    pendingFrame_ = frame.clone();
    frameReady_   = true;
    condition_.wakeOne();
}

void DetectionWorker::setEnabled(bool enabled)
{
    QMutexLocker lock(&mutex_);
    enabled_ = enabled;
    if (enabled_) condition_.wakeOne();
}

void DetectionWorker::stop()
{
    QMutexLocker lock(&mutex_);
    stopped_ = true;
    condition_.wakeOne();
}

void DetectionWorker::run()
{
    fpsWindowStart_ = QDateTime::currentMSecsSinceEpoch();

    while (true) {
        cv::Mat frame;
        inference::IDetector* det = nullptr;

        {
            QMutexLocker lock(&mutex_);
            while (!stopped_ && (!enabled_ || !frameReady_))
                condition_.wait(&mutex_);

            if (stopped_) break;

            frame      = std::move(pendingFrame_);
            frameReady_ = false;
            det        = detector_;
        }

        if (!det || frame.empty()) continue;

        // ── inference ─────────────────────────────────────────────────────
        std::vector<inference::Detection> detections;
        float inferMs = 0.f;
        float nmsMs   = 0.f;

        try {
            using Clock = std::chrono::high_resolution_clock;

            auto t0 = Clock::now();
            detections = det->infer(frame);
            auto t1 = Clock::now();

            inferMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
            // NMS is inside infer() for all backends; we can't split it here.
            // Report nmsMs = 0 unless a backend exposes it separately.
        }
        catch (const std::exception& e) {
            qWarning() << "DetectionWorker: infer() threw:" << e.what();
            continue;
        }

        // ── FPS ───────────────────────────────────────────────────────────
        ++frameCount_;
        const qint64 now    = QDateTime::currentMSecsSinceEpoch();
        const qint64 elapsed = now - fpsWindowStart_;
        float fps = 0.f;
        if (elapsed >= 1000) {
            fps = frameCount_ * 1000.f / static_cast<float>(elapsed);
            frameCount_    = 0;
            fpsWindowStart_ = now;
        }

        // ── rolling average latency ───────────────────────────────────────
        latencyHistory_.push_back(inferMs);
        if (static_cast<int>(latencyHistory_.size()) > kHistorySize)
            latencyHistory_.pop_front();

        float avgMs = 0.f;
        for (float v : latencyHistory_) avgMs += v;
        if (!latencyHistory_.empty())
            avgMs /= static_cast<float>(latencyHistory_.size());

        // ── pack results for GUI ──────────────────────────────────────────
        QVector<QRect>  boxes;
        QVector<int>    classIds;
        QVector<float>  confidences;
        boxes.reserve(static_cast<int>(detections.size()));
        classIds.reserve(boxes.capacity());
        confidences.reserve(boxes.capacity());

        for (const auto& d : detections) {
            boxes.append(QRect(
                static_cast<int>(d.rect.x),
                static_cast<int>(d.rect.y),
                static_cast<int>(d.rect.width),
                static_cast<int>(d.rect.height)
            ));
            classIds.append(d.classId);
            confidences.append(d.confidence);
        }

        emit detectionResultsReady(boxes, classIds, confidences);
        emit performanceMetricsUpdated(fps, inferMs, nmsMs, avgMs);
    }
}

// ─── MainWindow ───────────────────────────────────────────────────────────────

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setupUI();
    setupConnections();
}

MainWindow::~MainWindow()
{
    stopWorker();
    // detector_ destroyed by unique_ptr after worker is confirmed stopped
}

void MainWindow::setupUI()
{
    setWindowTitle("EdgeYOLO Qt GUI");
    resize(1280, 800);

    mainSplitter_ = new QSplitter(Qt::Horizontal, this);
    setCentralWidget(mainSplitter_);

    // ── left: video ───────────────────────────────────────────────────────
    videoWidget_ = new VideoWidget(this);
    mainSplitter_->addWidget(videoWidget_);

    // ── right: control panel ─────────────────────────────────────────────
    controlPanel_ = new QWidget(this);
    controlPanel_->setFixedWidth(300);
    controlLayout_ = new QVBoxLayout(controlPanel_);
    controlLayout_->setSpacing(6);

    configButton_  = new QPushButton("Configure…", this);
    editRunButton_ = new QPushButton("Switch to Run Mode", this);
    editRunButton_->setEnabled(false); // enabled after detector loaded

    controlLayout_->addWidget(configButton_);
    controlLayout_->addWidget(editRunButton_);

    // ── metrics ───────────────────────────────────────────────────────────
    metricsGroup_ = new QGroupBox("Performance Metrics", this);
    QVBoxLayout* mlay = new QVBoxLayout(metricsGroup_);
    fpsLabel_      = new QLabel("FPS: —", this);
    inferLatLabel_ = new QLabel("Inference: — ms", this);
    nmsLatLabel_   = new QLabel("NMS: — ms", this);
    avgLatLabel_   = new QLabel("Avg (30f): — ms", this);
    timeLabel_     = new QLabel("Time: —", this);
    for (QLabel* l : {fpsLabel_, inferLatLabel_, nmsLatLabel_, avgLatLabel_, timeLabel_})
        mlay->addWidget(l);
    controlLayout_->addWidget(metricsGroup_);

    // ── object checkboxes ─────────────────────────────────────────────────
    objectsGroup_  = new QGroupBox("Detectable Objects", this);
    objectsLayout_ = new QVBoxLayout(objectsGroup_);
    objectsLayout_->addWidget(new QLabel("(load a model to populate)", this));
    controlLayout_->addWidget(objectsGroup_);

    controlLayout_->addStretch();
    mainSplitter_->addWidget(controlPanel_);

    // ── status bar ────────────────────────────────────────────────────────
    statusBar_ = new QStatusBar(this);
    setStatusBar(statusBar_);
    statusBar_->showMessage("Ready — click Configure to load a model.");

    // ── clock timer ───────────────────────────────────────────────────────
    clockTimer_ = new QTimer(this);
    clockTimer_->start(1000);
}

void MainWindow::setupConnections()
{
    connect(configButton_,  &QPushButton::clicked, this, &MainWindow::openConfigDialog);
    connect(editRunButton_, &QPushButton::clicked, this, &MainWindow::toggleEditRunMode);
    connect(videoWidget_,   &VideoWidget::boundingBoxChanged,
            this, &MainWindow::handleBoundingBoxChanged);
    connect(clockTimer_, &QTimer::timeout, this, &MainWindow::updateClock);
}

// ─── config ───────────────────────────────────────────────────────────────────

void MainWindow::openConfigDialog()
{
    ConfigDialog dlg(this);
    if (dlg.exec() != QDialog::Accepted) return;

    // Stop any running worker before touching the detector
    stopWorker();

    modelFilePath_    = dlg.getModelFilePath();
    yamlFilePath_     = dlg.getYamlPath();
    isUsingVideoFile_ = dlg.isUsingVideoFile();
    isUsingRtspStream_ = dlg.isUsingRtspStream();

    // Update video source
    if (isUsingVideoFile_ || isUsingRtspStream_) {
        videoWidget_->setVideoSource(dlg.getVideoOrRtspPath());
        statusBar_->showMessage("Source: " + dlg.getVideoOrRtspPath());
    } else {
        videoWidget_->setCameraDevice(dlg.getCameraDeviceId());
        statusBar_->showMessage("Camera device: " + QString::number(dlg.getCameraDeviceId()));
    }

    // Derive ROI config path: same directory as model, named <modelBaseName>_roi.yaml
    {
        QFileInfo fi(modelFilePath_);
        roiConfigPath_ = fi.absolutePath() + "/" + fi.completeBaseName() + "_roi.yaml";
    }

    // Initialise detector
    initializeDetector(dlg.getBackend(), modelFilePath_, yamlFilePath_, 0.25f, 0.45f);

    // Load previously saved ROI (no-op if file absent)
    videoWidget_->loadRoiFromConfig(roiConfigPath_);
}

void MainWindow::initializeDetector(inference::Backend backend,
                                     const QString&     modelPath,
                                     const QString&     yamlPath,
                                     float              confThres,
                                     float              nmsThres)
{
    if (modelPath.isEmpty()) {
        statusBar_->showMessage("No model selected.");
        return;
    }

    try {
        detector_ = inference::DetectorFactory::create(
            backend,
            modelPath.toStdString(),
            yamlPath.toStdString(),
            confThres,
            nmsThres
        );
    }
    catch (const std::exception& e) {
        QMessageBox::critical(this, "Model Load Failed",
            QString("Failed to load model:\n%1").arg(e.what()));
        detector_.reset();
        editRunButton_->setEnabled(false);
        statusBar_->showMessage("Model load failed.");
        return;
    }

    populateClassCheckboxes(detector_->classNames());
    editRunButton_->setEnabled(true);
    statusBar_->showMessage(
        QString("Model loaded (%1, %2 classes).")
        .arg(inference::DetectorFactory::name(backend))
        .arg(static_cast<int>(detector_->classNames().size()))
    );
}

void MainWindow::populateClassCheckboxes(const std::vector<std::string>& names)
{
    // Clear existing
    for (auto* cb : classCheckboxes_) delete cb;
    classCheckboxes_.clear();

    // Remove placeholder label if present
    while (QLayoutItem* item = objectsLayout_->takeAt(0)) {
        delete item->widget();
        delete item;
    }

    for (int i = 0; i < static_cast<int>(names.size()); ++i) {
        auto* cb = new QCheckBox(QString::fromStdString(names[i]), this);
        cb->setChecked(true);
        classCheckboxes_[i] = cb;
        objectsLayout_->addWidget(cb);
    }

    if (names.empty())
        objectsLayout_->addWidget(new QLabel("(no classes)", this));
}

// ─── mode switching ───────────────────────────────────────────────────────────

void MainWindow::toggleEditRunMode()
{
    isEditMode_ = !isEditMode_;

    if (isEditMode_) {
        stopWorker();
        videoWidget_->setEditMode(true);
        editRunButton_->setText("Switch to Run Mode");
        statusBar_->showMessage("Edit Mode — draw region of interest.");
    } else {
        videoWidget_->setEditMode(false);
        editRunButton_->setText("Switch to Edit Mode");
        statusBar_->showMessage("Run Mode — detection active.");
        startWorker();
    }
}

// ─── worker management ────────────────────────────────────────────────────────

void MainWindow::startWorker()
{
    if (!detector_) return;
    if (worker_) return; // already running

    worker_ = new DetectionWorker(this);
    worker_->setDetector(detector_.get());

    connect(worker_, &DetectionWorker::detectionResultsReady,
            this,    &MainWindow::onDetectionResults);
    connect(worker_, &DetectionWorker::performanceMetricsUpdated,
            this,    &MainWindow::onPerformanceMetrics);

    worker_->setEnabled(true);
    worker_->start();

    // Feed frames from VideoWidget
    connect(videoWidget_, &VideoWidget::frameReady,
            worker_,      [this](const cv::Mat& f){ worker_->pushFrame(f); });
}

void MainWindow::stopWorker()
{
    if (!worker_) return;
    worker_->stop();
    worker_->wait(3000);
    delete worker_;
    worker_ = nullptr;

    // Disconnect frame feed
    disconnect(videoWidget_, &VideoWidget::frameReady, nullptr, nullptr);
}

// ─── slots ────────────────────────────────────────────────────────────────────

void MainWindow::onDetectionResults(QVector<QRect>  boxes,
                                     QVector<int>    classIds,
                                     QVector<float>  confidences)
{
    // Filter by unchecked classes
    QVector<QRect>  filteredBoxes;
    QVector<int>    filteredIds;
    QVector<float>  filteredConfs;

    for (int i = 0; i < boxes.size(); ++i) {
        const int id = classIds[i];
        auto it = classCheckboxes_.find(id);
        if (it != classCheckboxes_.end() && !it.value()->isChecked()) continue;
        filteredBoxes.append(boxes[i]);
        filteredIds.append(id);
        filteredConfs.append(confidences[i]);
    }

    videoWidget_->setDetectionResults(filteredBoxes, filteredIds, filteredConfs);
}

void MainWindow::onPerformanceMetrics(float fps, float inferLatency,
                                       float nmsLatency, float avgLatency)
{
    fpsLabel_->setText(     QString("FPS: %1")            .arg(fps,          0, 'f', 1));
    inferLatLabel_->setText(QString("Inference: %1 ms")   .arg(inferLatency, 0, 'f', 1));
    nmsLatLabel_->setText(  QString("NMS: %1 ms")         .arg(nmsLatency,   0, 'f', 1));
    avgLatLabel_->setText(  QString("Avg (30f): %1 ms")   .arg(avgLatency,   0, 'f', 1));
}

void MainWindow::updateClock()
{
    timeLabel_->setText("Time: " + QDateTime::currentDateTime().toString("HH:mm:ss"));
}

void MainWindow::handleBoundingBoxChanged(const QRect& box)
{
    currentBoundingBox_ = box;

    if (box.isNull() || box.isEmpty()) {
        statusBar_->showMessage("ROI cleared.");
    } else {
        statusBar_->showMessage(
            QString("ROI: (%1, %2)  %3 × %4")
            .arg(box.x()).arg(box.y()).arg(box.width()).arg(box.height()));
    }

    // Persist immediately — only if we have a config path (i.e. model is loaded)
    if (roiConfigPath_.isEmpty()) return;

    try {
        videoWidget_->saveRoiToConfig(roiConfigPath_);
    }
    catch (const std::exception& e) {
        qWarning() << "MainWindow: failed to save ROI config:" << e.what();
        statusBar_->showMessage(
            QString("Warning: could not save ROI — %1").arg(e.what()));
    }
}
