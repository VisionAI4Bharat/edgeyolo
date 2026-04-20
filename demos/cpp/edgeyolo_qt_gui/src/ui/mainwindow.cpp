#include "mainwindow.h"
#include "../debug_log.h"

#include <QDateTime>
#include <QFileInfo>
#include <QMessageBox>
#include <QDebug>

#include <chrono>
#include <yaml-cpp/yaml.h>

#define MW_TAG  "MainWindow"
#define WK_TAG  "DetWorker"

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
    DBG_LOG(WK_TAG, "thread started\n");

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

        if (!det) {
            ERR_LOG(WK_TAG, "no detector set — dropping frame\n");
            continue;
        }
        if (frame.empty()) {
            ERR_LOG(WK_TAG, "received empty frame — dropping\n");
            continue;
        }

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
            DBG_LOG(WK_TAG, "infer done: %zu detections in %.1f ms\n",
                detections.size(), static_cast<double>(inferMs));
        }
        catch (const std::exception& e) {
            ERR_LOG(WK_TAG, "infer() threw: %s\n", e.what());
            continue;
        }

        // ── FPS — EMA of instantaneous 1/latency; updates every frame ────
        if (inferMs > 0.f) {
            const float instantFps = 1000.f / inferMs;
            emaFps_ = (emaFps_ == 0.f)
                ? instantFps
                : kFpsAlpha * instantFps + (1.f - kFpsAlpha) * emaFps_;
        }
        const float fps = emaFps_;

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
    setWindowTitle("deepSightAI");
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

    configButton_ = new QPushButton("Configure…", this);
    controlLayout_->addWidget(configButton_);

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
    connect(configButton_, &QPushButton::clicked, this, &MainWindow::openConfigDialog);
    connect(videoWidget_,  &VideoWidget::boundingBoxChanged,
            this, &MainWindow::handleBoundingBoxChanged);
    connect(clockTimer_, &QTimer::timeout, this, &MainWindow::updateClock);
}

// ─── config ───────────────────────────────────────────────────────────────────

void MainWindow::loadFromConfigFile(const QString& configPath)
{
    YAML::Node cfg;
    try {
        cfg = YAML::LoadFile(configPath.toStdString());
    } catch (const YAML::Exception& e) {
        QMessageBox::critical(this, "Config Error",
            QString("Failed to parse config file:\n%1\n\n%2").arg(configPath).arg(e.what()));
        return;
    }

    stopWorker();

    debugLogging_ = cfg["debug_logging"].as<bool>(false);
    Debug::setEnabled(debugLogging_);

    modelFilePath_ = QString::fromStdString(cfg["model_file"].as<std::string>(""));
    yamlFilePath_  = QString::fromStdString(cfg["yaml_file"].as<std::string>(""));

    const int srcId = cfg["source"].as<int>(0);
    isUsingVideoFile_  = (srcId == 1);
    isUsingRtspStream_ = (srcId == 2);

    if (isUsingVideoFile_ || isUsingRtspStream_) {
        const QString path = isUsingVideoFile_
            ? QString::fromStdString(cfg["video_file"].as<std::string>(""))
            : QString::fromStdString(cfg["rtsp_url"].as<std::string>(""));
        videoWidget_->setVideoSource(path);
        statusBar_->showMessage("Source: " + path);
    } else {
        const int camId = cfg["camera_device_id"].as<int>(0);
        videoWidget_->setCameraDevice(camId);
        statusBar_->showMessage("Camera device: " + QString::number(camId));
    }

    {
        QFileInfo fi(modelFilePath_);
        roiConfigPath_ = fi.absolutePath() + "/" + fi.completeBaseName() + "_roi.yaml";
    }

    const bool rockchipHw = cfg["rockchip_hw"].as<bool>(false);
    videoWidget_->setRockchipHardware(rockchipHw);
    inference::Backend backend = rockchipHw
        ? inference::Backend::RKNN
        : static_cast<inference::Backend>(cfg["backend"].as<int>(0));

    const float confThres = static_cast<float>(cfg["conf_threshold"].as<double>(0.25));
    const float nmsThres  = static_cast<float>(cfg["nms_threshold"].as<double>(0.45));

    QStringList classLabels;
    if (cfg["class_labels"])
        for (const auto& n : cfg["class_labels"])
            classLabels << QString::fromStdString(n.as<std::string>());

    initializeDetector(backend, modelFilePath_, yamlFilePath_,
                       confThres, nmsThres, classLabels);

    videoWidget_->loadRoiFromConfig(roiConfigPath_);
}

void MainWindow::openConfigDialog()
{
    ConfigDialog dlg(this);
    if (dlg.exec() != QDialog::Accepted) return;

    // Apply debug logging flag first so subsequent calls already use it
    debugLogging_ = dlg.isDebugLoggingEnabled();
    Debug::setEnabled(debugLogging_);
    DBG_LOG(MW_TAG, "config accepted — debug logging %s\n",
        debugLogging_ ? "ON" : "OFF");

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

    const bool rockchipHw = dlg.isRockchipHardware();
    videoWidget_->setRockchipHardware(rockchipHw);

    // If Rockchip HW is selected, force RKNN backend
    inference::Backend backend = rockchipHw
        ? inference::Backend::RKNN
        : dlg.getBackend();

    // Initialise detector
    initializeDetector(backend, modelFilePath_, yamlFilePath_,
                       dlg.getConfThreshold(), dlg.getNmsThreshold(),
                       dlg.getClassLabels());

    // Load previously saved ROI (no-op if file absent)
    videoWidget_->loadRoiFromConfig(roiConfigPath_);
}

void MainWindow::initializeDetector(inference::Backend  backend,
                                     const QString&      modelPath,
                                     const QString&      yamlPath,
                                     float               confThres,
                                     float               nmsThres,
                                     const QStringList&  classLabels)
{
    if (modelPath.isEmpty()) {
        statusBar_->showMessage("No model selected.");
        return;
    }

    DBG_LOG(MW_TAG, "loading model: %s  yaml: %s  conf=%.2f  nms=%.2f\n",
        modelPath.toStdString().c_str(),
        yamlPath.isEmpty() ? "(auto)" : yamlPath.toStdString().c_str(),
        static_cast<double>(confThres), static_cast<double>(nmsThres));

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
        ERR_LOG(MW_TAG, "model load failed: %s\n", e.what());
        QMessageBox::critical(this, "Model Load Failed",
            QString("Failed to load model:\n%1").arg(e.what()));
        detector_.reset();
        statusBar_->showMessage("Model load failed.");

        // Stop the video capture thread because model failed to load
        videoWidget_->stopCaptureThread();

        return;
    }

    if (!classLabels.isEmpty()) {
        std::vector<std::string> labels;
        labels.reserve(classLabels.size());
        for (const QString& s : classLabels)
            labels.push_back(s.toStdString());
        detector_->setClassLabels(labels);
    }

    DBG_LOG(MW_TAG, "model loaded OK — %zu classes  input %dx%d\n",
        detector_->classNames().size(),
        detector_->inputSize().width, detector_->inputSize().height);

    const auto& names = detector_->classNames();
    populateClassCheckboxes(names);

    QStringList qnames;
    qnames.reserve(static_cast<int>(names.size()));
    for (const auto& n : names)
        qnames.append(QString::fromStdString(n));
    videoWidget_->setClassNames(qnames);

    videoWidget_->setEditMode(false);
    startWorker();

    statusBar_->showMessage(
        QString("Model loaded (%1, %2 classes) — detection running.")
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

// ─── worker management ────────────────────────────────────────────────────────

void MainWindow::startWorker()
{
    if (!detector_) {
        ERR_LOG(MW_TAG, "startWorker called but no detector — aborting\n");
        return;
    }
    if (worker_) return; // already running

    DBG_LOG(MW_TAG, "starting detection worker\n");
    worker_ = new DetectionWorker(this);
    worker_->setDetector(detector_.get());

    connect(worker_, &DetectionWorker::detectionResultsReady,
            this,    &MainWindow::onDetectionResults);
    connect(worker_, &DetectionWorker::performanceMetricsUpdated,
            this,    &MainWindow::onPerformanceMetrics);

    worker_->setEnabled(true);
    worker_->start();

    // Feed frames from VideoWidget.
    // Qt::DirectConnection is required here: frameReady is emitted from the
    // capture thread (QThread::create lambda), so the default auto-connection
    // would be queued. Queued connections copy their arguments, which requires
    // cv::Mat to be a registered Qt meta-type — it isn't — so the event would
    // be silently dropped and pushFrame() would never be called.
    // DirectConnection runs the lambda in the capture thread directly;
    // pushFrame() is mutex-protected so this is thread-safe.
    connect(videoWidget_, &VideoWidget::frameReady,
            worker_,      [this](const cv::Mat& f){ worker_->pushFrame(f); },
            Qt::DirectConnection);
}

void MainWindow::stopWorker()
{
    if (!worker_) return;
    DBG_LOG(MW_TAG, "stopping detection worker\n");
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
    DBG_LOG(MW_TAG, "onDetectionResults: %lld raw boxes\n", static_cast<long long>(boxes.size()));
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
