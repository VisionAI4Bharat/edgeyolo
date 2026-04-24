#include "mainwindow.h"
#include "../app_config.h"
#include "../debug_log.h"
#include <QDateTime>
#include <QFileInfo>
#include <QMessageBox>
#include <QStyle>
#include <map>

#define TAG "MainWindow"

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    dsai_setupUI();
    dsai_setupConnections();
    resize(1280, 800);
}

MainWindow::~MainWindow() { dsai_stopWorker(); }

void MainWindow::dsai_setupUI() {
    QWidget* central = new QWidget(this);
    setCentralWidget(central);
    QVBoxLayout* mainLayout = new QVBoxLayout(central);

    QSplitter* splitter = new QSplitter(Qt::Horizontal, central);
    videoWidget_ = new VideoWidget(splitter);
    
    QWidget* rightPanel = new QWidget(splitter);
    QVBoxLayout* rightLayout = new QVBoxLayout(rightPanel);
    
    QGroupBox* infoGroup = new QGroupBox("Performance", rightPanel);
    QVBoxLayout* infoLayout = new QVBoxLayout(infoGroup);
    fpsLabel_ = new QLabel("FPS: —", this);
    inferLatLabel_ = new QLabel("Inference: — ms", this);
    nmsLatLabel_ = new QLabel("NMS: — ms", this);
    avgLatLabel_ = new QLabel("Average: — ms", this);
    for (auto* l : {fpsLabel_, inferLatLabel_, nmsLatLabel_, avgLatLabel_}) {
        l->setStyleSheet("font-family: monospace; font-size: 11pt;");
        infoLayout->addWidget(l);
    }
    rightLayout->addWidget(infoGroup);

    objectsGroup_ = new QGroupBox("Detectable Class Filter", rightPanel);
    objectsLayout_ = new QVBoxLayout(objectsGroup_);
    objectsLayout_->addWidget(new QLabel("(Start to populate)", this));
    rightLayout->addWidget(objectsGroup_);
    
    QPushButton* cfgBtn = new QPushButton("Configure", rightPanel);
    connect(cfgBtn, &QPushButton::clicked, this, &MainWindow::dsai_openConfigDialog);
    rightLayout->addWidget(cfgBtn);

    startStopBtn_ = new QPushButton("Start", rightPanel);
    startStopBtn_->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    startStopBtn_->setMinimumHeight(40);
    connect(startStopBtn_, &QPushButton::clicked, this, &MainWindow::dsai_toggleStartStop);
    rightLayout->addWidget(startStopBtn_);

    rightLayout->addStretch();

    splitter->addWidget(videoWidget_);
    splitter->addWidget(rightPanel);
    splitter->setStretchFactor(0, 4);
    splitter->setStretchFactor(1, 1);
    mainLayout->addWidget(splitter);
    
    statusBar_ = new QStatusBar(this);
    setStatusBar(statusBar_);
    clockLabel_ = new QLabel();
    statusBar_->addPermanentWidget(clockLabel_);
}

void MainWindow::dsai_setupConnections() {
    connect(videoWidget_, &VideoWidget::frameReady, this, [this](const cv::Mat& frame) {
        if (worker_) worker_->dsai_pushFrame(frame);
    });
}

void MainWindow::dsai_populateClassCheckboxes(const std::vector<std::string>& names) {
    for (auto* cb : classCheckboxes_) delete cb;
    classCheckboxes_.clear();
    while (QLayoutItem* item = objectsLayout_->takeAt(0)) {
        if (item->widget()) delete item->widget();
        delete item;
    }
    for (int i = 0; i < (int)names.size(); i++) {
        auto* cb = new QCheckBox(QString::fromStdString(names[i]), this);
        cb->setChecked(true);
        classCheckboxes_[i] = cb;
        objectsLayout_->addWidget(cb);
    }
    if (names.empty()) objectsLayout_->addWidget(new QLabel("(no classes)"));
}

void MainWindow::dsai_toggleStartStop() {
    if (isRunning_) {
        dsai_stopWorker();
        videoWidget_->dsai_stopCaptureThread();
        
        // Clean class filter on stop
        dsai_populateClassCheckboxes({});
        objectsLayout_->addWidget(new QLabel("(Start to populate)", this));
        
        startStopBtn_->setText("Start");
        startStopBtn_->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
        isRunning_ = false;
    } else {
        if (modelFilePath_.isEmpty()) { QMessageBox::warning(this, "No Model", "Please load a configuration file first."); return; }
        
        dsai_startWorker();
        videoWidget_->dsai_startCaptureThread();
        
        // Populate class filter on start
        if (detector_) dsai_populateClassCheckboxes(detector_->dsai_classNames());

        startStopBtn_->setText("Stop");
        startStopBtn_->setIcon(style()->standardIcon(QStyle::SP_MediaStop));
        isRunning_ = true;
    }
}

void MainWindow::dsai_startWorker() {
    dsai_stopWorker();
    worker_ = new DetectionWorker(this);
    if (detector_) worker_->dsai_setDetector(detector_.get());
    connect(worker_, &DetectionWorker::detectionResultsReady, this, &MainWindow::dsai_onDetectionResults);
    connect(worker_, &DetectionWorker::performanceMetricsUpdated, this, &MainWindow::dsai_onPerformanceMetrics);
    worker_->start();
}

void MainWindow::dsai_stopWorker() {
    if (worker_) { worker_->dsai_stop(); worker_->wait(); delete worker_; worker_ = nullptr; }
}

void MainWindow::dsai_openConfigDialog() {
    ConfigDialog diag(this);
    if (diag.exec() == QDialog::Accepted) { dsai_loadFromConfigFile(ConfigDialog::dsai_configFilePath()); }
}

void MainWindow::dsai_onDetectionResults(const std::vector<inference::Detection>& detections) {
    std::vector<inference::Detection> filtered;
    for (const auto& d : detections) {
        if (classCheckboxes_.contains(d.classId) && classCheckboxes_[d.classId]->isChecked()) {
            filtered.push_back(d);
        }
    }
    videoWidget_->dsai_setDetectionResults(filtered);
}

void MainWindow::dsai_onPerformanceMetrics(float fps, float infer, float nms, float avg) {
    fpsLabel_->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
    inferLatLabel_->setText(QString("Inference: %1 ms").arg(infer, 0, 'f', 1));
    nmsLatLabel_->setText(QString("NMS: %1 ms").arg(nms, 0, 'f', 1));
    avgLatLabel_->setText(QString("Average: %1 ms").arg(avg, 0, 'f', 1));
}

void MainWindow::dsai_updateClock() { clockLabel_->setText(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss")); }
void MainWindow::dsai_handleBoundingBoxChanged(const QRect& rect) { if (!roiConfigPath_.isEmpty()) videoWidget_->dsai_saveRoiToConfig(roiConfigPath_); }

void MainWindow::dsai_loadFromConfigFile(const QString& path) {
    bool wasRunning = isRunning_;
    if (wasRunning) dsai_toggleStartStop();

    AppConfig cfg = AppConfig::dsai_loadFromFile(path.toStdString());
    Debug::dsai_setEnabled(cfg.debugLogging);
    cfg.dsai_logConfig();

    modelFilePath_ = QString::fromStdString(cfg.modelFile);
    yamlFilePath_ = QString::fromStdString(cfg.yamlFile);
    
    videoWidget_->dsai_setCameraDevice(cfg.cameraDeviceId);
    videoWidget_->dsai_setRockchipHardware(cfg.rockchipHw);
    if (cfg.source == SourceType::VideoFile || cfg.source == SourceType::Rtsp) {
        QString src = QString::fromStdString(cfg.videoFile.empty() ? cfg.rtspUrl : cfg.videoFile);
        videoWidget_->dsai_setVideoSource(src);
    }

    inference::Backend backend = static_cast<inference::Backend>(cfg.backend);
    dsai_initializeDetector(backend, modelFilePath_, yamlFilePath_, cfg.confThreshold, cfg.nmsThreshold, QStringList());

    videoWidget_->dsai_setAppConfig(cfg);
    videoWidget_->dsai_setModelInputSize(detector_->dsai_inputSize());

    QFileInfo fi(modelFilePath_);
    roiConfigPath_ = fi.absolutePath() + "/" + fi.completeBaseName() + "_roi.yaml";
    videoWidget_->dsai_loadRoiFromConfig(roiConfigPath_);

    if (wasRunning) dsai_toggleStartStop();
}

void MainWindow::dsai_initializeDetector(inference::Backend b, const QString& m, const QString& y, float c, float n, const QStringList& l) {
    detector_ = inference::DetectorFactory::dsai_create(b, m.toStdString(), y.toStdString(), c, n);
    QStringList qnames; for(const auto& name : detector_->dsai_classNames()) qnames << QString::fromStdString(name);
    videoWidget_->dsai_setClassNames(qnames);
}

// ─── DetectionWorker ──────────────────────────────────────────────────────────
DetectionWorker::DetectionWorker(QObject* parent) : QThread(parent) {}
DetectionWorker::~DetectionWorker() { dsai_stop(); wait(); }
void DetectionWorker::dsai_setDetector(inference::IDetector* d) { QMutexLocker lk(&mutex_); detector_ = d; }
void DetectionWorker::dsai_pushFrame(const cv::Mat& f) { QMutexLocker lk(&mutex_); pendingFrame_ = f.clone(); frameReady_ = true; condition_.wakeOne(); }
void DetectionWorker::dsai_setEnabled(bool e) { QMutexLocker lk(&mutex_); enabled_ = e; }
void DetectionWorker::dsai_stop() { QMutexLocker lk(&mutex_); stopped_ = true; condition_.wakeOne(); }

void DetectionWorker::run() {
    using namespace std::chrono;
    auto lastHeartbeat = steady_clock::now(); uint64_t frameCount = 0;
    while (true) {
        cv::Mat frame; inference::IDetector* det = nullptr;
        {
            QMutexLocker lk(&mutex_);
            while (!frameReady_ && !stopped_) condition_.wait(&mutex_);
            if (stopped_) break;
            frame = pendingFrame_.clone(); frameReady_ = false; det = detector_;
        }
        if (det && !frame.empty()) {
            auto t1 = steady_clock::now();
            auto results = det->dsai_infer(frame);
            auto t2 = steady_clock::now();
            float inferMs = duration<float, std::milli>(t2 - t1).count();
            float nmsMs = 0.0f; float totalMs = inferMs + nmsMs;
            latencyHistory_.push_back(totalMs);
            if (latencyHistory_.size() > 30) latencyHistory_.pop_front();
            float sum = 0; for (float v : latencyHistory_) sum += v;
            float avgMs = sum / latencyHistory_.size();
            float fps = 1000.0f / avgMs;
            frameCount++; auto now = steady_clock::now();
            if (duration_cast<seconds>(now - lastHeartbeat).count() >= 1) {
                DBG_LOG("FPS", "FPS=%.1f  infer_avg=%.1fms  frame=%llu\n", fps, avgMs, (unsigned long long)frameCount);
                lastHeartbeat = now;
            }
            if (Debug::isEnabled()) {
                std::map<int, int> counts; for (const auto& d : results) counts[d.classId]++;
                printf("[DETECTOR] frame %llu: %zu detections\n", (unsigned long long)frameCount, results.size());
                fflush(stdout);
            }
            emit detectionResultsReady(results);
            emit performanceMetricsUpdated(fps, inferMs, nmsMs, avgMs);
        }
    }
}
