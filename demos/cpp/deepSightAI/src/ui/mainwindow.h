/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSplitter>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QCheckBox>
#include <QTimer>
#include <QStatusBar>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QVector>
#include <QString>
#include <QMap>
#include <deque>
#include <memory>

#include "videowidget.h"
#include "configdialog.h"
#include "inference/IDetector.h"
#include "inference/DetectorFactory.h"

class DetectionWorker : public QThread {
    Q_OBJECT
public:
    explicit DetectionWorker(QObject* parent = nullptr);
    ~DetectionWorker() override;
    void dsai_setDetector(inference::IDetector* detector);
    void dsai_pushFrame(const cv::Mat& frame);
    void dsai_setEnabled(bool enabled);
    void dsai_stop();
signals:
    void detectionResultsReady(const std::vector<inference::Detection>& detections);
    void performanceMetricsUpdated(float fps, float inferMs, float nmsMs, float avgMs);
protected:
    void run() override;
private:
    inference::IDetector* detector_ = nullptr;
    cv::Mat pendingFrame_;
    bool frameReady_ = false;
    bool enabled_ = false;
    bool stopped_ = false;
    QMutex mutex_;
    QWaitCondition condition_;
    std::deque<float> latencyHistory_;
};

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();
    void dsai_loadFromConfigFile(const QString& path);
private slots:
    void dsai_openConfigDialog();
    void dsai_toggleStartStop();
    void dsai_onDetectionResults(const std::vector<inference::Detection>& detections);
    void dsai_onPerformanceMetrics(float fps, float inferMs, float nmsMs, float avgMs);
    void dsai_updateClock();
    void dsai_handleBoundingBoxChanged(const QRect& rect);
private:
    void dsai_setupUI();
    void dsai_setupConnections();
    void dsai_initializeDetector(inference::Backend backend, const QString& modelPath, const QString& yamlPath, float conf, float nms, const QStringList& labels);
    void dsai_populateClassCheckboxes(const std::vector<std::string>& names);
    void dsai_startWorker();
    void dsai_stopWorker();

    VideoWidget* videoWidget_ = nullptr;
    QPushButton* startStopBtn_ = nullptr;
    QLabel* fpsLabel_ = nullptr;
    QLabel* inferLatLabel_ = nullptr;
    QLabel* nmsLatLabel_ = nullptr;
    QLabel* avgLatLabel_ = nullptr;
    QLabel* clockLabel_ = nullptr;
    QStatusBar* statusBar_ = nullptr;

    QGroupBox*    objectsGroup_   = nullptr;
    QVBoxLayout*  objectsLayout_  = nullptr;
    QMap<int, QCheckBox*> classCheckboxes_;

    std::unique_ptr<inference::IDetector> detector_;
    DetectionWorker* worker_ = nullptr;
    QString modelFilePath_;
    QString yamlFilePath_;
    QString roiConfigPath_;
    bool isRunning_ = false;
};

#endif
