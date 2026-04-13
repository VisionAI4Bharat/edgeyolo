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

#include <memory>
#include <deque>

#include "videowidget.h"
#include "configdialog.h"
#include "inference/IDetector.h"
#include "inference/DetectorFactory.h"

// ─── DetectionWorker ──────────────────────────────────────────────────────────

/**
 * Background thread that pulls frames from a queue, runs IDetector::infer(),
 * and emits results back to the GUI thread.
 *
 * Ownership: MainWindow owns the unique_ptr<IDetector>; DetectionWorker holds
 * a raw non-owning pointer.  The detector must not be destroyed while the
 * worker is running — always call stop() and wait() before swapping detectors.
 */
class DetectionWorker : public QThread {
    Q_OBJECT
public:
    explicit DetectionWorker(QObject* parent = nullptr);
    ~DetectionWorker() override;

    /** Set the detector to use.  Must be called before start() or after stop(). */
    void setDetector(inference::IDetector* detector);

    /** Push a new frame.  Thread-safe; wakes the worker if it was idle. */
    void pushFrame(const cv::Mat& frame);

    /** Set active/paused state.  Thread-safe. */
    void setEnabled(bool enabled);

    /** Signal the run loop to exit, then caller should call wait(). */
    void stop();

signals:
    void detectionResultsReady(QVector<QRect>  boxes,
                                QVector<int>    classIds,
                                QVector<float>  confidences);
    void performanceMetricsUpdated(float fps,
                                   float inferenceLatencyMs,
                                   float nmsLatencyMs,
                                   float avgLatencyMs);

protected:
    void run() override;

private:
    inference::IDetector* detector_  = nullptr;
    cv::Mat               pendingFrame_;
    bool                  frameReady_ = false;
    bool                  enabled_    = false;
    bool                  stopped_    = false;

    QMutex          mutex_;
    QWaitCondition  condition_;

    // Latency rolling average (30 frames)
    static constexpr int kHistorySize = 30;
    std::deque<float> latencyHistory_;

    // FPS tracking
    int     frameCount_   = 0;
    qint64  fpsWindowStart_ = 0;
};

// ─── MainWindow ───────────────────────────────────────────────────────────────

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void openConfigDialog();
    void toggleEditRunMode();
    void onDetectionResults(QVector<QRect>  boxes,
                            QVector<int>    classIds,
                            QVector<float>  confidences);
    void onPerformanceMetrics(float fps,
                              float inferenceLatency,
                              float nmsLatency,
                              float avgLatency);
    void updateClock();
    void handleBoundingBoxChanged(const QRect& box);

private:
    void setupUI();
    void setupConnections();
    void initializeDetector(inference::Backend backend,
                            const QString&     modelPath,
                            const QString&     yamlPath,
                            float              confThres,
                            float              nmsThres);
    void populateClassCheckboxes(const std::vector<std::string>& names);
    void startWorker();
    void stopWorker();

    // ── UI ────────────────────────────────────────────────────────────────
    QSplitter*    mainSplitter_   = nullptr;
    VideoWidget*  videoWidget_    = nullptr;
    QWidget*      controlPanel_   = nullptr;
    QVBoxLayout*  controlLayout_  = nullptr;

    QPushButton*  configButton_   = nullptr;
    QPushButton*  editRunButton_  = nullptr;

    QGroupBox*    metricsGroup_   = nullptr;
    QLabel*       fpsLabel_       = nullptr;
    QLabel*       inferLatLabel_  = nullptr;
    QLabel*       nmsLatLabel_    = nullptr;
    QLabel*       avgLatLabel_    = nullptr;
    QLabel*       timeLabel_      = nullptr;

    QGroupBox*    objectsGroup_   = nullptr;
    QVBoxLayout*  objectsLayout_  = nullptr;
    QMap<int, QCheckBox*> classCheckboxes_; // classId -> checkbox

    QStatusBar*   statusBar_      = nullptr;

    // ── state ─────────────────────────────────────────────────────────────
    QString  modelFilePath_;
    QString  yamlFilePath_;
    QString  roiConfigPath_;   // YAML file where ROI is persisted
    bool     isEditMode_       = false;
    bool     isUsingVideoFile_ = false;
    bool     isUsingRtspStream_ = false;
    QRect    currentBoundingBox_;

    // ── detection ─────────────────────────────────────────────────────────
    std::unique_ptr<inference::IDetector> detector_;
    QMutex                                detectorMutex_;
    DetectionWorker*                      worker_  = nullptr;
    QTimer*                               clockTimer_ = nullptr;
};

#endif // MAINWINDOW_H
