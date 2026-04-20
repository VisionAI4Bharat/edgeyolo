#include "configdialog.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QRegularExpression>
#include <QScrollArea>
#include <QApplication>
#include <QDebug>
#include <QStandardPaths>
#include <stdexcept>
#include <fstream>
#include <yaml-cpp/yaml.h>

// ─── helpers ──────────────────────────────────────────────────────────────────

/*
 * Enumerate V4L2 video devices using v4l2-ctl when available, falling back to
 * scanning /dev/video*.  Returns a list of (deviceId, displayName) pairs.
 * Never throws; returns an empty list on any failure.
 */
static QVector<QPair<int,QString>> enumerateCameras() noexcept
{
    QVector<QPair<int,QString>> cameras;

    // ── preferred: v4l2-ctl --list-devices ────────────────────────────────
    QProcess proc;
    proc.start("v4l2-ctl", QStringList() << "--list-devices");
    if (proc.waitForFinished(3000)) {
        const QString output = QString::fromLocal8Bit(proc.readAllStandardOutput());
        // Output format:
        //   <Name> (<bus>):
        //           /dev/videoN
        //           /dev/videoM
        QString currentName;
        const QRegularExpression nameRe(R"(^(.+)\s*\(.*\):$)");
        const QRegularExpression devRe(R"(^\s+(/dev/video(\d+))\s*$)");
        for (const QString& line : output.split('\n')) {
            QRegularExpressionMatch m = nameRe.match(line);
            if (m.hasMatch()) {
                currentName = m.captured(1).trimmed();
                continue;
            }
            m = devRe.match(line);
            if (m.hasMatch()) {
                bool ok = false;
                int id = m.captured(2).toInt(&ok);
                if (ok) {
                    QString label = currentName.isEmpty()
                        ? QString("/dev/video%1").arg(id)
                        : QString("/dev/video%1  (%2)").arg(id).arg(currentName);
                    cameras.append({id, label});
                    currentName.clear();
                }
            }
        }
    }

    // ── fallback: scan /dev/video* ────────────────────────────────────────
    if (cameras.isEmpty()) {
        const QDir dev("/dev", "video*", QDir::Name, QDir::System);
        const QRegularExpression numRe(R"(video(\d+))");
        for (const QFileInfo& fi : dev.entryInfoList()) {
            QRegularExpressionMatch m = numRe.match(fi.fileName());
            if (m.hasMatch()) {
                bool ok = false;
                int id = m.captured(1).toInt(&ok);
                if (ok)
                    cameras.append({id, fi.absoluteFilePath()});
            }
        }
    }

    return cameras;
}

// ─── ConfigDialog ─────────────────────────────────────────────────────────────

ConfigDialog::ConfigDialog(QWidget *parent) : QDialog(parent)
{
    setWindowTitle("deepSightAI Configurator");
    setModal(true);
    setMinimumWidth(520);

    // Wrap everything in a scroll area so the dialog is usable on small screens
    QScrollArea* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);

    QWidget* container = new QWidget(scroll);
    QVBoxLayout* mainLayout = new QVBoxLayout(container);
    mainLayout->setSpacing(8);
    mainLayout->setContentsMargins(10, 10, 10, 10);

    setupBackendSection(mainLayout);
    mainLayout->addWidget(makeSeparator());
    setupModelSection(mainLayout);
    mainLayout->addWidget(makeSeparator());
    setupSourceSection(mainLayout);
    setupCameraSection(mainLayout);
    setupVideoFileSection(mainLayout);
    setupRtspSection(mainLayout);
    setupRockchipSection(mainLayout);
    mainLayout->addWidget(makeSeparator());
    setupResolutionSection(mainLayout);
    setupFpsSection(mainLayout);
    mainLayout->addWidget(makeSeparator());
    setupV4l2Section(mainLayout);
    mainLayout->addWidget(makeSeparator());
    setupThresholdsSection(mainLayout);
    mainLayout->addWidget(makeSeparator());
    setupRoiSection(mainLayout);
    mainLayout->addWidget(makeSeparator());
    setupInfoSection(mainLayout);
    mainLayout->addWidget(makeSeparator());
    setupDebugSection(mainLayout);
    mainLayout->addStretch();

    scroll->setWidget(container);

    QVBoxLayout* dialogLayout = new QVBoxLayout(this);
    dialogLayout->setContentsMargins(0, 0, 0, 0);
    dialogLayout->addWidget(scroll);
    setupButtonRow(dialogLayout);

    // Wire up OK / Cancel
    connect(okButton_,     &QPushButton::clicked, this, &ConfigDialog::onAccepted);
    connect(cancelButton_, &QPushButton::clicked, this, &QDialog::reject);

    loadConfig();
    updateSourceVisibility();
    updateInfoPanel();
}

// ─── section builders ─────────────────────────────────────────────────────────

inference::Backend ConfigDialog::getBackend() const
{
    if (!backendComboBox_) return inference::Backend::ONNX;
    const int idx = backendComboBox_->currentData().toInt();
    return static_cast<inference::Backend>(idx);
}

void ConfigDialog::setupBackendSection(QVBoxLayout* parent)
{
    backendGroup_ = new QGroupBox("Inference Backend", this);
    QHBoxLayout* lay = new QHBoxLayout(backendGroup_);

    backendComboBox_ = new QComboBox(this);

    // ONNX Runtime — always available
    backendComboBox_->addItem("ONNX Runtime", static_cast<int>(inference::Backend::ONNX));

    // OpenVINO — disabled at runtime if not compiled in
    {
        const int idx = backendComboBox_->count();
        backendComboBox_->addItem("OpenVINO", static_cast<int>(inference::Backend::OPENVINO));
        if (!inference::DetectorFactory::isAvailable(inference::Backend::OPENVINO)) {
            // Grey out the item; Qt doesn't have a built-in disable-item API on all styles,
            // so we mark it via the user data sentinel and protect in validation.
            auto* model = qobject_cast<QStandardItemModel*>(backendComboBox_->model());
            if (model) model->item(idx)->setEnabled(false);
        }
    }

    // RKNN — disabled at runtime if not compiled in
    {
        const int idx = backendComboBox_->count();
        backendComboBox_->addItem("RKNN (RV1106)", static_cast<int>(inference::Backend::RKNN));
        if (!inference::DetectorFactory::isAvailable(inference::Backend::RKNN)) {
            auto* model = qobject_cast<QStandardItemModel*>(backendComboBox_->model());
            if (model) model->item(idx)->setEnabled(false);
        }
    }

    lay->addWidget(new QLabel("Backend:", this));
    lay->addWidget(backendComboBox_, 1);
    parent->addWidget(backendGroup_);
}

void ConfigDialog::setupModelSection(QVBoxLayout* parent)
{
    modelGroup_ = new QGroupBox("Model", this);
    QFormLayout* form = new QFormLayout(modelGroup_);

    // Model file row
    modelFilePathEdit_ = new QLineEdit(this);
    modelFilePathEdit_->setReadOnly(true);
    modelFilePathEdit_->setPlaceholderText("Select model file (.onnx / .rknn)…");
    browseModelButton_ = new QPushButton("Browse…", this);
    connect(browseModelButton_, &QPushButton::clicked, this, &ConfigDialog::browseModelFile);

    QHBoxLayout* modelRow = new QHBoxLayout();
    modelRow->addWidget(modelFilePathEdit_, 1);
    modelRow->addWidget(browseModelButton_);
    form->addRow("Model file:", modelRow);

    // YAML config row
    yamlFilePathEdit_ = new QLineEdit(this);
    yamlFilePathEdit_->setPlaceholderText("Optional — auto-detected if empty");
    browseYamlButton_ = new QPushButton("Browse…", this);
    connect(browseYamlButton_, &QPushButton::clicked, this, &ConfigDialog::browseYamlFile);

    QHBoxLayout* yamlRow = new QHBoxLayout();
    yamlRow->addWidget(yamlFilePathEdit_, 1);
    yamlRow->addWidget(browseYamlButton_);
    form->addRow("Class config YAML:", yamlRow);

    editClassLabelsBtn_ = new QPushButton("Edit Class Labels…", this);
    connect(editClassLabelsBtn_, &QPushButton::clicked, this, &ConfigDialog::openClassLabelsEditor);
    form->addRow("", editClassLabelsBtn_);

    parent->addWidget(modelGroup_);
}

void ConfigDialog::setupSourceSection(QVBoxLayout* parent)
{
    sourceGroup_ = new QGroupBox("Video Source", this);
    QHBoxLayout* lay = new QHBoxLayout(sourceGroup_);

    cameraRadio_    = new QRadioButton("Camera",     this);
    videoFileRadio_ = new QRadioButton("Video File", this);
    rtspRadio_      = new QRadioButton("RTSP Stream", this);
    cameraRadio_->setChecked(true);

    sourceButtonGroup_ = new QButtonGroup(this);
    sourceButtonGroup_->addButton(cameraRadio_,    0);
    sourceButtonGroup_->addButton(videoFileRadio_, 1);
    sourceButtonGroup_->addButton(rtspRadio_,      2);

    lay->addWidget(cameraRadio_);
    lay->addWidget(videoFileRadio_);
    lay->addWidget(rtspRadio_);
    lay->addStretch();

    connect(sourceButtonGroup_, QOverload<int>::of(&QButtonGroup::idClicked),
            this, [this](int){ onSourceChanged(); });

    parent->addWidget(sourceGroup_);
}

void ConfigDialog::setupCameraSection(QVBoxLayout* parent)
{
    cameraGroup_ = new QGroupBox("Camera Device", this);
    QVBoxLayout* vlay = new QVBoxLayout(cameraGroup_);

    cameraComboBox_       = new QComboBox(this);
    refreshCamerasButton_ = new QPushButton("Refresh", this);

    connect(refreshCamerasButton_, &QPushButton::clicked, this, &ConfigDialog::refreshCameras);

    QHBoxLayout* lay = new QHBoxLayout();
    lay->addWidget(cameraComboBox_, 1);
    lay->addWidget(refreshCamerasButton_);
    vlay->addLayout(lay);

    parent->addWidget(cameraGroup_);

    populateCameras();
}

void ConfigDialog::setupVideoFileSection(QVBoxLayout* parent)
{
    videoFileGroup_ = new QGroupBox("Video File", this);
    QHBoxLayout* lay = new QHBoxLayout(videoFileGroup_);

    videoFilePathEdit_ = new QLineEdit(this);
    videoFilePathEdit_->setPlaceholderText("Path to .mp4, .avi, .mkv…");
    browseVideoButton_ = new QPushButton("Browse…", this);

    connect(browseVideoButton_, &QPushButton::clicked, this, &ConfigDialog::browseVideoFile);

    lay->addWidget(videoFilePathEdit_, 1);
    lay->addWidget(browseVideoButton_);
    parent->addWidget(videoFileGroup_);
}

void ConfigDialog::setupRtspSection(QVBoxLayout* parent)
{
    rtspGroup_ = new QGroupBox("RTSP Stream", this);
    QHBoxLayout* lay = new QHBoxLayout(rtspGroup_);

    rtspUrlEdit_ = new QLineEdit(this);
    rtspUrlEdit_->setPlaceholderText("rtsp://user:pass@host:port/path");

    lay->addWidget(rtspUrlEdit_);
    parent->addWidget(rtspGroup_);
}

void ConfigDialog::setupRockchipSection(QVBoxLayout* parent)
{
    rockchipHwCheckbox_ = new QCheckBox("Rockchip Hardware", this);
    rockchipHwCheckbox_->setToolTip(
        "When checked, selects the RKNN backend and enables Rockchip NPU inference.\n"
        "Only applicable for Camera and RTSP sources.\n"
        "Requires WITH_RKNN compiled in and a .rknn model file.");
    parent->addWidget(rockchipHwCheckbox_);
}

void ConfigDialog::setupResolutionSection(QVBoxLayout* parent)
{
    resolutionGroup_ = new QGroupBox("Resolution", this);
    QHBoxLayout* lay = new QHBoxLayout(resolutionGroup_);

    resolutionComboBox_ = new QComboBox(this);
    resolutionComboBox_->addItem("640 × 480",   QSize(640,  480));
    resolutionComboBox_->addItem("800 × 600",   QSize(800,  600));
    resolutionComboBox_->addItem("1280 × 720",  QSize(1280, 720));
    resolutionComboBox_->addItem("1920 × 1080", QSize(1920, 1080));
    resolutionComboBox_->addItem("Custom",       QSize(0,    0));
    resolutionComboBox_->setCurrentIndex(0);

    connect(resolutionComboBox_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ConfigDialog::onResolutionChanged);

    lay->addWidget(resolutionComboBox_);
    lay->addStretch();
    parent->addWidget(resolutionGroup_);
}

void ConfigDialog::setupFpsSection(QVBoxLayout* parent)
{
    fpsGroup_ = new QGroupBox("Target FPS", this);
    QHBoxLayout* lay = new QHBoxLayout(fpsGroup_);

    fpsComboBox_ = new QComboBox(this);
    fpsComboBox_->addItem("15",  15);
    fpsComboBox_->addItem("25",  25);
    fpsComboBox_->addItem("30",  30);
    fpsComboBox_->addItem("60",  60);
    fpsComboBox_->addItem("Max", -1);
    fpsComboBox_->setCurrentIndex(2); // default 30

    lay->addWidget(fpsComboBox_);
    lay->addStretch();
    parent->addWidget(fpsGroup_);
}

void ConfigDialog::setupV4l2Section(QVBoxLayout* parent)
{
    v4l2ControlsGroup_ = new QGroupBox("Camera Controls (V4L2)", this);
    QFormLayout* form = new QFormLayout(v4l2ControlsGroup_);

    auto makeRow = [&](const QString& label, QSlider*& slider, QSpinBox*& spin,
                       int minVal, int maxVal, int defaultVal)
    {
        QHBoxLayout* row = makeSliderRow(slider, spin, minVal, maxVal, defaultVal);
        form->addRow(label, row);
    };

    makeRow("Gain:",       gainSlider_,       gainSpinBox_,       0, 255, 50);
    makeRow("Gamma:",      gammaSlider_,      gammaSpinBox_,      72, 500, 100);
    makeRow("Brightness:", brightnessSlider_, brightnessSpinBox_, -64, 64, 0);

    // Slider ↔ spin synchronisation
    connect(gainSlider_,       &QSlider::valueChanged, this, &ConfigDialog::onGainSliderChanged);
    connect(gainSpinBox_,      QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ConfigDialog::onGainSpinChanged);
    connect(gammaSlider_,      &QSlider::valueChanged, this, &ConfigDialog::onGammaSliderChanged);
    connect(gammaSpinBox_,     QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ConfigDialog::onGammaSpinChanged);
    connect(brightnessSlider_, &QSlider::valueChanged, this, &ConfigDialog::onBrightnessSliderChanged);
    connect(brightnessSpinBox_,QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ConfigDialog::onBrightnessSpinChanged);

    applyV4l2Button_ = new QPushButton("Apply to Device", this);
    connect(applyV4l2Button_, &QPushButton::clicked, this, &ConfigDialog::applyV4l2Settings);
    form->addRow("", applyV4l2Button_);

    parent->addWidget(v4l2ControlsGroup_);
}

void ConfigDialog::setupThresholdsSection(QVBoxLayout* parent)
{
    thresholdsGroup_ = new QGroupBox("Detection Thresholds", this);
    QFormLayout* form = new QFormLayout(thresholdsGroup_);

    auto makeDoubleSpin = [&](double min, double max, double step, double defaultVal) {
        auto* spin = new QDoubleSpinBox(this);
        spin->setRange(min, max);
        spin->setSingleStep(step);
        spin->setDecimals(2);
        spin->setValue(defaultVal);
        spin->setFixedWidth(80);
        return spin;
    };

    confThresSpin_ = makeDoubleSpin(0.01, 0.99, 0.05, 0.25);
    confThresSpin_->setToolTip("Objectness × class score threshold; lower = more detections");
    form->addRow("Confidence:", confThresSpin_);

    nmsThresSpin_ = makeDoubleSpin(0.01, 0.99, 0.05, 0.45);
    nmsThresSpin_->setToolTip("IoU threshold for NMS; lower = fewer overlapping boxes");
    form->addRow("NMS IoU:", nmsThresSpin_);

    parent->addWidget(thresholdsGroup_);
}

void ConfigDialog::setupRoiSection(QVBoxLayout* parent)
{
    roiGroup_ = new QGroupBox("Region of Interest (ROI)", this);
    QVBoxLayout* vlay = new QVBoxLayout(roiGroup_);

    enableRoiCheckbox_ = new QCheckBox("Enable ROI filtering", this);
    roiInfoLabel_      = new QLabel("No ROI set — draw one in Edit mode after closing this dialog.", this);
    roiInfoLabel_->setWordWrap(true);
    clearRoiButton_    = new QPushButton("Clear ROI", this);

    connect(clearRoiButton_, &QPushButton::clicked, this, &ConfigDialog::clearRoi);

    vlay->addWidget(enableRoiCheckbox_);
    vlay->addWidget(roiInfoLabel_);
    vlay->addWidget(clearRoiButton_);
    parent->addWidget(roiGroup_);
}

void ConfigDialog::setupInfoSection(QVBoxLayout* parent)
{
    infoGroup_ = new QGroupBox("Information", this);
    QVBoxLayout* vlay = new QVBoxLayout(infoGroup_);

    infoTextBrowser_ = new QTextBrowser(this);
    infoTextBrowser_->setMaximumHeight(100);
    infoTextBrowser_->setOpenExternalLinks(false);

    vlay->addWidget(infoTextBrowser_);
    parent->addWidget(infoGroup_);
}

void ConfigDialog::setupDebugSection(QVBoxLayout* parent)
{
    debugGroup_ = new QGroupBox("Diagnostics", this);
    QVBoxLayout* vlay = new QVBoxLayout(debugGroup_);

    debugLoggingCheckbox_ = new QCheckBox("Enable debug logging (stdout/stderr)", this);
    debugLoggingCheckbox_->setToolTip(
        "When checked, verbose logs are printed to stdout (info) and stderr (errors).\n"
        "Useful for diagnosing inference, FPS, ROI, and video-source issues.");
    vlay->addWidget(debugLoggingCheckbox_);

    parent->addWidget(debugGroup_);
}

void ConfigDialog::setupButtonRow(QVBoxLayout* parent)
{
    QHBoxLayout* row = new QHBoxLayout();
    row->setContentsMargins(10, 6, 10, 10);

    okButton_     = new QPushButton("Apply",  this);
    cancelButton_ = new QPushButton("Cancel", this);
    okButton_->setDefault(true);

    row->addStretch();
    row->addWidget(okButton_);
    row->addWidget(cancelButton_);
    parent->addLayout(row);
}

// ─── static helpers ───────────────────────────────────────────────────────────

QWidget* ConfigDialog::makeSeparator()
{
    QFrame* line = new QFrame();
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);
    return line;
}

QHBoxLayout* ConfigDialog::makeSliderRow(QSlider*& slider, QSpinBox*& spin,
                                          int minVal, int maxVal, int defaultVal)
{
    QHBoxLayout* row = new QHBoxLayout();
    slider = new QSlider(Qt::Horizontal);
    spin   = new QSpinBox();

    slider->setRange(minVal, maxVal);
    slider->setValue(defaultVal);
    spin->setRange(minVal, maxVal);
    spin->setValue(defaultVal);
    spin->setFixedWidth(64);

    row->addWidget(slider, 1);
    row->addWidget(spin);
    return row;
}

// ─── internal helpers ─────────────────────────────────────────────────────────

void ConfigDialog::populateCameras()
{
    cameraComboBox_->clear();
    availableCameraIds_.clear();

    const auto cameras = enumerateCameras();
    if (cameras.isEmpty()) {
        // Graceful degradation: assume /dev/video0 exists
        cameraComboBox_->addItem("/dev/video0 (assumed)");
        availableCameraIds_.append(0);
        qWarning() << "ConfigDialog: no cameras detected, defaulting to /dev/video0";
        return;
    }

    for (const auto& [id, name] : cameras) {
        cameraComboBox_->addItem(name);
        availableCameraIds_.append(id);
    }
}

void ConfigDialog::updateSourceVisibility()
{
    const bool isCamera    = cameraRadio_->isChecked();
    const bool isVideoFile = videoFileRadio_->isChecked();
    const bool isRtsp      = rtspRadio_->isChecked();

    cameraGroup_->setVisible(isCamera);
    if (rockchipHwCheckbox_) {
        const bool hwEnabled = isCamera || isRtsp;
        rockchipHwCheckbox_->setEnabled(hwEnabled);
        if (!hwEnabled) rockchipHwCheckbox_->setChecked(false);
    }
    v4l2ControlsGroup_->setVisible(isCamera);
    videoFileGroup_->setVisible(isVideoFile);
    rtspGroup_->setVisible(isRtsp);

    // Resolution / FPS only relevant for camera sources (hidden for video file and RTSP)
    resolutionGroup_->setVisible(isCamera);
    fpsGroup_->setVisible(isCamera);

    // ROI always available
    roiGroup_->setVisible(true);

    adjustSize();
}

void ConfigDialog::updateInfoPanel()
{
    if (!infoTextBrowser_)
        return;

    QStringList lines;
    lines << "<b>Current selection:</b>";
    lines << QString("Model: %1").arg(
        modelFilePath_.isEmpty() ? "<i>not set</i>" : QFileInfo(modelFilePath_).fileName());

    if (cameraRadio_->isChecked()) {
        const int idx = cameraComboBox_->currentIndex();
        const QString camName = (idx >= 0) ? cameraComboBox_->currentText() : "<i>none</i>";
        lines << QString("Source: Camera — %1").arg(camName);
    } else if (videoFileRadio_->isChecked()) {
        lines << QString("Source: Video file — %1").arg(
            videoFilePathEdit_->text().isEmpty() ? "<i>not set</i>"
                                                 : videoFilePathEdit_->text());
    } else {
        lines << QString("Source: RTSP — %1").arg(
            rtspUrlEdit_->text().isEmpty() ? "<i>not set</i>" : rtspUrlEdit_->text());
    }

    const QSize res = resolutionComboBox_->currentData().toSize();
    if (res.isValid() && !res.isNull())
        lines << QString("Resolution: %1 × %2").arg(res.width()).arg(res.height());

    const int fps = getFps();
    lines << QString("FPS: %1").arg(fps < 0 ? "Max" : QString::number(fps));

    if (enableRoiCheckbox_->isChecked() && roi_.isValid())
        lines << QString("ROI: (%1,%2) %3×%4")
                     .arg(roi_.x()).arg(roi_.y())
                     .arg(roi_.width()).arg(roi_.height());

    infoTextBrowser_->setHtml(lines.join("<br>"));
}

bool ConfigDialog::validateInputs(QString& errorMsg) const
{
    if (modelFilePath_.isEmpty()) {
        errorMsg = "Please select an ONNX model file.";
        return false;
    }
    if (!QFile::exists(modelFilePath_)) {
        errorMsg = QString("Model file not found:\n%1").arg(modelFilePath_);
        return false;
    }
    if (videoFileRadio_->isChecked()) {
        const QString path = videoFilePathEdit_->text().trimmed();
        if (path.isEmpty()) {
            errorMsg = "Please specify a video file path.";
            return false;
        }
        if (!QFile::exists(path)) {
            errorMsg = QString("Video file not found:\n%1").arg(path);
            return false;
        }
    }
    if (rtspRadio_->isChecked()) {
        const QString url = rtspUrlEdit_->text().trimmed();
        if (url.isEmpty()) {
            errorMsg = "Please enter an RTSP URL.";
            return false;
        }
        if (!url.startsWith("rtsp://", Qt::CaseInsensitive) &&
            !url.startsWith("rtsps://", Qt::CaseInsensitive)) {
            errorMsg = "RTSP URL must start with rtsp:// or rtsps://";
            return false;
        }
    }
    if (cameraRadio_->isChecked() && availableCameraIds_.isEmpty()) {
        errorMsg = "No camera devices are available.";
        return false;
    }
    return true;
}

// ─── getters ──────────────────────────────────────────────────────────────────

QString ConfigDialog::getVideoOrRtspPath() const
{
    if (videoFileRadio_->isChecked())
        return videoFilePathEdit_->text().trimmed();
    if (rtspRadio_->isChecked())
        return rtspUrlEdit_->text().trimmed();
    return {};
}

int ConfigDialog::getFps() const
{
    if (!fpsComboBox_)
        return 30;
    const QVariant v = fpsComboBox_->currentData();
    return v.isValid() ? v.toInt() : 30;
}

int ConfigDialog::getWidth() const
{
    if (!resolutionComboBox_)
        return 640;
    const QSize s = resolutionComboBox_->currentData().toSize();
    return (s.isValid() && s.width() > 0) ? s.width() : 640;
}

int ConfigDialog::getHeight() const
{
    if (!resolutionComboBox_)
        return 480;
    const QSize s = resolutionComboBox_->currentData().toSize();
    return (s.isValid() && s.height() > 0) ? s.height() : 480;
}

// ─── slots ────────────────────────────────────────────────────────────────────

void ConfigDialog::browseModelFile()
{
    const inference::Backend backend = getBackend();
    const bool isRknn = (backend == inference::Backend::RKNN);

    const QString filter = isRknn
        ? "RKNN Models (*.rknn);;All Files (*)"
        : "ONNX Models (*.onnx);;All Files (*)";

    const QString path = QFileDialog::getOpenFileName(
        this,
        isRknn ? "Select RKNN Model" : "Select ONNX Model",
        modelFilePath_.isEmpty() ? QDir::homePath() : QFileInfo(modelFilePath_).absolutePath(),
        filter
    );
    if (path.isEmpty())
        return;

    if (!QFile::exists(path)) {
        QMessageBox::warning(this, "File Not Found",
                             "The selected file does not exist:\n" + path);
        return;
    }

    modelFilePath_ = path;
    modelFilePathEdit_->setText(QFileInfo(path).fileName());
    modelFilePathEdit_->setToolTip(path);

    // Auto-fill YAML path if a sidecar exists next to the model
    if (yamlFilePathEdit_->text().isEmpty()) {
        const QString autoYaml = QFileInfo(path).absolutePath() + "/" +
                                 QFileInfo(path).completeBaseName() + ".yaml";
        if (QFile::exists(autoYaml)) {
            yamlFilePath_ = autoYaml;
            yamlFilePathEdit_->setText(autoYaml);
        }
    }

    updateInfoPanel();
}

void ConfigDialog::browseYamlFile()
{
    const QString path = QFileDialog::getOpenFileName(
        this,
        "Select Class Config YAML",
        yamlFilePath_.isEmpty() ? QDir::homePath() : QFileInfo(yamlFilePath_).absolutePath(),
        "YAML Files (*.yaml *.yml);;All Files (*)"
    );
    if (path.isEmpty())
        return;

    if (!QFile::exists(path)) {
        QMessageBox::warning(this, "File Not Found",
                             "The selected file does not exist:\n" + path);
        return;
    }

    yamlFilePath_ = path;
    yamlFilePathEdit_->setText(path);
    updateInfoPanel();
}

void ConfigDialog::openClassLabelsEditor()
{
    ClassLabelsDialog dlg(classLabels_, yamlFilePathEdit_->text().trimmed(), this);
    if (dlg.exec() == QDialog::Accepted)
        classLabels_ = dlg.getClassLabels();
}

void ConfigDialog::browseVideoFile()
{
    const QString path = QFileDialog::getOpenFileName(
        this,
        "Select Video File",
        QDir::homePath(),
        "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;All Files (*)"
    );
    if (path.isEmpty())
        return;

    if (!QFile::exists(path)) {
        QMessageBox::warning(this, "File Not Found",
                             "The selected file does not exist:\n" + path);
        return;
    }

    videoFilePathEdit_->setText(path);
    updateInfoPanel();
}

void ConfigDialog::onSourceChanged()
{
    updateSourceVisibility();
    updateInfoPanel();
}

void ConfigDialog::refreshCameras()
{
    populateCameras();
    updateInfoPanel();
}

void ConfigDialog::onResolutionChanged(int /*index*/)
{
    updateInfoPanel();
}

// Slider ↔ spin sync — use blockSignals to avoid infinite recursion
void ConfigDialog::onGainSliderChanged(int value)
{
    gainSpinBox_->blockSignals(true);
    gainSpinBox_->setValue(value);
    gainSpinBox_->blockSignals(false);
}
void ConfigDialog::onGainSpinChanged(int value)
{
    gainSlider_->blockSignals(true);
    gainSlider_->setValue(value);
    gainSlider_->blockSignals(false);
}
void ConfigDialog::onGammaSliderChanged(int value)
{
    gammaSpinBox_->blockSignals(true);
    gammaSpinBox_->setValue(value);
    gammaSpinBox_->blockSignals(false);
}
void ConfigDialog::onGammaSpinChanged(int value)
{
    gammaSlider_->blockSignals(true);
    gammaSlider_->setValue(value);
    gammaSlider_->blockSignals(false);
}
void ConfigDialog::onBrightnessSliderChanged(int value)
{
    brightnessSpinBox_->blockSignals(true);
    brightnessSpinBox_->setValue(value);
    brightnessSpinBox_->blockSignals(false);
}
void ConfigDialog::onBrightnessSpinChanged(int value)
{
    brightnessSlider_->blockSignals(true);
    brightnessSlider_->setValue(value);
    brightnessSlider_->blockSignals(false);
}

void ConfigDialog::applyV4l2Settings()
{
    if (!cameraRadio_->isChecked()) {
        QMessageBox::information(this, "V4L2 Controls",
                                 "V4L2 controls are only applicable to camera sources.");
        return;
    }

    const int idx = cameraComboBox_->currentIndex();
    if (idx < 0 || idx >= availableCameraIds_.size()) {
        QMessageBox::warning(this, "No Camera", "No camera device selected.");
        return;
    }

    const int deviceId = availableCameraIds_[idx];
    const QString devPath = QString("/dev/video%1").arg(deviceId);

    // Apply settings via v4l2-ctl.  Non-fatal on partial failure.
    struct Control { const char* name; int value; };
    const QVector<Control> controls = {
        { "gain",       gainSpinBox_->value()       },
        { "gamma",      gammaSpinBox_->value()      },
        { "brightness", brightnessSpinBox_->value() },
    };

    QStringList failed;
    for (const auto& ctrl : controls) {
        QProcess p;
        p.start("v4l2-ctl", QStringList()
                << "-d" << devPath
                << "--set-ctrl" << QString("%1=%2").arg(ctrl.name).arg(ctrl.value));
        if (!p.waitForFinished(2000) || p.exitCode() != 0) {
            failed << ctrl.name;
            qWarning() << "ConfigDialog: v4l2-ctl failed for control" << ctrl.name
                       << "on" << devPath;
        }
    }

    if (failed.isEmpty()) {
        QMessageBox::information(this, "V4L2 Controls",
                                 "All controls applied successfully to " + devPath);
    } else {
        QMessageBox::warning(this, "V4L2 Controls",
                             QString("Failed to apply: %1\n\n"
                                     "Some controls may not be supported by this device.")
                             .arg(failed.join(", ")));
    }
}

void ConfigDialog::clearRoi()
{
    roi_ = QRect();
    roiInfoLabel_->setText("No ROI set — draw one in Edit mode after closing this dialog.");
    updateInfoPanel();
}

void ConfigDialog::onAccepted()
{
    // Validate selected backend is actually available
    const inference::Backend backend = getBackend();
    if (!inference::DetectorFactory::isAvailable(backend)) {
        QMessageBox::warning(this, "Backend Unavailable",
            QString("The selected backend '%1' was not compiled into this binary.\n"
                    "Please select a different backend.")
            .arg(inference::DetectorFactory::name(backend)));
        return;
    }

    // Commit live edits before validation
    if (cameraRadio_->isChecked()) {
        const int idx = cameraComboBox_->currentIndex();
        cameraDeviceId_ = (idx >= 0 && idx < availableCameraIds_.size())
                          ? availableCameraIds_[idx]
                          : 0;
    }

    yamlFilePath_ = yamlFilePathEdit_->text().trimmed();

    QString errorMsg;
    if (!validateInputs(errorMsg)) {
        QMessageBox::warning(this, "Configuration Error", errorMsg);
        return;   // Keep dialog open
    }

    saveConfig();
    accept();
}

// ─── config persistence ───────────────────────────────────────────────────────

QString ConfigDialog::configFilePath()
{
    const QString dir = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
    QDir().mkpath(dir);
    return dir + "/config.yaml";
}

void ConfigDialog::saveConfig()
{
    YAML::Node cfg;

    cfg["backend"]          = backendComboBox_->currentData().toInt();
    cfg["model_file"]       = modelFilePath_.toStdString();
    cfg["yaml_file"]        = yamlFilePath_.toStdString();
    cfg["source"]           = sourceButtonGroup_->checkedId();
    cfg["camera_device_id"] = cameraDeviceId_;
    cfg["video_file"]       = videoFilePathEdit_->text().toStdString();
    cfg["rtsp_url"]         = rtspUrlEdit_->text().toStdString();
    cfg["resolution_index"] = resolutionComboBox_->currentIndex();
    cfg["fps_index"]        = fpsComboBox_->currentIndex();
    cfg["gain"]             = gainSpinBox_->value();
    cfg["gamma"]            = gammaSpinBox_->value();
    cfg["brightness"]       = brightnessSpinBox_->value();
    cfg["conf_threshold"]   = confThresSpin_->value();
    cfg["nms_threshold"]    = nmsThresSpin_->value();
    cfg["rockchip_hw"]      = rockchipHwCheckbox_->isChecked();

    if (!classLabels_.isEmpty()) {
        YAML::Node lblNode;
        for (const QString& lbl : classLabels_)
            lblNode.push_back(lbl.toStdString());
        cfg["class_labels"] = lblNode;
    }
    cfg["roi_enabled"]      = enableRoiCheckbox_->isChecked();
    cfg["debug_logging"]    = debugLoggingCheckbox_->isChecked();

    if (roi_.isValid()) {
        cfg["roi"]["x"]      = roi_.x();
        cfg["roi"]["y"]      = roi_.y();
        cfg["roi"]["width"]  = roi_.width();
        cfg["roi"]["height"] = roi_.height();
    }

    const std::string path = configFilePath().toStdString();
    std::ofstream fout(path);
    if (!fout.is_open()) {
        qWarning() << "ConfigDialog: cannot write config to" << configFilePath();
        return;
    }
    fout << cfg;
    qDebug() << "ConfigDialog: saved config to" << configFilePath();
}

void ConfigDialog::loadConfig()
{
    const std::string path = configFilePath().toStdString();
    {
        std::ifstream probe(path);
        if (!probe.good()) return;  // first run
    }

    YAML::Node cfg;
    try {
        cfg = YAML::LoadFile(path);
    }
    catch (const YAML::Exception& e) {
        qWarning() << "ConfigDialog: failed to parse config YAML:" << e.what();
        return;
    }

    // Backend
    if (cfg["backend"]) {
        const int backendVal = cfg["backend"].as<int>();
        for (int i = 0; i < backendComboBox_->count(); ++i) {
            if (backendComboBox_->itemData(i).toInt() == backendVal) {
                backendComboBox_->setCurrentIndex(i);
                break;
            }
        }
    }

    // Model
    if (cfg["model_file"]) {
        modelFilePath_ = QString::fromStdString(cfg["model_file"].as<std::string>());
        if (!modelFilePath_.isEmpty() && QFile::exists(modelFilePath_)) {
            modelFilePathEdit_->setText(QFileInfo(modelFilePath_).fileName());
            modelFilePathEdit_->setToolTip(modelFilePath_);
        } else {
            modelFilePath_.clear();
        }
    }
    if (cfg["yaml_file"]) {
        yamlFilePath_ = QString::fromStdString(cfg["yaml_file"].as<std::string>());
        yamlFilePathEdit_->setText(yamlFilePath_);
    }

    // Source
    if (cfg["source"]) {
        const int srcId = cfg["source"].as<int>(0);
        if (auto* btn = sourceButtonGroup_->button(srcId))
            btn->setChecked(true);
    }

    // Camera
    if (cfg["camera_device_id"]) {
        cameraDeviceId_ = cfg["camera_device_id"].as<int>(0);
        for (int i = 0; i < availableCameraIds_.size(); ++i) {
            if (availableCameraIds_[i] == cameraDeviceId_) {
                cameraComboBox_->setCurrentIndex(i);
                break;
            }
        }
    }

    // Video / RTSP
    if (cfg["video_file"])
        videoFilePathEdit_->setText(QString::fromStdString(cfg["video_file"].as<std::string>()));
    if (cfg["rtsp_url"])
        rtspUrlEdit_->setText(QString::fromStdString(cfg["rtsp_url"].as<std::string>()));

    // Resolution / FPS
    if (cfg["resolution_index"])
        resolutionComboBox_->setCurrentIndex(cfg["resolution_index"].as<int>(0));
    if (cfg["fps_index"])
        fpsComboBox_->setCurrentIndex(cfg["fps_index"].as<int>(2));

    // V4L2 controls
    if (cfg["gain"])       gainSpinBox_->setValue(cfg["gain"].as<int>(50));
    if (cfg["gamma"])      gammaSpinBox_->setValue(cfg["gamma"].as<int>(100));
    if (cfg["brightness"]) brightnessSpinBox_->setValue(cfg["brightness"].as<int>(0));

    // Thresholds
    if (cfg["conf_threshold"])
        confThresSpin_->setValue(cfg["conf_threshold"].as<double>(0.25));
    if (cfg["nms_threshold"])
        nmsThresSpin_->setValue(cfg["nms_threshold"].as<double>(0.45));

    // Class labels
    if (cfg["class_labels"]) {
        classLabels_.clear();
        for (const auto& n : cfg["class_labels"])
            classLabels_ << QString::fromStdString(n.as<std::string>());
    }

    // ROI
    if (cfg["roi_enabled"])
        enableRoiCheckbox_->setChecked(cfg["roi_enabled"].as<bool>(false));
    if (cfg["roi"]) {
        const YAML::Node& r = cfg["roi"];
        roi_ = QRect(r["x"].as<int>(0), r["y"].as<int>(0),
                     r["width"].as<int>(0), r["height"].as<int>(0));
        if (roi_.isValid())
            roiInfoLabel_->setText(QString("ROI: (%1,%2) %3×%4")
                .arg(roi_.x()).arg(roi_.y()).arg(roi_.width()).arg(roi_.height()));
    }

    // Rockchip / Debug
    if (cfg["rockchip_hw"])
        rockchipHwCheckbox_->setChecked(cfg["rockchip_hw"].as<bool>(false));
    if (cfg["debug_logging"])
        debugLoggingCheckbox_->setChecked(cfg["debug_logging"].as<bool>(false));
}
