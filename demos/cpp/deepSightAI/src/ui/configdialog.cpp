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

#include "configdialog.h"
#include "../app_config.h"
#include "../capture/ICapture.h"

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

    dsai_setupBackendSection(mainLayout);
    mainLayout->addWidget(dsai_makeSeparator());
    dsai_setupModelSection(mainLayout);
    mainLayout->addWidget(dsai_makeSeparator());
    dsai_setupSourceSection(mainLayout);
    dsai_setupCameraSection(mainLayout);
    dsai_setupVideoFileSection(mainLayout);
    dsai_setupRtspSection(mainLayout);
    dsai_setupRockchipSection(mainLayout);
    mainLayout->addWidget(dsai_makeSeparator());
    dsai_setupResolutionSection(mainLayout);
    dsai_setupFpsSection(mainLayout);
    mainLayout->addWidget(dsai_makeSeparator());
    dsai_setupV4l2Section(mainLayout);
    mainLayout->addWidget(dsai_makeSeparator());
    dsai_setupThresholdsSection(mainLayout);
    mainLayout->addWidget(dsai_makeSeparator());
    dsai_setupRoiSection(mainLayout);
    mainLayout->addWidget(dsai_makeSeparator());
    dsai_setupInfoSection(mainLayout);
    mainLayout->addWidget(dsai_makeSeparator());
    dsai_setupDebugSection(mainLayout);
    mainLayout->addStretch();

    scroll->setWidget(container);

    QVBoxLayout* dialogLayout = new QVBoxLayout(this);
    dialogLayout->setContentsMargins(0, 0, 0, 0);
    dialogLayout->addWidget(scroll);
    dsai_setupButtonRow(dialogLayout);

    // Wire up OK / Cancel
    connect(okButton_,     &QPushButton::clicked, this, &ConfigDialog::dsai_onAccepted);
    connect(cancelButton_, &QPushButton::clicked, this, &QDialog::reject);

    dsai_loadConfig();
    dsai_updateSourceVisibility();
    dsai_updateInfoPanel();
}

// ─── section builders ─────────────────────────────────────────────────────────

inference::Backend ConfigDialog::dsai_getBackend() const
{
    if (!backendComboBox_) return inference::Backend::ONNX;
    const int idx = backendComboBox_->currentData().toInt();
    return static_cast<inference::Backend>(idx);
}

void ConfigDialog::dsai_setupBackendSection(QVBoxLayout* parent)
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
        if (!inference::DetectorFactory::dsai_isAvailable(inference::Backend::OPENVINO)) {
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
        if (!inference::DetectorFactory::dsai_isAvailable(inference::Backend::RKNN)) {
            auto* model = qobject_cast<QStandardItemModel*>(backendComboBox_->model());
            if (model) model->item(idx)->setEnabled(false);
        }
    }

    lay->addWidget(new QLabel("Backend:", this));
    lay->addWidget(backendComboBox_, 1);
    parent->addWidget(backendGroup_);
}

void ConfigDialog::dsai_setupModelSection(QVBoxLayout* parent)
{
    modelGroup_ = new QGroupBox("Model", this);
    QFormLayout* form = new QFormLayout(modelGroup_);

    // Model file row
    modelFilePathEdit_ = new QLineEdit(this);
    modelFilePathEdit_->setReadOnly(true);
    modelFilePathEdit_->setPlaceholderText("Select model file (.onnx / .xml / .rknn)…");
    connect(backendComboBox_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
        const inference::Backend b = dsai_getBackend();
        if (b == inference::Backend::RKNN)
            modelFilePathEdit_->setPlaceholderText("Select RKNN INT8 model (.rknn)…");
        else if (b == inference::Backend::OPENVINO)
            modelFilePathEdit_->setPlaceholderText("Select OpenVINO IR model (.xml)…");
        else
            modelFilePathEdit_->setPlaceholderText("Select ONNX model (.onnx)…");
    });
    browseModelButton_ = new QPushButton("Browse…", this);
    connect(browseModelButton_, &QPushButton::clicked, this, &ConfigDialog::dsai_browseModelFile);

    QHBoxLayout* modelRow = new QHBoxLayout();
    modelRow->addWidget(modelFilePathEdit_, 1);
    modelRow->addWidget(browseModelButton_);
    form->addRow("Model file:", modelRow);

    // YAML config row
    yamlFilePathEdit_ = new QLineEdit(this);
    yamlFilePathEdit_->setPlaceholderText("Optional — auto-detected if empty");
    browseYamlButton_ = new QPushButton("Browse…", this);
    connect(browseYamlButton_, &QPushButton::clicked, this, &ConfigDialog::dsai_browseYamlFile);

    QHBoxLayout* yamlRow = new QHBoxLayout();
    yamlRow->addWidget(yamlFilePathEdit_, 1);
    yamlRow->addWidget(browseYamlButton_);
    form->addRow("Class config YAML:", yamlRow);

    editClassLabelsBtn_ = new QPushButton("Edit Class Labels…", this);
    connect(editClassLabelsBtn_, &QPushButton::clicked, this, &ConfigDialog::dsai_openClassLabelsEditor);
    form->addRow("", editClassLabelsBtn_);

    parent->addWidget(modelGroup_);
}

void ConfigDialog::dsai_setupSourceSection(QVBoxLayout* parent)
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
            this, [this](int){ dsai_onSourceChanged(); });

    parent->addWidget(sourceGroup_);
}

void ConfigDialog::dsai_setupCameraSection(QVBoxLayout* parent)
{
    cameraGroup_ = new QGroupBox("Camera Device", this);
    QVBoxLayout* vlay = new QVBoxLayout(cameraGroup_);

    cameraComboBox_       = new QComboBox(this);
    refreshCamerasButton_ = new QPushButton("Refresh", this);

    connect(refreshCamerasButton_, &QPushButton::clicked, this, &ConfigDialog::dsai_refreshCameras);
    connect(cameraComboBox_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int idx) {
                int devId = (idx >= 0 && idx < availableCameraIds_.size())
                            ? availableCameraIds_[idx] : 0;
                dsai_populateResolutionCombo(devId);
            });

    QHBoxLayout* lay = new QHBoxLayout();
    lay->addWidget(cameraComboBox_, 1);
    lay->addWidget(refreshCamerasButton_);
    vlay->addLayout(lay);

    parent->addWidget(cameraGroup_);

    dsai_populateCameras();
}

void ConfigDialog::dsai_setupVideoFileSection(QVBoxLayout* parent)
{
    videoFileGroup_ = new QGroupBox("Video File", this);
    QHBoxLayout* lay = new QHBoxLayout(videoFileGroup_);

    videoFilePathEdit_ = new QLineEdit(this);
    videoFilePathEdit_->setPlaceholderText("Path to .mp4, .avi, .mkv…");
    browseVideoButton_ = new QPushButton("Browse…", this);

    connect(browseVideoButton_, &QPushButton::clicked, this, &ConfigDialog::dsai_browseVideoFile);

    lay->addWidget(videoFilePathEdit_, 1);
    lay->addWidget(browseVideoButton_);
    parent->addWidget(videoFileGroup_);
}

void ConfigDialog::dsai_setupRtspSection(QVBoxLayout* parent)
{
    rtspGroup_ = new QGroupBox("RTSP Stream", this);
    QHBoxLayout* lay = new QHBoxLayout(rtspGroup_);

    rtspUrlEdit_ = new QLineEdit(this);
    rtspUrlEdit_->setPlaceholderText("rtsp://user:pass@host:port/path");

    lay->addWidget(rtspUrlEdit_);
    parent->addWidget(rtspGroup_);
}

void ConfigDialog::dsai_setupRockchipSection(QVBoxLayout* parent)
{
    rockchipHwCheckbox_ = new QCheckBox("Rockchip Hardware", this);
    rockchipHwCheckbox_->setToolTip(
        "When checked, selects the RKNN backend and enables Rockchip NPU inference.\n"
        "Only applicable for Camera and RTSP sources.\n"
        "Requires WITH_RKNN compiled in and a .rknn model file.");
    parent->addWidget(rockchipHwCheckbox_);
}

void ConfigDialog::dsai_setupResolutionSection(QVBoxLayout* parent)
{
    resolutionGroup_ = new QGroupBox("Resolution", this);
    QHBoxLayout* lay = new QHBoxLayout(resolutionGroup_);

    resolutionComboBox_ = new QComboBox(this);
    dsai_populateResolutionCombo(0);  // populate from V4L2 device 0 on construction

    connect(resolutionComboBox_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ConfigDialog::dsai_onResolutionChanged);

    lay->addWidget(resolutionComboBox_);
    lay->addStretch();
    parent->addWidget(resolutionGroup_);
}

void ConfigDialog::dsai_setupFpsSection(QVBoxLayout* parent)
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

void ConfigDialog::dsai_setupV4l2Section(QVBoxLayout* parent)
{
    v4l2ControlsGroup_ = new QGroupBox("Camera Controls (V4L2)", this);
    QFormLayout* form = new QFormLayout(v4l2ControlsGroup_);

    auto makeRow = [&](const QString& label, QSlider*& slider, QSpinBox*& spin,
                       int minVal, int maxVal, int defaultVal)
    {
        QHBoxLayout* row = dsai_makeSliderRow(slider, spin, minVal, maxVal, defaultVal);
        form->addRow(label, row);
    };

    makeRow("Gain:",       gainSlider_,       gainSpinBox_,       0, 255, 50);
    makeRow("Gamma:",      gammaSlider_,      gammaSpinBox_,      72, 500, 100);
    makeRow("Brightness:", brightnessSlider_, brightnessSpinBox_, -64, 64, 0);

    // Slider ↔ spin synchronisation
    connect(gainSlider_,       &QSlider::valueChanged, this, &ConfigDialog::dsai_onGainSliderChanged);
    connect(gainSpinBox_,      QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ConfigDialog::dsai_onGainSpinChanged);
    connect(gammaSlider_,      &QSlider::valueChanged, this, &ConfigDialog::dsai_onGammaSliderChanged);
    connect(gammaSpinBox_,     QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ConfigDialog::dsai_onGammaSpinChanged);
    connect(brightnessSlider_, &QSlider::valueChanged, this, &ConfigDialog::dsai_onBrightnessSliderChanged);
    connect(brightnessSpinBox_,QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ConfigDialog::dsai_onBrightnessSpinChanged);

    applyV4l2Button_ = new QPushButton("Apply to Device", this);
    connect(applyV4l2Button_, &QPushButton::clicked, this, &ConfigDialog::dsai_applyV4l2Settings);
    form->addRow("", applyV4l2Button_);

    parent->addWidget(v4l2ControlsGroup_);
}

void ConfigDialog::dsai_setupThresholdsSection(QVBoxLayout* parent)
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

void ConfigDialog::dsai_setupRoiSection(QVBoxLayout* parent)
{
    roiGroup_ = new QGroupBox("Region of Interest (ROI)", this);
    QVBoxLayout* vlay = new QVBoxLayout(roiGroup_);

    enableRoiCheckbox_ = new QCheckBox("Enable ROI filtering", this);
    roiInfoLabel_      = new QLabel("No ROI set — draw one in Edit mode after closing this dialog.", this);
    roiInfoLabel_->setWordWrap(true);
    clearRoiButton_    = new QPushButton("Clear ROI", this);

    connect(clearRoiButton_, &QPushButton::clicked, this, &ConfigDialog::dsai_clearRoi);

    vlay->addWidget(enableRoiCheckbox_);
    vlay->addWidget(roiInfoLabel_);
    vlay->addWidget(clearRoiButton_);
    parent->addWidget(roiGroup_);
}

void ConfigDialog::dsai_setupInfoSection(QVBoxLayout* parent)
{
    infoGroup_ = new QGroupBox("Information", this);
    QVBoxLayout* vlay = new QVBoxLayout(infoGroup_);

    infoTextBrowser_ = new QTextBrowser(this);
    infoTextBrowser_->setMaximumHeight(100);
    infoTextBrowser_->setOpenExternalLinks(false);

    vlay->addWidget(infoTextBrowser_);
    parent->addWidget(infoGroup_);
}

void ConfigDialog::dsai_setupDebugSection(QVBoxLayout* parent)
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

void ConfigDialog::dsai_setupButtonRow(QVBoxLayout* parent)
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

QWidget* ConfigDialog::dsai_makeSeparator()
{
    QFrame* line = new QFrame();
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);
    return line;
}

QHBoxLayout* ConfigDialog::dsai_makeSliderRow(QSlider*& slider, QSpinBox*& spin,
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

void ConfigDialog::dsai_populateCameras()
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

void ConfigDialog::dsai_populateResolutionCombo(int devId) {
    resolutionComboBox_->blockSignals(true);
    resolutionComboBox_->clear();

    const auto modes = deepSightAI::ICapture::dsai_enumerateModes(devId);
    for (const auto& m : modes) {
        QString label = QString("%1 × %2 @ %3 fps")
                        .arg(m.width).arg(m.height).arg(m.fps);
        resolutionComboBox_->addItem(label,
            QVariant::fromValue(QSize(m.width, m.height)));
    }

    if (modes.empty()) {
        // V4L2 enumeration unavailable — offer common sizes as fallback
        for (auto [w, h] : std::initializer_list<std::pair<int,int>>{
                {640,480},{1280,720},{1920,1080}})
            resolutionComboBox_->addItem(
                QString("%1 × %2").arg(w).arg(h),
                QVariant::fromValue(QSize(w, h)));
    }
    resolutionComboBox_->blockSignals(false);
}

void ConfigDialog::dsai_updateSourceVisibility()
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

void ConfigDialog::dsai_updateInfoPanel()
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

    const int fps = dsai_getFps();
    lines << QString("FPS: %1").arg(fps < 0 ? "Max" : QString::number(fps));

    if (enableRoiCheckbox_->isChecked() && roi_.isValid())
        lines << QString("ROI: (%1,%2) %3×%4")
                     .arg(roi_.x()).arg(roi_.y())
                     .arg(roi_.width()).arg(roi_.height());

    infoTextBrowser_->setHtml(lines.join("<br>"));
}

bool ConfigDialog::dsai_validateInputs(QString& errorMsg) const
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

QString ConfigDialog::dsai_getVideoOrRtspPath() const
{
    if (videoFileRadio_->isChecked())
        return videoFilePathEdit_->text().trimmed();
    if (rtspRadio_->isChecked())
        return rtspUrlEdit_->text().trimmed();
    return {};
}

int ConfigDialog::dsai_getFps() const
{
    if (!fpsComboBox_)
        return 30;
    const QVariant v = fpsComboBox_->currentData();
    return v.isValid() ? v.toInt() : 30;
}

int ConfigDialog::dsai_getWidth() const
{
    if (!resolutionComboBox_)
        return 640;
    const QSize s = resolutionComboBox_->currentData().toSize();
    return (s.isValid() && s.width() > 0) ? s.width() : 640;
}

int ConfigDialog::dsai_getHeight() const
{
    if (!resolutionComboBox_)
        return 480;
    const QSize s = resolutionComboBox_->currentData().toSize();
    return (s.isValid() && s.height() > 0) ? s.height() : 480;
}

// ─── slots ────────────────────────────────────────────────────────────────────

void ConfigDialog::dsai_browseModelFile()
{
    const inference::Backend backend = dsai_getBackend();

    QString filter, title;
    switch (backend) {
        case inference::Backend::RKNN:
            filter = "RKNN INT8 Models (*.rknn);;All Files (*)";
            title  = "Select RKNN INT8 Model";
            break;
        case inference::Backend::OPENVINO:
            filter = "OpenVINO IR Models (*.xml);;All Files (*)";
            title  = "Select OpenVINO IR Model (.xml)";
            break;
        case inference::Backend::ONNX:
        default:
            filter = "ONNX Models (*.onnx);;All Files (*)";
            title  = "Select ONNX Model";
            break;
    }

    const QString path = QFileDialog::getOpenFileName(
        this,
        title,
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

    dsai_updateInfoPanel();
}

void ConfigDialog::dsai_browseYamlFile()
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
    dsai_updateInfoPanel();
}

void ConfigDialog::dsai_openClassLabelsEditor()
{
    ClassLabelsDialog dlg(classLabels_, yamlFilePathEdit_->text().trimmed(), this);
    if (dlg.exec() == QDialog::Accepted)
        classLabels_ = dlg.dsai_getClassLabels();
}

void ConfigDialog::dsai_browseVideoFile()
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
    dsai_updateInfoPanel();
}

void ConfigDialog::dsai_onSourceChanged()
{
    dsai_updateSourceVisibility();
    dsai_updateInfoPanel();
}

void ConfigDialog::dsai_refreshCameras()
{
    dsai_populateCameras();
    dsai_updateInfoPanel();
}

void ConfigDialog::dsai_onResolutionChanged(int /*index*/)
{
    dsai_updateInfoPanel();
}

// Slider ↔ spin sync — use blockSignals to avoid infinite recursion
void ConfigDialog::dsai_onGainSliderChanged(int value)
{
    gainSpinBox_->blockSignals(true);
    gainSpinBox_->setValue(value);
    gainSpinBox_->blockSignals(false);
}
void ConfigDialog::dsai_onGainSpinChanged(int value)
{
    gainSlider_->blockSignals(true);
    gainSlider_->setValue(value);
    gainSlider_->blockSignals(false);
}
void ConfigDialog::dsai_onGammaSliderChanged(int value)
{
    gammaSpinBox_->blockSignals(true);
    gammaSpinBox_->setValue(value);
    gammaSpinBox_->blockSignals(false);
}
void ConfigDialog::dsai_onGammaSpinChanged(int value)
{
    gammaSlider_->blockSignals(true);
    gammaSlider_->setValue(value);
    gammaSlider_->blockSignals(false);
}
void ConfigDialog::dsai_onBrightnessSliderChanged(int value)
{
    brightnessSpinBox_->blockSignals(true);
    brightnessSpinBox_->setValue(value);
    brightnessSpinBox_->blockSignals(false);
}
void ConfigDialog::dsai_onBrightnessSpinChanged(int value)
{
    brightnessSlider_->blockSignals(true);
    brightnessSlider_->setValue(value);
    brightnessSlider_->blockSignals(false);
}

void ConfigDialog::dsai_applyV4l2Settings()
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

void ConfigDialog::dsai_clearRoi()
{
    roi_ = QRect();
    roiInfoLabel_->setText("No ROI set — draw one in Edit mode after closing this dialog.");
    dsai_updateInfoPanel();
}

void ConfigDialog::dsai_onAccepted()
{
    // Validate selected backend is actually available
    const inference::Backend backend = dsai_getBackend();
    if (!inference::DetectorFactory::dsai_isAvailable(backend)) {
        QMessageBox::warning(this, "Backend Unavailable",
            QString("The selected backend '%1' was not compiled into this binary.\n"
                    "Please select a different backend.")
            .arg(inference::DetectorFactory::dsai_name(backend)));
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
    if (!dsai_validateInputs(errorMsg)) {
        QMessageBox::warning(this, "Configuration Error", errorMsg);
        return;   // Keep dialog open
    }

    dsai_saveConfig();
    accept();
}

// ─── config persistence ───────────────────────────────────────────────────────

QString ConfigDialog::dsai_configFilePath()
{
    return QString::fromStdString(AppConfig::dsai_defaultPath());
}

void ConfigDialog::dsai_saveConfig()
{
    AppConfig cfg;

    cfg.backend          = static_cast<Backend>(dsai_getBackend());
    cfg.modelFile        = modelFilePath_.toStdString();
    cfg.yamlFile         = yamlFilePath_.toStdString();
    cfg.source           = static_cast<SourceType>(sourceButtonGroup_->checkedId());
    cfg.cameraDeviceId   = cameraDeviceId_;
    cfg.videoFile        = videoFilePathEdit_->text().toStdString();
    cfg.rtspUrl          = rtspUrlEdit_->text().toStdString();
    cfg.captureWidth  = dsai_getWidth();
    cfg.captureHeight = dsai_getHeight();
    cfg.captureFps    = dsai_getFps();
    cfg.gain             = gainSpinBox_->value();
    cfg.gamma            = gammaSpinBox_->value();
    cfg.brightness       = brightnessSpinBox_->value();
    cfg.confThreshold    = static_cast<float>(confThresSpin_->value());
    cfg.nmsThreshold     = static_cast<float>(nmsThresSpin_->value());
    cfg.rockchipHw       = rockchipHwCheckbox_->isChecked();
    cfg.roiEnabled       = enableRoiCheckbox_->isChecked();
    cfg.debugLogging     = debugLoggingCheckbox_->isChecked();

    for (const QString& lbl : classLabels_)
        cfg.classLabels.push_back(lbl.toStdString());

    if (roi_.isValid()) {
        cfg.roi.x      = roi_.x();
        cfg.roi.y      = roi_.y();
        cfg.roi.width  = roi_.width();
        cfg.roi.height = roi_.height();
    }

    try {
        cfg.dsai_saveToFile(dsai_configFilePath().toStdString());
        qDebug() << "ConfigDialog: saved config to" << dsai_configFilePath();
    } catch (const std::exception& e) {
        qWarning() << "ConfigDialog: failed to save config:" << e.what();
    }
}


void ConfigDialog::dsai_loadConfig()
{
    const std::string path = dsai_configFilePath().toStdString();
    if (!std::filesystem::exists(path)) return;

    AppConfig cfg;
    try {
        cfg = AppConfig::dsai_loadFromFile(path);
    }
    catch (const std::exception& e) {
        qWarning() << "ConfigDialog: failed to load config:" << e.what();
        return;
    }

    // Backend
    const int backendVal = static_cast<int>(cfg.backend);
    for (int i = 0; i < backendComboBox_->count(); ++i) {
        if (backendComboBox_->itemData(i).toInt() == backendVal) {
            backendComboBox_->setCurrentIndex(i);
            break;
        }
    }

    // Model
    modelFilePath_ = QString::fromStdString(cfg.modelFile);
    if (!modelFilePath_.isEmpty() && QFile::exists(modelFilePath_)) {
        modelFilePathEdit_->setText(QFileInfo(modelFilePath_).fileName());
        modelFilePathEdit_->setToolTip(modelFilePath_);
    } else {
        modelFilePath_.clear();
    }
    yamlFilePath_ = QString::fromStdString(cfg.yamlFile);
    yamlFilePathEdit_->setText(yamlFilePath_);

    // Source
    if (auto* btn = sourceButtonGroup_->button(static_cast<int>(cfg.source)))
        btn->setChecked(true);

    // Camera
    cameraDeviceId_ = cfg.cameraDeviceId;
    for (int i = 0; i < availableCameraIds_.size(); ++i) {
        if (availableCameraIds_[i] == cameraDeviceId_) {
            cameraComboBox_->setCurrentIndex(i);
            break;
        }
    }

    // Video / RTSP
    videoFilePathEdit_->setText(QString::fromStdString(cfg.videoFile));
    rtspUrlEdit_->setText(QString::fromStdString(cfg.rtspUrl));

    // Resolution — find the entry matching captureWidth × captureHeight
    for (int i = 0; i < resolutionComboBox_->count(); ++i) {
        QSize s = resolutionComboBox_->itemData(i).toSize();
        if (s.width() == cfg.captureWidth && s.height() == cfg.captureHeight) {
            resolutionComboBox_->setCurrentIndex(i);
            break;
        }
    }
    // FPS — find closest match
    for (int i = 0; i < fpsComboBox_->count(); ++i) {
        if (fpsComboBox_->itemData(i).toInt() == cfg.captureFps) {
            fpsComboBox_->setCurrentIndex(i);
            break;
        }
    }

    // V4L2 controls
    gainSpinBox_->setValue(cfg.gain);
    gammaSpinBox_->setValue(cfg.gamma);
    brightnessSpinBox_->setValue(cfg.brightness);

    // Thresholds
    confThresSpin_->setValue(cfg.confThreshold);
    nmsThresSpin_->setValue(cfg.nmsThreshold);

    // Class labels
    classLabels_.clear();
    for (const auto& n : cfg.classLabels)
        classLabels_ << QString::fromStdString(n);

    // ROI
    enableRoiCheckbox_->setChecked(cfg.roiEnabled);
    roi_ = QRect(cfg.roi.x, cfg.roi.y, cfg.roi.width, cfg.roi.height);
    if (roi_.isValid())
        roiInfoLabel_->setText(QString("ROI: (%1,%2) %3×%4")
            .arg(roi_.x()).arg(roi_.y()).arg(roi_.width()).arg(roi_.height()));

    // Rockchip / Debug
    rockchipHwCheckbox_->setChecked(cfg.rockchipHw);
    debugLoggingCheckbox_->setChecked(cfg.debugLogging);
}
