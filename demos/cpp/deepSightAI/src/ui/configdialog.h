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

#ifndef CONFIGDIALOG_H
#define CONFIGDIALOG_H

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QLineEdit>
#include <QComboBox>
#include <QRadioButton>
#include <QButtonGroup>
#include <QTextBrowser>
#include <QSlider>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QFileDialog>
#include <QFrame>
#include <QScrollArea>
#include <QMessageBox>
#include <QSettings>
#include <QStandardItemModel>

#include "inference/DetectorFactory.h"
#include "classlabelsdialog.h"

class ConfigDialog : public QDialog {
    Q_OBJECT
public:
    explicit ConfigDialog(QWidget *parent = nullptr);
    ~ConfigDialog() override = default;

    // Getters — only valid after Accepted
    QString dsai_getModelFilePath() const  { return modelFilePath_; }
    QString dsai_getYamlPath() const       { return yamlFilePath_; }
    inference::Backend dsai_getBackend() const;
    int     dsai_getCameraDeviceId() const { return cameraDeviceId_; }
    QString dsai_getVideoOrRtspPath() const;
    bool    dsai_isUsingVideoFile() const  { return videoFileRadio_->isChecked(); }
    bool    dsai_isUsingRtspStream() const { return rtspRadio_->isChecked(); }
    int     dsai_getFps() const;
    int     dsai_getWidth() const;
    int     dsai_getHeight() const;
    int     dsai_getGain() const       { return gainSpinBox_->value(); }
    int     dsai_getGamma() const      { return gammaSpinBox_->value(); }
    int     dsai_getBrightness() const { return brightnessSpinBox_->value(); }
    QRect   dsai_getRoi() const        { return roi_; }
    bool    dsai_isRoiEnabled() const  { return enableRoiCheckbox_->isChecked(); }
    bool    dsai_isDebugLoggingEnabled()  const { return debugLoggingCheckbox_ && debugLoggingCheckbox_->isChecked(); }
    bool    dsai_isRockchipHardware()    const { return rockchipHwCheckbox_    && rockchipHwCheckbox_->isChecked(); }
    float        dsai_getConfThreshold() const { return confThresSpin_ ? static_cast<float>(confThresSpin_->value()) : 0.25f; }
    float        dsai_getNmsThreshold()  const { return nmsThresSpin_  ? static_cast<float>(nmsThresSpin_->value())  : 0.45f; }
    QStringList  dsai_getClassLabels()   const { return classLabels_; }

private slots:
    void dsai_browseModelFile();
    void dsai_browseYamlFile();
    void dsai_openClassLabelsEditor();
    void dsai_browseVideoFile();
    void dsai_onSourceChanged();
    void dsai_refreshCameras();
    void dsai_onResolutionChanged(int index);
    void dsai_onGainSliderChanged(int value);
    void dsai_onGainSpinChanged(int value);
    void dsai_onGammaSliderChanged(int value);
    void dsai_onGammaSpinChanged(int value);
    void dsai_onBrightnessSliderChanged(int value);
    void dsai_onBrightnessSpinChanged(int value);
    void dsai_applyV4l2Settings();
    void dsai_clearRoi();
    void dsai_onAccepted();

private:
    void dsai_setupBackendSection(QVBoxLayout* parent);
    void dsai_setupModelSection(QVBoxLayout* parent);
    void dsai_setupSourceSection(QVBoxLayout* parent);
    void dsai_setupCameraSection(QVBoxLayout* parent);
    void dsai_setupVideoFileSection(QVBoxLayout* parent);
    void dsai_setupRtspSection(QVBoxLayout* parent);
    void dsai_setupRockchipSection(QVBoxLayout* parent);
    void dsai_setupResolutionSection(QVBoxLayout* parent);
    void dsai_setupV4l2Section(QVBoxLayout* parent);
    void dsai_setupFpsSection(QVBoxLayout* parent);
    void dsai_setupRoiSection(QVBoxLayout* parent);
    void dsai_setupInfoSection(QVBoxLayout* parent);
    void dsai_setupThresholdsSection(QVBoxLayout* parent);
    void dsai_setupDebugSection(QVBoxLayout* parent);
    void dsai_setupButtonRow(QVBoxLayout* parent);

    static QWidget* dsai_makeSeparator();
    static QHBoxLayout* dsai_makeSliderRow(QSlider*& slider, QSpinBox*& spin,
                                      int minVal, int maxVal, int defaultVal);
    void dsai_populateCameras();
    void dsai_updateSourceVisibility();
    void dsai_updateInfoPanel();
    bool dsai_validateInputs(QString& errorMsg) const;
    void dsai_loadConfig();
    void dsai_saveConfig();
    static QString dsai_configFilePath();

    // — Backend section —
    QGroupBox* backendGroup_          = nullptr;
    QComboBox* backendComboBox_       = nullptr;

    // — Model section —
    QGroupBox*   modelGroup_          = nullptr;
    QLineEdit*   modelFilePathEdit_   = nullptr;
    QPushButton* browseModelButton_   = nullptr;
    QLineEdit*   yamlFilePathEdit_    = nullptr;
    QPushButton* browseYamlButton_    = nullptr;
    QPushButton* editClassLabelsBtn_  = nullptr;

    // — Source selection —
    QGroupBox*    sourceGroup_        = nullptr;
    QRadioButton* cameraRadio_        = nullptr;
    QRadioButton* videoFileRadio_     = nullptr;
    QRadioButton* rtspRadio_          = nullptr;
    QButtonGroup* sourceButtonGroup_  = nullptr;

    // — Camera section —
    QGroupBox*   cameraGroup_         = nullptr;
    QComboBox*   cameraComboBox_      = nullptr;
    QPushButton* refreshCamerasButton_= nullptr;

    // — Video file section —
    QGroupBox*   videoFileGroup_      = nullptr;
    QLineEdit*   videoFilePathEdit_   = nullptr;
    QPushButton* browseVideoButton_   = nullptr;

    // — RTSP section —
    QGroupBox* rtspGroup_             = nullptr;
    QLineEdit* rtspUrlEdit_           = nullptr;

    // — Resolution section —
    QGroupBox* resolutionGroup_       = nullptr;
    QComboBox* resolutionComboBox_    = nullptr;

    // — V4L2 controls —
    QGroupBox*   v4l2ControlsGroup_   = nullptr;
    QSlider*     gainSlider_          = nullptr;
    QSpinBox*    gainSpinBox_         = nullptr;
    QSlider*     gammaSlider_         = nullptr;
    QSpinBox*    gammaSpinBox_        = nullptr;
    QSlider*     brightnessSlider_    = nullptr;
    QSpinBox*    brightnessSpinBox_   = nullptr;
    QPushButton* applyV4l2Button_     = nullptr;

    // — FPS —
    QGroupBox* fpsGroup_              = nullptr;
    QComboBox* fpsComboBox_           = nullptr;

    // — ROI —
    QGroupBox*   roiGroup_            = nullptr;
    QLabel*      roiInfoLabel_        = nullptr;
    QCheckBox*   enableRoiCheckbox_   = nullptr;
    QPushButton* clearRoiButton_      = nullptr;

    // — Info panel —
    QGroupBox*    infoGroup_          = nullptr;
    QTextBrowser* infoTextBrowser_    = nullptr;

    // — Thresholds —
    QGroupBox*       thresholdsGroup_    = nullptr;
    QDoubleSpinBox*  confThresSpin_      = nullptr;
    QDoubleSpinBox*  nmsThresSpin_       = nullptr;

    // — Debug —
    QGroupBox*  debugGroup_           = nullptr;
    QCheckBox*  debugLoggingCheckbox_ = nullptr;
    QCheckBox*  rockchipHwCheckbox_   = nullptr;

    // — Dialog buttons —
    QPushButton* okButton_            = nullptr;
    QPushButton* cancelButton_        = nullptr;

    // — State —
    QString     modelFilePath_;
    QString     yamlFilePath_;
    QStringList classLabels_;
    int         cameraDeviceId_ = 0;
    QRect       roi_;

    // Camera device indices matching combo items
    QVector<int> availableCameraIds_;
};

#endif // CONFIGDIALOG_H
