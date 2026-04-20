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
    QString getModelFilePath() const  { return modelFilePath_; }
    QString getYamlPath() const       { return yamlFilePath_; }
    inference::Backend getBackend() const;
    int     getCameraDeviceId() const { return cameraDeviceId_; }
    QString getVideoOrRtspPath() const;
    bool    isUsingVideoFile() const  { return videoFileRadio_->isChecked(); }
    bool    isUsingRtspStream() const { return rtspRadio_->isChecked(); }
    int     getFps() const;
    int     getWidth() const;
    int     getHeight() const;
    int     getGain() const       { return gainSpinBox_->value(); }
    int     getGamma() const      { return gammaSpinBox_->value(); }
    int     getBrightness() const { return brightnessSpinBox_->value(); }
    QRect   getRoi() const        { return roi_; }
    bool    isRoiEnabled() const  { return enableRoiCheckbox_->isChecked(); }
    bool    isDebugLoggingEnabled() const { return debugLoggingCheckbox_ && debugLoggingCheckbox_->isChecked(); }
    float        getConfThreshold() const { return confThresSpin_ ? static_cast<float>(confThresSpin_->value()) : 0.25f; }
    float        getNmsThreshold()  const { return nmsThresSpin_  ? static_cast<float>(nmsThresSpin_->value())  : 0.45f; }
    QStringList  getClassLabels()   const { return classLabels_; }

private slots:
    void browseModelFile();
    void browseYamlFile();
    void openClassLabelsEditor();
    void browseVideoFile();
    void onSourceChanged();
    void refreshCameras();
    void onResolutionChanged(int index);
    void onGainSliderChanged(int value);
    void onGainSpinChanged(int value);
    void onGammaSliderChanged(int value);
    void onGammaSpinChanged(int value);
    void onBrightnessSliderChanged(int value);
    void onBrightnessSpinChanged(int value);
    void applyV4l2Settings();
    void clearRoi();
    void onAccepted();

private:
    void setupBackendSection(QVBoxLayout* parent);
    void setupModelSection(QVBoxLayout* parent);
    void setupSourceSection(QVBoxLayout* parent);
    void setupCameraSection(QVBoxLayout* parent);
    void setupVideoFileSection(QVBoxLayout* parent);
    void setupRtspSection(QVBoxLayout* parent);
    void setupResolutionSection(QVBoxLayout* parent);
    void setupV4l2Section(QVBoxLayout* parent);
    void setupFpsSection(QVBoxLayout* parent);
    void setupRoiSection(QVBoxLayout* parent);
    void setupInfoSection(QVBoxLayout* parent);
    void setupThresholdsSection(QVBoxLayout* parent);
    void setupDebugSection(QVBoxLayout* parent);
    void setupButtonRow(QVBoxLayout* parent);

    static QWidget* makeSeparator();
    static QHBoxLayout* makeSliderRow(QSlider*& slider, QSpinBox*& spin,
                                      int minVal, int maxVal, int defaultVal);
    void populateCameras();
    void updateSourceVisibility();
    void updateInfoPanel();
    bool validateInputs(QString& errorMsg) const;
    void loadConfig();
    void saveConfig();
    static QString configFilePath();

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
