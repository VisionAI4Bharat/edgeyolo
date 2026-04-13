# Qt6 GUI Application for EdgeYOLO

## Context
The user wants to create a Qt6-based GUI application for the EdgeYOLO object detection system. The application should:
1. Provide dialogs to load configuration, model, and video/camera device
2. Populate camera list using v4l2-ctl or similar
3. Support Qt6 with fallback to Qt5
4. Fetch and compile opencv-mobile on first run, saving output for future use in thirdparty/
5. Use FetchContent for dependencies where needed
6. Have a split-pane UI:
   - Left pane: Video display with editable bounding box (1m long, blue) created via mouse
   - Right pane: Control buttons (Edit/Run mode), performance metrics (FPS, latency), time display, and object selection checkboxes
7. Save bounding box configuration to config file
8. Use ONNX EdgeYOLO model format
9. Be a standalone CMake-based application with no external dependencies (use FetchContent)

## Implementation Approach

### 1. Project Structure
```
demos/cpp/edgeyolo_qt_gui/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── mainwindow.cpp
│   ├── mainwindow.h
│   ├── videowidget.cpp
│   ├── videowidget.h
│   ├── configdialog.cpp
│   ├── configdialog.h
│   ├── modeldialog.cpp
│   ├── modeldialog.h
│   ├── cameradialog.cpp
│   └── cameradialog.h
├── thirdparty/           # Will contain fetched OpenCV and other dependencies
└── resources/            # Icons, UI files if needed
```

### 2. Key Components

#### MainWindow
- Central widget with QSplitter (horizontal)
- Left pane: VideoWidget (custom QWidget for video display and bounding box editing)
- Right pane: Control panel with:
  - Edit/Run mode buttons
  - Performance metrics display (FPS, inference latency, NMS latency, average latency)
  - Current time display
  - Object selection checkboxes (dynamically populated from model config)
  - Status bar

#### VideoWidget (Custom QWidget)
- Handles video frame display using QImage/QPixmap
- Mouse event handling for bounding box creation/editing in Edit mode
- Draws bounding box (blue, 1m equivalent pixels based on calibration)
- Emits signals for bounding box changes
- In Run mode: displays detection results from EdgeYOLO

#### Dialogs
- ConfigDialog: File selection for YAML config
- ModelDialog: File selection for ONNX model
- CameraDialog: Lists available V4L2 devices using v4l2-ctl or QCameraInfo

### 3. EdgeYOLO Integration
- Use existing third_party/edgeyolo/deployment/yolo/Detector class
- Wrap detection functionality in a separate thread to avoid blocking GUI
- Load model and config via the dialogs
- Process video frames from cv::VideoCapture
- Convert cv::Mat to QImage for display
- Draw detection results on video feed

### 4. Build System
- CMake with FetchContent for:
  - Qt6 (with Qt5 fallback)
  - OpenCV (opencv-mobile preferred for size)
  - yaml-cpp (for config parsing)
  - Any other needed dependencies
- Download and build opencv-mobile on first run, cache in thirdparty/
- Link against existing EdgeYOLO deployment libraries

### 5. Data Flow
1. User selects config, model, camera via dialogs
2. Application initializes EdgeYOLO Detector with selected model/config
3. Video capture thread starts, reading frames from camera
4. In Edit mode: 
   - Display raw video feed
   - Allow user to draw/edit bounding box with mouse
   - Save bounding box to config when changed
5. In Run mode:
   - Process each frame through EdgeYOLO Detector
   - Apply bounding box as region of interest (optional)
   - Draw detection results on frame
   - Calculate and display performance metrics
   - Update object detection checkboxes based on detected classes

### 6. Performance Metrics
- FPS: frames per second displayed
- Inference latency: time for model inference
- NMS latency: time for non-maximum suppression
- Average latency: moving average over last N frames (configurable, default 30)
- Current time: displayed in HH:MM:SS format

### 7. Object Selection
- Dynamically populate checkboxes from model's class names in config
- When unchecked, filter out detections of that class (set confidence to 0 or skip drawing)
- Update in real-time as user toggles checkboxes

## Verification Plan
1. Build application successfully with CMake
2. Run application and verify dialogs work
3. Verify camera detection and selection
4. Verify bounding box creation and editing in Edit mode
5. Verify mode switching (Edit/Run)
6. Verify detection results display in Run mode
7. Verify performance metrics update correctly
8. Verify object selection filtering works
9. Verify bounding box saves to and loads from config file
10. Test with actual EdgeYOLO ONNX model if available

## Inference Backends

Three backends behind a single `IDetector` interface (`src/inference/IDetector.h`).

### Model format notes
- **Your ONNX model** (`edgeyolo_person_forklift_tiny_416_opset13_416x416_b16.onnx`):
  - Input: `[16, 3, 416, 416]` — at runtime batch=1 (first slice)
  - Output: `[16, 3549, 7]` — **fused head**, `7 = cx, cy, w, h, obj_conf, cls0, cls1`
  - Post-processing: mirrors `mnn_det::YOLO::generate_yolo_proposals` (obj_conf × cls_score, threshold, scale by letterbox factor, NMS)
- **RKNN model** (`.rknn`):
  - **Decoupled 9-output INT8 head** — post-processing already inside `RKNN::YOLO`
  - Target: RV1106, single NPU core (`RKNN_NPU_CORE_0`)

### Backend: ONNX Runtime (default, desktop)
- Uses `onnxruntime` C++ API (`Ort::Session`)
- Runs INT8 quantised ONNX natively on CPU EP
- Pre-processing: `detect::resizeAndPad` from `common.hpp`
- Sidecar `.yaml` (same base name as `.onnx`) provides class names and img_size
- **Do NOT use OpenCV dnn** — use ONNX Runtime

### Backend: OpenVINO (optional, desktop/Intel)
- `ov::Core::compile_model()` loads the same `.onnx` directly
- Same pre/post-processing as ONNX Runtime backend
- Compiled in with `-DWITH_OPENVINO=ON`

### Backend: RKNN (optional, RV1106)
- Wraps `RKNN::YOLO` from `third_party/edgeyolo/cpp/rknn/include/image_utils/rknn.h`
- Compiled in with `-DWITH_RKNN=ON`

### File layout
```
src/inference/
  IDetector.h           ← pure abstract: load / infer / classNames / inputSize
  DetectorFactory.h/.cpp ← enum Backend{ONNX,OPENVINO,RKNN}; create(...)
  OnnxDetector.h/.cpp   ← ONNX Runtime
  OpenVinoDetector.h/.cpp ← OpenVINO (guarded WITH_OPENVINO)
  RknnDetector.h/.cpp   ← RKNN (guarded WITH_RKNN)
```

## Files to Reference/Reuse
- Fused head post-processing: `third_party/edgeyolo/cpp/mnn/include/image_utils/mnn.h` (`generate_yolo_proposals`)
- Decoupled INT8 head: `third_party/edgeyolo/cpp/rknn/include/image_utils/rknn.h` (`RKNN::YOLO`)
- Common pre/post utils: `third_party/edgeyolo/deployment/yolo/common.hpp` (`resizeAndPad`, `nms_sorted_bboxes`, `Object`)
- RKNN API: `third_party/edgeyolo/cpp/rknn/include/rknn_api.h`
- Camera examples: `third_party/edgeyolo/cpp/rknn/src/demo.cpp`