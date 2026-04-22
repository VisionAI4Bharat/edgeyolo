# Plan: High-Performance Capture System for RV1106

## Objective
Implement a professional, multi-platform capture system that leverages hardware acceleration on the Rockchip RV1106 (RKMPI, VENC, RGN/OSD) while maintaining compatibility with other platforms via an abstract interface.

## 1. Abstract Interface (ICapture)
- **File**: src/capture/ICapture.h
- **Methods**:
    - dsai_openCamera(): Initialise platform camera.
    - dsai_openRtsp(): Open RTSP source.
    - dsai_read(cv::Mat& frame): Get the next frame.
    - dsai_release(): Cleanup.
    - dsai_setOSD(const Detections& results): Update hardware overlays (RV1106 specific).
    - Implementation of a production-grade, threaded capture system specifically for the Rockchip RV1106.
## 2. RV1106 Implementation (RockchipCapture)
- **Threaded Architecture**:
    - Capture Thread: Pulls frames from VI channel 0 (high-res) and channel 1 (inference-res).
    - Inference Thread: Consumes frames from channel 1.
    - OSD Thread: Updates RGN (Region) bitmaps based on inference results.
    - RTSP Thread: Serves the encoded (VENC) stream via rtsp_demo.
    
- **Hardware Acceleration**:
    - Use RK_MPI_SYS_Bind to link VI directly to VENC.
    - Use OVERLAY_RGN for zero-CPU bounding boxes.
- **Dynamic Config**:
    - Port and Path read from AppConfig.

## 3. Configuration Updates
- Add rtsp_port (default 8554) and rtsp_path (default "/live/0") to AppConfig.
# Plan: RV1106 High-Performance Capture Library                                                            │ │
│ │                                                                                                            │ │ 
│ │ ## 1. Architectural Overview                                                                               │ │ 
│ │ Implementation of a production-grade, threaded capture system specifically for the Rockchip RV1106. It     │ │ 
│ │ will be compiled as a shared library (\`libdeepSightAI_capture.so\`) and provide a unified \`ICapture\`    │ │ 
│ │ interface.                                                                                                 │ │ 
│ │                                                                                                            │ │ 
│ │ ## 2. Multi-Threaded Model                                                                                 │ │ 
│ │ To achieve maximum speed, the system will use three dedicated threads:                                     │ │ 
│ │ - **Inference Thread**: Pulls raw YUV frames from VI Channel 1, performs RGA-accelerated BGR conversion,   │ │ 
│ │ and updates the \`currentFrame\` Mat.                                                                      │ │ 
│ │ - **RTSP Thread**: Pulls encoded H.265 packets from VENC Channel 0 and serves them via \`rtsp_demo\`.      │ │ 
│ │ - **OSD/RGN Thread**: Consumes detection results and updates Rockchip Region (RGN) bitmaps to draw         │ │ 
│ │ hardware-accelerated bounding boxes on the VENC stream.                                                    │ │ 
│ │                                                                                                            │ ││ │ - **Logging**: Comprehensive \`DBG_LOG\` coverage for every stage (Init, Link, Process, Release).