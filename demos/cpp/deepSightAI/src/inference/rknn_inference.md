# Plan: Dynamic RKNN Inference System

## Objective
Modernize RknnDetector to be dynamic, high-performance, and extensible.

## 1. Dynamic Model Handling
- Query model input attributes (width, height, channels, format) after rknn_init.
- No hardcoded dimensions.

## 2. Hardware Accelerated Pre-processing
- RGA: Use librga for hardware-accelerated image scaling and format conversion (BGR -> RGB).
- Fallback: Implement cv::resize as a robust software fallback.

## 3. Extensible Model Support
- Support different models (EdgeYOLO, RetinaFace, etc.).
- Abstract post-processing logic into a registry or plugin system.

## 4. Performance
- Maintain zero-copy DMA buffer flow.
- Optimize tensor memory mapping.
