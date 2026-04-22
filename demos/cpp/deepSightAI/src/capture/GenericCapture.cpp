#include "GenericCapture.h"

namespace deepSightAI {

GenericCapture::GenericCapture() {}
GenericCapture::~GenericCapture() { dsai_release(); }

bool GenericCapture::dsai_openCamera(int devId, int width, int height, double fps) {
#ifdef HAVE_OPENCV_VIDEOIO
    if (!cap_.open(devId, cv::CAP_V4L2)) {
        if (!cap_.open(devId, cv::CAP_ANY)) {
            lastErr_ = "Failed to open camera device";
            return false;
        }
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap_.set(cv::CAP_PROP_FPS, fps);
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    isOpen_ = cap_.isOpened();
    return isOpen_;
#else
    return false;
#endif
}

bool GenericCapture::dsai_openSource(const std::string& path) {
#ifdef HAVE_OPENCV_VIDEOIO
    if (!cap_.open(path)) { lastErr_ = "Failed to open source: " + path; return false; }
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    isOpen_ = cap_.isOpened();
    return isOpen_;
#else
    return false;
#endif
}

bool GenericCapture::dsai_read(cv::Mat& frame) {
#ifdef HAVE_OPENCV_VIDEOIO
    if (!cap_.read(frame) || frame.empty()) {
        // Auto-loop for video files
        if (cap_.get(cv::CAP_PROP_FRAME_COUNT) > 0) {
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            return cap_.read(frame);
        }
        return false;
    }
    return true;
#else
    return false;
#endif
}

void GenericCapture::dsai_release() {
#ifdef HAVE_OPENCV_VIDEOIO
    if (cap_.isOpened()) cap_.release();
#endif
    isOpen_ = false;
}

bool GenericCapture::dsai_isOpened() const { return isOpen_; }

int GenericCapture::dsai_captureWidth() const { 
#ifdef HAVE_OPENCV_VIDEOIO
    return isOpen_ ? cap_.get(cv::CAP_PROP_FRAME_WIDTH) : 0; 
#else
    return 0;
#endif
}

int GenericCapture::dsai_captureHeight() const { 
#ifdef HAVE_OPENCV_VIDEOIO
    return isOpen_ ? cap_.get(cv::CAP_PROP_FRAME_HEIGHT) : 0; 
#else
    return 0;
#endif
}

double GenericCapture::dsai_captureFps() const { 
#ifdef HAVE_OPENCV_VIDEOIO
    return isOpen_ ? cap_.get(cv::CAP_PROP_FPS) : 0; 
#else
    return 0;
#endif
}

}
