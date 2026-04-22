#include "GenericCapture.h"

namespace deepSightAI {

GenericCapture::GenericCapture() {}
GenericCapture::~GenericCapture() { dsai_release(); }

bool GenericCapture::dsai_openCamera(int devId, int width, int height, double fps) {
#ifdef HAVE_OPENCV_VIDEOIO
    if (!cap_.open(devId)) {
        lastErr_ = "Failed to open camera";
        return false;
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap_.set(cv::CAP_PROP_FPS, fps);
    isOpen_ = true;
    return true;
#else
    lastErr_ = "OpenCV videoio not available";
    return false;
#endif
}


bool GenericCapture::dsai_read(cv::Mat& frame) {
#ifdef HAVE_OPENCV_VIDEOIO
    return cap_.read(frame);
#else
    return false;
#endif
}

void GenericCapture::dsai_release() {
#ifdef HAVE_OPENCV_VIDEOIO
    cap_.release();
#endif
    isOpen_ = false;
}

bool GenericCapture::dsai_isOpened() const { return isOpen_; }

int GenericCapture::dsai_captureWidth() const {
#ifdef HAVE_OPENCV_VIDEOIO
    return cap_.get(cv::CAP_PROP_FRAME_WIDTH);
#else
    return 0;
#endif
}

int GenericCapture::dsai_captureHeight() const {
#ifdef HAVE_OPENCV_VIDEOIO
    return cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
#else
    return 0;
#endif
}

double GenericCapture::dsai_captureFps() const {
#ifdef HAVE_OPENCV_VIDEOIO
    return cap_.get(cv::CAP_PROP_FPS);
#else
    return 0;
#endif
}

}
