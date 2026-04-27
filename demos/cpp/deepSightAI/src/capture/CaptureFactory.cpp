#include "CaptureFactory.h"
#ifdef WITH_RKNN
#include "rv1106_capture.h"
#else
#include "GenericCapture.h"
#endif

#ifdef __linux__
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/ioctl.h>
#  include <linux/videodev2.h>
#endif

#include <algorithm>
#include <string>

namespace deepSightAI {

std::unique_ptr<ICapture> CaptureFactory::dsai_create() {
#ifdef WITH_RKNN
    return std::make_unique<RV1106Capture>();
#else
    return std::make_unique<GenericCapture>();
#endif
}

std::vector<CameraMode> ICapture::dsai_enumerateModes(int devId) {
    std::vector<CameraMode> modes;
#ifdef __linux__
    std::string path = "/dev/video" + std::to_string(devId);
    int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0) return modes;

    v4l2_fmtdesc fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    while (ioctl(fd, VIDIOC_ENUM_FMT, &fmt) == 0) {
        v4l2_frmsizeenum frmsize{};
        frmsize.pixel_format = fmt.pixelformat;
        while (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) == 0) {
            auto trySize = [&](uint32_t w, uint32_t h) {
                v4l2_frmivalenum ival{};
                ival.pixel_format = fmt.pixelformat;
                ival.width = w; ival.height = h;
                while (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &ival) == 0) {
                    if (ival.type == V4L2_FRMIVAL_TYPE_DISCRETE && ival.discrete.numerator > 0) {
                        CameraMode m;
                        m.width  = (int)w;
                        m.height = (int)h;
                        m.fps    = (int)(ival.discrete.denominator / ival.discrete.numerator);
                        // deduplicate
                        bool found = false;
                        for (auto& e : modes)
                            if (e.width == m.width && e.height == m.height && e.fps == m.fps)
                                { found = true; break; }
                        if (!found) modes.push_back(m);
                    }
                    ival.index++;
                }
                // Stepwise / continuous intervals — add common fps values
                if (ival.type == V4L2_FRMIVAL_TYPE_STEPWISE
                    || ival.type == V4L2_FRMIVAL_TYPE_CONTINUOUS) {
                    for (int fps : {15, 25, 30, 60}) {
                        CameraMode m; m.width = (int)w; m.height = (int)h; m.fps = fps;
                        bool found = false;
                        for (auto& e : modes)
                            if (e.width == m.width && e.height == m.height && e.fps == m.fps)
                                { found = true; break; }
                        if (!found) modes.push_back(m);
                    }
                }
            };
            if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                trySize(frmsize.discrete.width, frmsize.discrete.height);
            } else {
                // Stepwise / continuous — probe common resolutions within bounds
                auto& s = frmsize.stepwise;
                for (auto [w, h] : std::initializer_list<std::pair<int,int>>{
                        {320,240},{640,480},{720,480},{1280,720},{1920,1080}}) {
                    if ((uint32_t)w >= s.min_width  && (uint32_t)w <= s.max_width
                     && (uint32_t)h >= s.min_height && (uint32_t)h <= s.max_height
                     && w % s.step_width == 0 && h % s.step_height == 0)
                        trySize(w, h);
                }
            }
            frmsize.index++;
        }
        fmt.index++;
    }
    ::close(fd);

    std::sort(modes.begin(), modes.end(), [](const CameraMode& a, const CameraMode& b) {
        if (a.width != b.width)  return a.width  > b.width;
        if (a.height != b.height) return a.height > b.height;
        return a.fps > b.fps;
    });
#endif
    return modes;
}

}
