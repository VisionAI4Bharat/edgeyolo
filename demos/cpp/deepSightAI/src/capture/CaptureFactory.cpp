#include "CaptureFactory.h"
#ifdef WITH_RKNN
#include "rv1106_capture.h"
#else
#include "GenericCapture.h"
#endif

namespace deepSightAI {

std::unique_ptr<ICapture> CaptureFactory::dsai_create() {
#ifdef WITH_RKNN
    return std::make_unique<RV1106Capture>();
#else
    return std::make_unique<GenericCapture>();
#endif
}

}
