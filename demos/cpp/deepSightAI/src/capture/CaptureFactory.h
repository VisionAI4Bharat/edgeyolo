#pragma once
#include "ICapture.h"
#include <memory>

namespace deepSightAI {

class CaptureFactory {
public:
    static std::unique_ptr<ICapture> dsai_create();
};

}
