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

#include "rockchip_capture.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/imgproc.hpp>
#ifdef HAVE_OPENCV_VIDEOIO
#  include <opencv2/videoio.hpp>
#endif

#ifdef WITH_RKNN
// RK MPI — video input, system, memory-block
#include "rk_mpi_sys.h"
#include "rk_mpi_vi.h"
#include "rk_mpi_mb.h"
// RKAIQ — ISP initialisation wrapper from Rockchip sample_comm
#include "sample_comm.h"
#endif

namespace deepSightAI {

// ─── private implementation struct ────────────────────────────────────────────

struct RockchipCapture::Impl {
    bool        isOpen           = false;
    bool        isCam            = false;   // true = VI camera, false = RTSP/OpenCV
    int         width            = 0;
    int         height           = 0;
    double      fps              = 30.0;
    std::string lastErr;

#ifdef WITH_RKNN
    // VI pipeline state
    int         devId            = 0;
    int         chnId            = 0;
    bool        ispRunning       = false;
    bool        mpiInitialised   = false;

    // DMA-backed memory block used as the BGR staging buffer.
    // Writing into a DMA buffer avoids an extra memcpy before sending
    // the frame to the RKNN NPU (which reads from DMA-coherent memory).
    MB_POOL     mbPool           = MB_INVALID_POOLID;
    MB_BLK      mbBlk            = RK_NULL;
    unsigned char* mbData        = nullptr;
#endif

    // OpenCV VideoCapture — used for RTSP path and for the desktop/fallback
    // camera path when WITH_RKNN is not defined.
    // Not available when building with opencv-mobile (no videoio module).
#ifdef HAVE_OPENCV_VIDEOIO
    cv::VideoCapture cvCap;
#endif

    void setError(const std::string& msg) {
        lastErr = msg;
        fprintf(stderr, "[RockchipCapture] ERROR: %s\n", msg.c_str());
    }
};

// ─── Rockchip VI helpers (compiled only when WITH_RKNN is defined) ─────────────

#ifdef WITH_RKNN

static bool vi_open(RockchipCapture::Impl* p,
                    const RockchipCapture::CameraConfig& cfg)
{
    // 1. RKAIQ ISP — must start before RK MPI so the sensor is configured
    SAMPLE_COMM_ISP_Init(cfg.devId, RK_AIQ_WORKING_MODE_NORMAL,
                         RK_FALSE, cfg.iqDir.c_str());
    SAMPLE_COMM_ISP_Run(cfg.devId);
    p->ispRunning = true;

    // 2. RK MPI system
    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
        p->setError("RK_MPI_SYS_Init failed");
        return false;
    }
    p->mpiInitialised = true;

    // 3. VI device: configure → enable → bind pipe
    {
        VI_DEV_ATTR_S    devAttr;
        VI_DEV_BIND_PIPE_S bindPipe;
        memset(&devAttr,   0, sizeof(devAttr));
        memset(&bindPipe,  0, sizeof(bindPipe));

        RK_S32 ret = RK_MPI_VI_GetDevAttr(cfg.devId, &devAttr);
        if (ret == RK_ERR_VI_NOT_CONFIG) {
            ret = RK_MPI_VI_SetDevAttr(cfg.devId, &devAttr);
            if (ret != RK_SUCCESS) {
                p->setError("RK_MPI_VI_SetDevAttr failed: " + std::to_string(ret));
                return false;
            }
        }

        if (RK_MPI_VI_GetDevIsEnable(cfg.devId) != RK_SUCCESS) {
            ret = RK_MPI_VI_EnableDev(cfg.devId);
            if (ret != RK_SUCCESS) {
                p->setError("RK_MPI_VI_EnableDev failed: " + std::to_string(ret));
                return false;
            }
            bindPipe.u32Num    = 1;
            bindPipe.PipeId[0] = cfg.devId;
            ret = RK_MPI_VI_SetDevBindPipe(cfg.devId, &bindPipe);
            if (ret != RK_SUCCESS) {
                p->setError("RK_MPI_VI_SetDevBindPipe failed: " + std::to_string(ret));
                return false;
            }
        }
    }

    // 4. VI channel: YUV420SP, DMABUF, depth=2 (non-zero required for GetChnFrame)
    {
        VI_CHN_ATTR_S chnAttr;
        memset(&chnAttr, 0, sizeof(chnAttr));
        chnAttr.stIspOpt.u32BufCount  = 2;
        chnAttr.stIspOpt.enMemoryType = VI_V4L2_MEMORY_TYPE_DMABUF;
        chnAttr.stSize.u32Width       = static_cast<RK_U32>(cfg.width);
        chnAttr.stSize.u32Height      = static_cast<RK_U32>(cfg.height);
        chnAttr.enPixelFormat         = RK_FMT_YUV420SP;
        chnAttr.enCompressMode        = COMPRESS_MODE_NONE;
        chnAttr.u32Depth              = 2;

        RK_S32 ret = RK_MPI_VI_SetChnAttr(cfg.devId, cfg.chnId, &chnAttr);
        ret |= RK_MPI_VI_EnableChn(cfg.devId, cfg.chnId);
        if (ret) {
            p->setError("VI channel init failed: " + std::to_string(ret));
            return false;
        }
    }

    // 5. Allocate a DMA-backed MB pool for the BGR staging frame.
    //    One block of width*height*3 bytes is sufficient; the pool holds
    //    a single pre-allocated block (u32MBCnt=1).
    {
        const RK_U64 frameBytes = static_cast<RK_U64>(cfg.width * cfg.height * 3);
        MB_POOL_CONFIG_S poolCfg;
        memset(&poolCfg, 0, sizeof(poolCfg));
        poolCfg.u64MBSize   = frameBytes;
        poolCfg.u32MBCnt    = 1;
        poolCfg.enAllocType = MB_ALLOC_TYPE_DMA;

        p->mbPool = RK_MPI_MB_CreatePool(&poolCfg);
        if (p->mbPool == MB_INVALID_POOLID) {
            p->setError("RK_MPI_MB_CreatePool failed");
            return false;
        }

        p->mbBlk = RK_MPI_MB_GetMB(p->mbPool, frameBytes, RK_TRUE);
        if (!p->mbBlk) {
            p->setError("RK_MPI_MB_GetMB failed");
            return false;
        }
        p->mbData = static_cast<unsigned char*>(
            RK_MPI_MB_Handle2VirAddr(p->mbBlk));
    }

    p->devId  = cfg.devId;
    p->chnId  = cfg.chnId;
    p->width  = cfg.width;
    p->height = cfg.height;
    p->fps    = cfg.fps;
    return true;
}

static void vi_close(RockchipCapture::Impl* p)
{
    if (p->mbBlk) {
        RK_MPI_MB_ReleaseMB(p->mbBlk);
        p->mbBlk  = RK_NULL;
        p->mbData = nullptr;
    }
    if (p->mbPool != MB_INVALID_POOLID) {
        RK_MPI_MB_DestroyPool(p->mbPool);
        p->mbPool = MB_INVALID_POOLID;
    }

    RK_MPI_VI_DisableChn(p->devId, p->chnId);
    RK_MPI_VI_DisableDev(p->devId);

    if (p->ispRunning) {
        SAMPLE_COMM_ISP_Stop(p->devId);
        p->ispRunning = false;
    }
    if (p->mpiInitialised) {
        RK_MPI_SYS_Exit();
        p->mpiInitialised = false;
    }
}

// Capture one YUV420SP frame from VI and convert to BGR.
// Blocks until the ISP/VI pipeline delivers a new frame (timeout = -1 = infinite).
static bool vi_read(RockchipCapture::Impl* p, cv::Mat& bgrOut)
{
    VIDEO_FRAME_INFO_S viFrame;
    const RK_S32 ret = RK_MPI_VI_GetChnFrame(p->devId, p->chnId, &viFrame, -1);
    if (ret != RK_SUCCESS) {
        p->setError("RK_MPI_VI_GetChnFrame failed: " + std::to_string(ret));
        return false;
    }

    // Map the YUV420SP (NV12) DMA buffer into virtual address space
    void* const yuvPtr = RK_MPI_MB_Handle2VirAddr(viFrame.stVFrame.pMbBlk);
    const int H = p->height;
    const int W = p->width;

    // In-place colour conversion: YUV → BGR written into the pre-allocated DMA block.
    // cv::cvtColor does not allocate a new Mat when the destination is already the
    // right size and type — so p->mbData acts as a zero-copy BGR staging area.
    cv::Mat yuv420sp(H + H / 2, W, CV_8UC1, yuvPtr);
    cv::Mat bgrStage(H, W, CV_8UC3, p->mbData);
    cv::cvtColor(yuv420sp, bgrStage, cv::COLOR_YUV420sp2BGR);

    // Deep-copy to the caller's Mat so we can immediately release the VI frame
    // buffer back to the hardware — the DMA block must not be held across frames.
    bgrOut = bgrStage.clone();

    if (RK_MPI_VI_ReleaseChnFrame(p->devId, p->chnId, &viFrame) != RK_SUCCESS)
        fprintf(stderr, "[RockchipCapture] WARN: RK_MPI_VI_ReleaseChnFrame failed\n");

    return true;
}

#endif  // WITH_RKNN

// ─── RockchipCapture public API ────────────────────────────────────────────────

RockchipCapture::RockchipCapture()
    : pImpl_(std::make_unique<Impl>()) {}

RockchipCapture::~RockchipCapture() { dsai_release(); }

RockchipCapture::RockchipCapture(RockchipCapture&& o) noexcept
    : pImpl_(std::move(o.pImpl_)) {}

RockchipCapture& RockchipCapture::operator=(RockchipCapture&& o) noexcept {
    if (this != &o) { dsai_release(); pImpl_ = std::move(o.pImpl_); }
    return *this;
}

bool RockchipCapture::dsai_openCamera(const CameraConfig& cfg) {
    dsai_release();
    pImpl_ = std::make_unique<Impl>();
    pImpl_->isCam = true;

#ifdef WITH_RKNN
    if (!vi_open(pImpl_.get(), cfg)) return false;
    pImpl_->isOpen = true;
    return true;
#else
#ifdef HAVE_OPENCV_VIDEOIO
    // Desktop / development fallback: V4L2 via OpenCV
    pImpl_->setError("Built without WITH_RKNN — using cv::VideoCapture fallback");
    pImpl_->cvCap.open(cfg.devId, cv::CAP_V4L2);
    if (!pImpl_->cvCap.dsai_isOpened()) pImpl_->cvCap.open(cfg.devId);
    if (!pImpl_->cvCap.dsai_isOpened()) {
        pImpl_->setError("cv::VideoCapture could not open device " +
                          std::to_string(cfg.devId));
        return false;
    }
    pImpl_->cvCap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
    pImpl_->cvCap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
    pImpl_->cvCap.set(cv::CAP_PROP_FPS,          cfg.fps);
    pImpl_->width  = static_cast<int>(pImpl_->cvCap.get(cv::CAP_PROP_FRAME_WIDTH));
    pImpl_->height = static_cast<int>(pImpl_->cvCap.get(cv::CAP_PROP_FRAME_HEIGHT));
    pImpl_->fps    = pImpl_->cvCap.get(cv::CAP_PROP_FPS);
    pImpl_->isOpen = true;
    return true;
#else
    pImpl_->setError("openCamera requires WITH_RKNN or opencv videoio module");
    return false;
#endif
#endif
}

bool RockchipCapture::dsai_openRtsp(const RtspConfig& cfg) {
    dsai_release();
    pImpl_ = std::make_unique<Impl>();
    pImpl_->isCam = false;

#ifdef HAVE_OPENCV_VIDEOIO
    pImpl_->cvCap.open(cfg.url, cv::CAP_FFMPEG);
    if (!pImpl_->cvCap.dsai_isOpened()) {
        pImpl_->setError("Failed to open RTSP stream: " + cfg.url);
        return false;
    }
    pImpl_->width  = static_cast<int>(pImpl_->cvCap.get(cv::CAP_PROP_FRAME_WIDTH));
    pImpl_->height = static_cast<int>(pImpl_->cvCap.get(cv::CAP_PROP_FRAME_HEIGHT));
    pImpl_->fps    = pImpl_->cvCap.get(cv::CAP_PROP_FPS);
    pImpl_->isOpen = true;
    return true;
#else
    (void)cfg;
    pImpl_->setError("RTSP requires opencv videoio module (not available in this build)");
    return false;
#endif
}

bool RockchipCapture::dsai_isOpened() const noexcept {
    return pImpl_ && pImpl_->isOpen;
}

bool RockchipCapture::dsai_read(cv::Mat& bgrFrame) {
    if (!pImpl_ || !pImpl_->isOpen) return false;

#ifdef WITH_RKNN
    if (pImpl_->isCam) return vi_read(pImpl_.get(), bgrFrame);
#endif

#ifdef HAVE_OPENCV_VIDEOIO
    return pImpl_->cvCap.dsai_read(bgrFrame);
#else
    return false;
#endif
}

void RockchipCapture::dsai_release() {
    if (!pImpl_ || !pImpl_->isOpen) return;

#ifdef WITH_RKNN
    if (pImpl_->isCam) vi_close(pImpl_.get());
#endif

#ifdef HAVE_OPENCV_VIDEOIO
    if (pImpl_->cvCap.dsai_isOpened()) pImpl_->cvCap.dsai_release();
#endif
    pImpl_->isOpen = false;
}

int    RockchipCapture::dsai_captureWidth()  const noexcept { return pImpl_ ? pImpl_->width  : 0; }
int    RockchipCapture::dsai_captureHeight() const noexcept { return pImpl_ ? pImpl_->height : 0; }
double RockchipCapture::dsai_captureFps()    const noexcept { return pImpl_ ? pImpl_->fps    : 0.0; }

std::string RockchipCapture::dsai_lastError() const {
    return pImpl_ ? pImpl_->lastErr : std::string{};
}

}  // namespace deepSightAI
