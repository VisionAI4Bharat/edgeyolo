/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 */

#include "rv1106_capture.h"
#include "../app_config.h"
#include "../debug_log.h"

#include <cstdio>
#include <cstring>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <unistd.h>
#include <stdexcept>

#include "rk_mpi_sys.h"
#include "rk_mpi_vi.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_venc.h"
#include "rk_mpi_rgn.h"
#include "sample_comm.h"
#include "rtsp_demo.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace deepSightAI {

static const char* TAG = "RV1106CAP";

#define CHECK_RET(ret, msg) \
    if ((ret) != RK_SUCCESS) { \
        char errBuf[512]; \
        snprintf(errBuf, sizeof(errBuf), "[%s] %s failed with %#x", TAG, (msg), (ret)); \
        throw std::runtime_error(errBuf); \
    }

static void set_argb8888_corner(uint32_t *buf, int type, uint32_t color) {
    for (int i = 0; i < 20; i++) {
        for(int j = 0; j < 2; j++) {
            if (type == 0 || type == 1) buf[i + j*20] = color;
            if (type == 2 || type == 3) buf[i + (20-2+j)*20] = color;
        }
    }
    for (int j = 0; j < 20; j++) {
        for(int i = 0; i < 2; i++) {
            if (type == 0 || type == 3) buf[i + j*20] = color;
            if (type == 1 || type == 2) buf[(20-2+i) + j*20] = color;
        }
    }
}

struct RV1106Capture::Impl {
    std::atomic<bool> isOpen{false};
    int width = 1920; int height = 1080;
    int modelW = 0; int modelH = 0;
    double fps = 30.0;
    std::string lastErr;

    int rtspPort = 8554;
    std::string rtspUrl = "/live/0";
    std::string iqDir = "/etc/iqfiles";

    rtsp_demo_handle rtsplive = nullptr;
    rtsp_session_handle rtsp_session = nullptr;
    
    std::thread vencThread;
    std::thread viInferenceThread;
    std::thread rgnThread;
    std::atomic<bool> running{false};

    mutable std::mutex frameMutex;
    cv::Mat currentFrame;

    std::mutex osdMutex;
    std::vector<inference::Detection> detections;
    bool osdDirty = false;

    MPP_CHN_S stViChn0, stViChn1, stVencChn0;

    void setError(const std::string& msg) {
        lastErr = msg;
        fprintf(stderr, "[%s] FATAL: %s\n", TAG, msg.c_str());
    }

    void updateOSD() {
        MPP_CHN_S stMppChn; stMppChn.enModId = RK_ID_VENC; stMppChn.s32DevId = 0; stMppChn.s32ChnId = 0;
        for(int i = 0; i < 16; i++) { RK_MPI_RGN_DetachFromChn(i, &stMppChn); RK_MPI_RGN_Destroy(i); }
        std::lock_guard<std::mutex> lock(osdMutex);
        int count = std::min((int)detections.size(), 4);
        for (int i = 0; i < count; i++) {
            auto& d = detections[i];
            int cx[4] = {(int)d.rect.x, (int)(d.rect.x+d.rect.width), (int)(d.rect.x+d.rect.width), (int)d.rect.x};
            int cy[4] = {(int)d.rect.y, (int)d.rect.y, (int)(d.rect.y+d.rect.height), (int)(d.rect.y+d.rect.height)};
            for (int t = 0; t < 4; t++) {
                RGN_HANDLE h = i * 4 + t;
                RGN_ATTR_S attr; attr.enType = OVERLAY_RGN; attr.unAttr.stOverlay.enPixelFmt = (PIXEL_FORMAT_E)RK_FMT_ARGB8888;
                attr.unAttr.stOverlay.stSize.u32Width = 20; attr.unAttr.stOverlay.stSize.u32Height = 20;
                if (RK_MPI_RGN_Create(h, &attr) != RK_SUCCESS) continue;
                RGN_CHN_ATTR_S cattr; memset(&cattr, 0, sizeof(cattr)); cattr.bShow = RK_TRUE; cattr.enType = OVERLAY_RGN;
                cattr.unChnAttr.stOverlayChn.stPoint.s32X = std::max(0, cx[t]-10) & ~1;
                cattr.unChnAttr.stOverlayChn.stPoint.s32Y = std::max(0, cy[t]-10) & ~1;
                RK_MPI_RGN_AttachToChn(h, &stMppChn, &cattr);
                BITMAP_S bm; bm.enPixelFormat = (PIXEL_FORMAT_E)RK_FMT_ARGB8888; bm.u32Width = 20; bm.u32Height = 20;
                bm.pData = malloc(20*20*4); set_argb8888_corner((uint32_t*)bm.pData, t, 0x00FF00FF);
                RK_MPI_RGN_SetBitMap(h, &bm); free(bm.pData);
            }
        }
    }
};

RV1106Capture::RV1106Capture() : pImpl_(std::make_unique<Impl>()) {}
RV1106Capture::~RV1106Capture() { dsai_release(); }

bool RV1106Capture::dsai_openCamera(int devId, int width, int height, double fps) {
    try {
        pImpl_->width = width; pImpl_->height = height; pImpl_->fps = fps;
        CHECK_RET(SAMPLE_COMM_ISP_Init(0, RK_AIQ_WORKING_MODE_NORMAL, RK_FALSE, pImpl_->iqDir.c_str()), "ISP Init");
        CHECK_RET(SAMPLE_COMM_ISP_Run(0), "ISP Run");
        CHECK_RET(RK_MPI_SYS_Init(), "SYS Init");
        pImpl_->rtsplive = create_rtsp_demo(pImpl_->rtspPort);
        pImpl_->rtsp_session = rtsp_new_session(pImpl_->rtsplive, pImpl_->rtspUrl.c_str());
        rtsp_set_video(pImpl_->rtsp_session, RTSP_CODEC_ID_VIDEO_H265, NULL, 0);
        VI_DEV_ATTR_S da; memset(&da, 0, sizeof(da));
        CHECK_RET(RK_MPI_VI_SetDevAttr(0, &da), "VI SetDev");
        CHECK_RET(RK_MPI_VI_EnableDev(0), "VI EnableDev");
        VI_CHN_ATTR_S ca; memset(&ca, 0, sizeof(ca)); ca.stIspOpt.u32BufCount = 3; ca.stIspOpt.enMemoryType = VI_V4L2_MEMORY_TYPE_DMABUF;
        ca.enPixelFormat = RK_FMT_YUV420SP; ca.stSize.u32Width = width; ca.stSize.u32Height = height;
        CHECK_RET(RK_MPI_VI_SetChnAttr(0, 0, &ca), "VI C0 Attr");
        CHECK_RET(RK_MPI_VI_EnableChn(0, 0), "VI C0 Enable");
        ca.stSize.u32Width = pImpl_->modelW; ca.stSize.u32Height = pImpl_->modelH; ca.u32Depth = 2;
        CHECK_RET(RK_MPI_VI_SetChnAttr(0, 1, &ca), "VI C1 Attr");
        CHECK_RET(RK_MPI_VI_EnableChn(0, 1), "VI C1 Enable");
        VENC_CHN_ATTR_S va; memset(&va, 0, sizeof(va)); va.stVencAttr.enType = RK_VIDEO_ID_HEVC;
        va.stVencAttr.u32MaxPicWidth = width; va.stVencAttr.u32MaxPicHeight = height;
        va.stVencAttr.u32PicWidth = width; va.stVencAttr.u32PicHeight = height;
        va.stVencAttr.u32BufSize = width*height*3/2; va.stRcAttr.enRcMode = VENC_RC_MODE_H265CBR;
        va.stRcAttr.stH265Cbr.u32BitRate = 4096; va.stRcAttr.stH265Cbr.u32Gop = 60;
        CHECK_RET(RK_MPI_VENC_CreateChn(0, &va), "VENC Create");
        pImpl_->stViChn0.enModId=RK_ID_VI; pImpl_->stViChn0.s32ChnId=0; pImpl_->stVencChn0.enModId=RK_ID_VENC; pImpl_->stVencChn0.s32ChnId=0;
        CHECK_RET(RK_MPI_SYS_Bind(&pImpl_->stViChn0, &pImpl_->stVencChn0), "Bind");
        VENC_RECV_PIC_PARAM_S sp; memset(&sp, 0, sizeof(sp));
        CHECK_RET(RK_MPI_VENC_StartRecvFrame(0, &sp), "VENC Start");
        pImpl_->running = true;
        pImpl_->vencThread = std::thread([this](){
            VENC_STREAM_S s; s.pstPack = (VENC_PACK_S*)malloc(sizeof(VENC_PACK_S));
            while(pImpl_->running){ if(RK_MPI_VENC_GetStream(0, &s, 100)==RK_SUCCESS){
                void* p = RK_MPI_MB_Handle2VirAddr(s.pstPack->pMbBlk);
                rtsp_tx_video(pImpl_->rtsp_session, (uint8_t*)p, s.pstPack->u32Len, s.pstPack->u64PTS);
                rtsp_do_event(pImpl_->rtsplive); RK_MPI_VENC_ReleaseStream(0, &s);
            }} free(s.pstPack);
        });
        pImpl_->viInferenceThread = std::thread([this](){
            VIDEO_FRAME_INFO_S f;
            while(pImpl_->running){ if(RK_MPI_VI_GetChnFrame(0,1,&f,100)==RK_SUCCESS){
                void* p = RK_MPI_MB_Handle2VirAddr(f.stVFrame.pMbBlk);
                cv::Mat yuv(pImpl_->modelH+pImpl_->modelH/2, pImpl_->modelW, CV_8UC1, p);
                { std::lock_guard<std::mutex> l(pImpl_->frameMutex); cv::cvtColor(yuv, pImpl_->currentFrame, cv::COLOR_YUV420sp2BGR); }
                RK_MPI_VI_ReleaseChnFrame(0,1,&f);
            }}
        });
        pImpl_->rgnThread = std::thread([this](){
            while(pImpl_->running){ if(pImpl_->osdDirty){ pImpl_->updateOSD(); pImpl_->osdDirty=false; } usleep(30000); }
        });
        pImpl_->isOpen = true; return true;
    } catch(const std::exception& e) { pImpl_->setError(e.what()); return false; }
}

bool RV1106Capture::dsai_openSource(const std::string&) { return false; }

bool RV1106Capture::dsai_read(cv::Mat& frame) {
    std::lock_guard<std::mutex> l(pImpl_->frameMutex);
    if(pImpl_->currentFrame.empty()) return false;
    pImpl_->currentFrame.copyTo(frame); return true;
}

void RV1106Capture::dsai_release() {
    if(!pImpl_->isOpen) return; pImpl_->isOpen=false; pImpl_->running=false;
    if(pImpl_->vencThread.joinable()) pImpl_->vencThread.join();
    if(pImpl_->viInferenceThread.joinable()) pImpl_->viInferenceThread.join();
    if(pImpl_->rgnThread.joinable()) pImpl_->rgnThread.join();
    RK_MPI_SYS_UnBind(&pImpl_->stViChn0, &pImpl_->stVencChn0);
    RK_MPI_VENC_StopRecvFrame(0); RK_MPI_VENC_DestroyChn(0);
    RK_MPI_VI_DisableChn(0,0); RK_MPI_VI_DisableChn(0,1); RK_MPI_VI_DisableDev(0);
    RK_MPI_SYS_Exit(); SAMPLE_COMM_ISP_Stop(0);
}

void RV1106Capture::dsai_setOSD(const std::vector<inference::Detection>& detections) {
    std::lock_guard<std::mutex> l(pImpl_->osdMutex);
    pImpl_->detections = detections; pImpl_->osdDirty = true;
}

void RV1106Capture::dsai_setAppConfig(const AppConfig& cfg) {
    pImpl_->rtspPort = cfg.rtspPort;
    pImpl_->rtspUrl  = cfg.rtspUrl;
    pImpl_->iqDir    = cfg.iqDir;
}

void RV1106Capture::dsai_setModelInputSize(int w, int h) {
    pImpl_->modelW = w;
    pImpl_->modelH = h;
}

bool RV1106Capture::dsai_isOpened() const { return pImpl_->isOpen; }
int RV1106Capture::dsai_captureWidth() const { return pImpl_->width; }
int RV1106Capture::dsai_captureHeight() const { return pImpl_->height; }
double RV1106Capture::dsai_captureFps() const { return pImpl_->fps; }
std::string RV1106Capture::dsai_lastError() const { return pImpl_->lastErr; }

}
