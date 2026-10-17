// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "cuda_transfer.h"
#include <cstdio>

// Persistent imports belong to a renderer allocation, not to an individual
// frame. CUDA signals Vulkan directly; no per-frame host synchronization.
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html
struct VideoTransfer {
    CudaTransfer& cuda{CudaTransfer::get()};
    AVBufferRef* device{};
    CUcontext context{};
    CUstream stream{};
    CUexternalMemory memory{};
    CUexternalSemaphore semaphore{};
    CUdeviceptr pointer{};
    CUevent event{};
    uint64_t size{}, pitch{}, chroma{};
    bool pending{}, recorded{}, initialize{};

    struct Current {
        CudaTransfer& cuda;
        Current(CudaTransfer& cuda, CUcontext context) : cuda(cuda) { cuda.check(cuda.cuCtxPushCurrent_v2(context)); }
        ~Current() { CUcontext previous{}; cuda.cuCtxPopCurrent_v2(&previous); }
    };
    static const AVCUDADeviceContext* frame_context(const AVFrame* frame) {
        if (!frame || frame->format != AV_PIX_FMT_CUDA || !frame->hw_frames_ctx)
            throw std::runtime_error("invalid CUDA video frame");
        const auto* frames = reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
        return reinterpret_cast<const AVCUDADeviceContext*>(frames->device_ctx->hwctx);
    }
    void open(const AVFrame* frame, int fd, int semaphore_fd, uint64_t allocation_size,
              const uint8_t* uuid, uint64_t row_pitch, uint64_t chroma_offset, bool clear) {
        const auto* source = frame_context(frame);
        const auto* frames = reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
        device = av_buffer_ref(frames->device_ref);
        if (!device) throw std::bad_alloc();
        context = source->cuda_ctx; stream = source->stream;
        size = allocation_size; pitch = row_pitch; chroma = chroma_offset; initialize = clear;
        Current current(cuda, context);
        CUdevice actual{}; CUuuid id{};
        cuda.check(cuda.cuCtxGetDevice(&actual)); cuda.check(cuda.cuDeviceGetUuid(&id, actual));
        if (std::memcmp(id.bytes, uuid, 16) != 0) throw std::runtime_error("decoded frame belongs to another GPU");
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC description{};
        description.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        description.handle.fd = dup(fd); description.size = size; description.flags = 1;
        if (description.handle.fd < 0) throw std::runtime_error("cannot duplicate Vulkan memory FD");
        const auto imported = cuda.cuImportExternalMemory(&memory, &description);
        if (imported != CUDA_SUCCESS) close(description.handle.fd);
        cuda.check(imported);
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer{}; buffer.size = size;
        cuda.check(cuda.cuExternalMemoryGetMappedBuffer(&pointer, memory, &buffer));
        CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC sem{};
        sem.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD; sem.handle.fd = dup(semaphore_fd);
        if (sem.handle.fd < 0) throw std::runtime_error("cannot duplicate Vulkan semaphore FD");
        const auto sem_imported = cuda.cuImportExternalSemaphore(&semaphore, &sem);
        if (sem_imported != CUDA_SUCCESS) close(sem.handle.fd);
        cuda.check(sem_imported);
        cuda.check(cuda.cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
    }
    void copy(const AVFrame* frame) {
        if (frame_context(frame)->cuda_ctx != context)
            throw std::runtime_error("video transfer belongs to another decoder context");
        if (!frame->data[0] || !frame->data[1] || frame->width <= 0 || frame->height <= 0
            || frame->linesize[0] < frame->width || frame->linesize[1] < (frame->width+1)/2*2
            || pitch < static_cast<uint64_t>((frame->width+1)/2*2)
            || chroma < pitch*frame->height || size < chroma+pitch*((frame->height+1)/2))
            throw std::runtime_error("invalid NV12 plane dimensions or external allocation extent");
        Current current(cuda, context);
        pending = true; recorded = false;
        if (initialize) cuda.check(cuda.cuMemsetD8Async(pointer, 0, size, stream));
        for (int plane=0; plane<2; ++plane) {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.srcDevice = reinterpret_cast<CUdeviceptr>(frame->data[plane]); copy.srcPitch = frame->linesize[plane];
            copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.dstDevice = pointer + (plane ? chroma : 0); copy.dstPitch = pitch;
            copy.WidthInBytes = plane ? (frame->width+1)/2*2 : frame->width;
            copy.Height = plane ? (frame->height+1)/2 : frame->height;
            cuda.check(cuda.cuMemcpy2DAsync_v2(&copy, stream));
        }
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal{};
        cuda.check(cuda.cuSignalExternalSemaphoresAsync(&semaphore, &signal, 1, stream));
        cuda.check(cuda.cuEventRecord(event, stream));
        recorded = true; initialize = false;
    }
    void wait() {
        if (!pending) return;
        Current current(cuda, context);
        // The event covers only this transfer. A failed enqueue can precede its
        // record; that exceptional path must drain the partially queued stream.
        cuda.check(recorded ? cuda.cuEventSynchronize(event) : cuda.cuStreamSynchronize(stream));
        pending = false;
    }
    ~VideoTransfer() {
        if (context) {
            try {
                wait();
                Current current(cuda, context);
                if (event) cuda.cuEventDestroy_v2(event);
                if (semaphore) cuda.cuDestroyExternalSemaphore(semaphore);
                if (pointer) cuda.cuMemFree_v2(pointer);
                if (memory) cuda.cuDestroyExternalMemory(memory);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "cannot retire external video transfer: %s\n", error.what());
                std::terminate();
            }
        }
        av_buffer_unref(&device);
    }
};
