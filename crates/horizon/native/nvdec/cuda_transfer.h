// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "dynlink_cuda.h"
#include <dlfcn.h>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unistd.h>
extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

// Public driver ABI from FFmpeg/nv-codec-headers (MIT); no CUDA SDK dependency.
struct CudaTransfer {
    void* library{};
#define CUDA_SYMBOL(name) t##name* name{};
    CUDA_SYMBOL(cuImportExternalSemaphore) CUDA_SYMBOL(cuSignalExternalSemaphoresAsync) CUDA_SYMBOL(cuDestroyExternalSemaphore) CUDA_SYMBOL(cuCtxGetDevice)
    CUDA_SYMBOL(cuInit) CUDA_SYMBOL(cuDeviceGetCount) CUDA_SYMBOL(cuDeviceGet)
    CUDA_SYMBOL(cuDeviceGetUuid) CUDA_SYMBOL(cuCtxPushCurrent_v2) CUDA_SYMBOL(cuCtxPopCurrent_v2)
    CUDA_SYMBOL(cuImportExternalMemory) CUDA_SYMBOL(cuExternalMemoryGetMappedBuffer)
    CUDA_SYMBOL(cuMemcpy2DAsync_v2) CUDA_SYMBOL(cuStreamSynchronize)
    CUDA_SYMBOL(cuMemsetD8Async) CUDA_SYMBOL(cuMemFree_v2) CUDA_SYMBOL(cuDestroyExternalMemory) CUDA_SYMBOL(cuGetErrorString)
#undef CUDA_SYMBOL
    CudaTransfer() {
        library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error("CUDA driver unavailable");
        try {
#define CUDA_SYMBOL(name) name = reinterpret_cast<t##name*>(dlsym(library, #name)); if (!name) throw std::runtime_error("CUDA symbol unavailable: " #name);
            CUDA_SYMBOL(cuImportExternalSemaphore) CUDA_SYMBOL(cuSignalExternalSemaphoresAsync) CUDA_SYMBOL(cuDestroyExternalSemaphore) CUDA_SYMBOL(cuCtxGetDevice)
            CUDA_SYMBOL(cuInit) CUDA_SYMBOL(cuDeviceGetCount) CUDA_SYMBOL(cuDeviceGet)
            CUDA_SYMBOL(cuDeviceGetUuid) CUDA_SYMBOL(cuCtxPushCurrent_v2) CUDA_SYMBOL(cuCtxPopCurrent_v2)
            CUDA_SYMBOL(cuImportExternalMemory) CUDA_SYMBOL(cuExternalMemoryGetMappedBuffer)
            CUDA_SYMBOL(cuMemcpy2DAsync_v2) CUDA_SYMBOL(cuStreamSynchronize)
            CUDA_SYMBOL(cuMemsetD8Async) CUDA_SYMBOL(cuMemFree_v2) CUDA_SYMBOL(cuDestroyExternalMemory) CUDA_SYMBOL(cuGetErrorString)
#undef CUDA_SYMBOL
            check(cuInit(0));
        } catch (...) { dlclose(library); throw; }
    }
    ~CudaTransfer() { dlclose(library); }
    void check(CUresult result) const {
        if (result == CUDA_SUCCESS) return;
        const char* message = nullptr; cuGetErrorString(result, &message);
        throw std::runtime_error(std::string("CUDA: ") + (message ? message : "unknown error"));
    }
    static CudaTransfer& get() { static CudaTransfer driver; return driver; }
    int device(const uint8_t* uuid) {
        int count{}; check(cuDeviceGetCount(&count));
        for (int i=0;i<count;++i) {
            CUdevice device{}; CUuuid id{}; check(cuDeviceGet(&device,i)); check(cuDeviceGetUuid(&id,device));
            if (std::memcmp(id.bytes,uuid,16)==0) return i;
        }
        throw std::runtime_error("CUDA decoder cannot match Vulkan device UUID");
    }
};
