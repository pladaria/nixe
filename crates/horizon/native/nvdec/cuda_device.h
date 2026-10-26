// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "dynlink_cuda.h"
#include <dlfcn.h>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
extern "C" {
#include <libavutil/hwcontext.h>
}

// Public driver ABI from FFmpeg/nv-codec-headers (MIT); no CUDA SDK dependency.
struct CudaDevice {
    void* library{};
    bool available{};
#define CUDA_SYMBOL(name) t##name* name{};
    CUDA_SYMBOL(cuInit) CUDA_SYMBOL(cuDeviceGetCount) CUDA_SYMBOL(cuDeviceGet)
    CUDA_SYMBOL(cuDeviceGetUuid) CUDA_SYMBOL(cuGetErrorString)
#undef CUDA_SYMBOL
    CudaDevice() {
        library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error("CUDA driver unavailable");
        try {
#define CUDA_SYMBOL(name) name = reinterpret_cast<t##name*>(dlsym(library, #name)); if (!name) throw std::runtime_error("CUDA symbol unavailable: " #name);
            CUDA_SYMBOL(cuInit) CUDA_SYMBOL(cuDeviceGetCount) CUDA_SYMBOL(cuDeviceGet)
            CUDA_SYMBOL(cuDeviceGetUuid) CUDA_SYMBOL(cuGetErrorString)
#undef CUDA_SYMBOL
            const auto initialized = cuInit(0);
            // CUDA_ERROR_NO_DEVICE is a capability result, not a failed decode.
            // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
            if (static_cast<int>(initialized) != 100) { check(initialized); available = true; }
        } catch (...) { dlclose(library); throw; }
    }
    ~CudaDevice() { dlclose(library); }
    void check(CUresult result) const {
        if (result == CUDA_SUCCESS) return;
        const char* message = nullptr; cuGetErrorString(result, &message);
        throw std::runtime_error(std::string("CUDA: ") + (message ? message : "unknown error"));
    }
    static CudaDevice& get() { static CudaDevice driver; return driver; }
    int find_device(const uint8_t* uuid) {
        if (!available) return -1;
        int count{}; check(cuDeviceGetCount(&count));
        for (int i=0;i<count;++i) {
            CUdevice device{}; CUuuid id{}; check(cuDeviceGet(&device,i)); check(cuDeviceGetUuid(&id,device));
            if (std::memcmp(id.bytes,uuid,16)==0) return i;
        }
        return -1;
    }
    int device(const uint8_t* uuid) {
        const auto ordinal = find_device(uuid);
        if (ordinal < 0) throw std::runtime_error("CUDA decoder cannot match Vulkan device UUID");
        return ordinal;
    }
};
