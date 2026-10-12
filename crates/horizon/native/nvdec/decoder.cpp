// SPDX-License-Identifier: GPL-3.0-or-later
// FFmpeg owns the codec state and reference frames; the bridge hides its ABI from Rust.
// https://ffmpeg.org/doxygen/trunk/group__lavc__encdec.html
#include "vp9.h"
#include <cstring>
#include <string>
#include <memory>
#ifdef __linux__
#include "cuda_transfer.h"
#endif
static thread_local std::string native_error;
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/pixdesc.h>
}

struct Decoder {
    Tegra::Decoders::VP9 headers;
    AVCodecContext* context{};
    AVFrame* frame{};
    std::string error;
    ~Decoder() { av_frame_free(&frame); avcodec_free_context(&context); }
    void check(int code) {
        if (code < 0) {
            char text[AV_ERROR_MAX_STRING_SIZE]; av_strerror(code, text, sizeof(text));
            throw std::runtime_error(text);
        }
    }
};
extern "C" {
const char* nixe_video_error() noexcept { return native_error.c_str(); }
Decoder* nixe_nvdec_create(const uint8_t* uuid) noexcept {
    try {
        auto decoder=std::make_unique<Decoder>();
        const AVCodec* codec=avcodec_find_decoder(AV_CODEC_ID_VP9);
        if (!codec) throw std::runtime_error("FFmpeg VP9 decoder unavailable");
        decoder->context=avcodec_alloc_context3(codec);
        decoder->frame=av_frame_alloc();
        if (!decoder->context || !decoder->frame) throw std::bad_alloc();
#ifdef __linux__
        if (uuid) {
            auto& cuda = CudaTransfer::get();
            const auto ordinal = std::to_string(cuda.device(uuid));
            decoder->check(av_hwdevice_ctx_create(&decoder->context->hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, ordinal.c_str(), nullptr, 0));
            decoder->context->get_format = [](AVCodecContext* context, const AVPixelFormat* formats) {
                // FFmpeg's mapped NVDEC chroma offset uses the visible luma height;
                // NV12 decoder surfaces round it up to an even number of rows.
                // Keep the exact software decoder for odd heights, rather than
                // interpreting padding as chroma or rewriting FFmpeg-owned planes.
                // https://ffmpeg.org/doxygen/4.4/nvdec_8c_source.html
                if (context->coded_height & 1) {
                    for (; *formats != AV_PIX_FMT_NONE; ++formats)
                        if (*formats == AV_PIX_FMT_YUV420P) return *formats;
                    return AV_PIX_FMT_NONE;
                }
                for (; *formats != AV_PIX_FMT_NONE; ++formats) if (*formats == AV_PIX_FMT_CUDA) return *formats;
                return AV_PIX_FMT_NONE;
            };
        }
#else
        (void)uuid;
#endif
        // Slice threading preserves packet/surface order without frame-thread latency.
        decoder->context->thread_type=FF_THREAD_SLICE;
        decoder->context->thread_count=0;
        decoder->context->err_recognition=AV_EF_EXPLODE;
        decoder->check(avcodec_open2(decoder->context, codec, nullptr));
        return decoder.release();
    } catch (const std::exception& e) { native_error=e.what(); return nullptr; }
}
void nixe_nvdec_destroy(Decoder* decoder) noexcept { delete decoder; }
const char* nixe_nvdec_error(const Decoder* decoder) noexcept { return decoder->error.c_str(); }
int nixe_nvdec_send(Decoder* decoder, const uint8_t* picture, const uint8_t* entropy,
                   const uint8_t* tiles, size_t length, const uint64_t* offsets,
                   uint64_t origin) noexcept {
    try {
        Tegra::Decoders::PictureInfo info;
        Tegra::Decoders::EntropyProbs probs;
        std::memcpy(&info,picture,sizeof(info)); std::memcpy(&probs,entropy,sizeof(probs));
        std::array<uint64_t,4> refs;
        std::copy_n(offsets,4,refs.begin());
        AVPacket* packet=av_packet_alloc();
        if (!packet) throw std::bad_alloc();
        std::unique_ptr<AVPacket, void(*)(AVPacket*)> owner(packet, [](AVPacket* p){av_packet_free(&p);});
        decoder->check(decoder->headers.ComposeFrame(info,probs,{tiles,length},refs,origin,packet));
        packet->pts=packet->dts=static_cast<int64_t>(origin);
        decoder->check(avcodec_send_packet(decoder->context,packet));
        return 1;
    } catch (const std::exception& error) { decoder->error=error.what(); return -1; }
    catch (...) { decoder->error="unknown native decoder failure"; return -1; }
}
// Software planes are borrowed until the next receive; CUDA output is an owned AVFrame reference.
// No AVFrame layout crosses FFI.
int nixe_nvdec_receive(Decoder* decoder, uint64_t* origin, uint32_t* width, uint32_t* height,
                      const uint8_t** planes, int32_t* strides, AVFrame** device_frame) noexcept {
    try {
        av_frame_unref(decoder->frame);
        const int result=avcodec_receive_frame(decoder->context,decoder->frame);
        if (result==AVERROR(EAGAIN) || result==AVERROR_EOF) return 0;
        decoder->check(result);
        const AVFrame* frame=decoder->frame;
        const bool hardware = frame->format==AV_PIX_FMT_CUDA;
        if (!hardware && frame->format!=AV_PIX_FMT_YUV420P)
            throw std::runtime_error("NVDEC VP9 output requires an unsupported pixel format");
        if (frame->pts<=0 || frame->width<=0 || frame->height<=0)
            throw std::runtime_error("NVDEC decoded frame has invalid dimensions or origin");
        *origin=static_cast<uint64_t>(frame->pts); *width=frame->width; *height=frame->height;
        if (hardware) {
            const auto* frames=reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
            if (frames->sw_format!=AV_PIX_FMT_NV12) throw std::runtime_error("NVDEC CUDA output is not NV12");
            *device_frame=av_frame_clone(frame);
            if (!*device_frame) throw std::bad_alloc();
        }
        for (int i=0;i<3;++i) { planes[i]=frame->data[i]; strides[i]=frame->linesize[i]; }
        return 1;
    } catch (const std::exception& error) { decoder->error=error.what(); return -1; }
    catch (...) { decoder->error="unknown native decoder failure"; return -1; }
}
void nixe_video_frame_free(AVFrame* frame) noexcept { av_frame_free(&frame); }
int nixe_video_frame_copy(AVFrame* frame, int fd, int semaphore_fd, uint64_t allocation_size, const uint8_t* uuid,
                          uint64_t row_pitch, uint64_t chroma_offset, int initialize) noexcept {
#ifdef __linux__
    try {
        if (!frame || frame->format!=AV_PIX_FMT_CUDA || !frame->hw_frames_ctx)
            throw std::runtime_error("invalid CUDA video frame");
        if (row_pitch < static_cast<uint64_t>((frame->width+1)/2*2)
            || chroma_offset < row_pitch*frame->height
            || allocation_size < chroma_offset+row_pitch*((frame->height+1)/2))
            throw std::runtime_error("external video buffer does not contain both planes");
        auto& cuda=CudaTransfer::get();
        const auto* frames=reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
        const auto* context=reinterpret_cast<const AVCUDADeviceContext*>(frames->device_ctx->hwctx);
        cuda.check(cuda.cuCtxPushCurrent_v2(context->cuda_ctx));
        struct Scope {
            CudaTransfer& cuda; CUexternalMemory memory{}; CUexternalSemaphore semaphore{}; CUdeviceptr pointer{}; CUstream stream{}; bool synchronized{};
            ~Scope() { if (!synchronized) cuda.cuStreamSynchronize(stream); if (semaphore) cuda.cuDestroyExternalSemaphore(semaphore); if (pointer) cuda.cuMemFree_v2(pointer); if (memory) cuda.cuDestroyExternalMemory(memory); CUcontext popped{}; cuda.cuCtxPopCurrent_v2(&popped); }
        } scope{cuda};
        scope.stream=context->stream;
        CUdevice actual_device{}; CUuuid actual_uuid{};
        cuda.check(cuda.cuCtxGetDevice(&actual_device)); cuda.check(cuda.cuDeviceGetUuid(&actual_uuid,actual_device));
        if (std::memcmp(actual_uuid.bytes,uuid,16)!=0) throw std::runtime_error("decoded frame belongs to another GPU");
        if (!frame->data[0] || !frame->data[1] || frame->linesize[0]<frame->width || frame->linesize[1]<(frame->width+1)/2*2) throw std::runtime_error("invalid NV12 device plane strides");
        // CUDA owns a successfully imported FD; the caller keeps its original.
        // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC descriptor{};
        descriptor.type=CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        descriptor.handle.fd=dup(fd); descriptor.size=allocation_size; descriptor.flags=1; // dedicated allocation
        if (descriptor.handle.fd<0) throw std::runtime_error("cannot duplicate Vulkan memory FD");
        const auto imported=cuda.cuImportExternalMemory(&scope.memory,&descriptor);
        if (imported!=CUDA_SUCCESS) close(descriptor.handle.fd);
        cuda.check(imported);
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer{}; buffer.size=allocation_size;
        cuda.check(cuda.cuExternalMemoryGetMappedBuffer(&scope.pointer,scope.memory,&buffer));
        if (initialize) cuda.check(cuda.cuMemsetD8Async(scope.pointer,0,allocation_size,context->stream));
        for (int plane=0;plane<2;++plane) {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType=CU_MEMORYTYPE_DEVICE; copy.srcDevice=reinterpret_cast<CUdeviceptr>(frame->data[plane]); copy.srcPitch=frame->linesize[plane];
            copy.dstMemoryType=CU_MEMORYTYPE_DEVICE; copy.dstDevice=scope.pointer+(plane ? chroma_offset : 0); copy.dstPitch=row_pitch;
            copy.WidthInBytes=plane ? (frame->width+1)/2*2 : frame->width;
            copy.Height=plane ? (frame->height+1)/2 : frame->height;
            cuda.check(cuda.cuMemcpy2DAsync_v2(&copy,context->stream));
        }
        CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC sem{};
        sem.type=CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD; sem.handle.fd=dup(semaphore_fd);
        if (sem.handle.fd<0) throw std::runtime_error("cannot duplicate Vulkan semaphore FD");
        const auto sem_imported=cuda.cuImportExternalSemaphore(&scope.semaphore,&sem);
        if (sem_imported!=CUDA_SUCCESS) close(sem.handle.fd);
        cuda.check(sem_imported);
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal{};
        cuda.check(cuda.cuSignalExternalSemaphoresAsync(&scope.semaphore,&signal,1,context->stream));
        // Complete actual device writes before Vulkan consumes the shared buffer.
        cuda.check(cuda.cuStreamSynchronize(context->stream));
        scope.synchronized=true;
        return 1;
    } catch (const std::exception& e) { native_error=e.what(); return -1; }
#else
    native_error="CUDA/Vulkan video interop requires Linux"; return -1;
#endif
}

}
