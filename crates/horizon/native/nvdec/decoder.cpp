// SPDX-License-Identifier: GPL-3.0-or-later
// FFmpeg owns the codec state and reference frames; the bridge hides its ABI from Rust.
// https://ffmpeg.org/doxygen/trunk/group__lavc__encdec.html
#include "vp9.h"
#include <cstring>
#include <string>
#include <memory>
#ifdef __linux__
#include "cuda_device.h"
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
Decoder* nixe_nvdec_create(uint32_t codec_id, const uint8_t* uuid) noexcept {
    try {
        auto decoder=std::make_unique<Decoder>();
        const auto id = codec_id == 3 ? AV_CODEC_ID_H264 : codec_id == 9 ? AV_CODEC_ID_VP9 : AV_CODEC_ID_NONE;
        const AVCodec* codec=avcodec_find_decoder(id);
        if (!codec) throw std::runtime_error("requested FFmpeg decoder unavailable");
        decoder->context=avcodec_alloc_context3(codec);
        decoder->frame=av_frame_alloc();
        if (!decoder->context || !decoder->frame) throw std::bad_alloc();
#ifdef __linux__
        if (uuid) {
            auto& cuda = CudaDevice::get();
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
int nixe_nvdec_send_packet(Decoder* decoder, const uint8_t* bytes, size_t length, uint64_t origin) noexcept {
    try {
        AVPacket* packet=av_packet_alloc();
        if (!packet) throw std::bad_alloc();
        std::unique_ptr<AVPacket, void(*)(AVPacket*)> owner(packet, [](AVPacket* p){av_packet_free(&p);});
        decoder->check(av_new_packet(packet,static_cast<int>(length)));
        std::memcpy(packet->data,bytes,length);
        packet->pts=packet->dts=static_cast<int64_t>(origin);
        decoder->check(avcodec_send_packet(decoder->context,packet));
        return 1;
    } catch (const std::exception& error) { decoder->error=error.what(); return -1; }
    catch (...) { decoder->error="unknown native decoder failure"; return -1; }
}
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
// Every output owns an AVFrame reference, independent of subsequent decoder receives.
// No AVFrame layout crosses FFI.
int nixe_nvdec_receive(Decoder* decoder, uint64_t* origin, uint32_t* width, uint32_t* height,
                      AVFrame** retained_frame) noexcept {
    try {
        av_frame_unref(decoder->frame);
        const int result=avcodec_receive_frame(decoder->context,decoder->frame);
        if (result==AVERROR(EAGAIN) || result==AVERROR_EOF) return 0;
        decoder->check(result);
        const AVFrame* frame=decoder->frame;
        const bool hardware = frame->format==AV_PIX_FMT_CUDA;
        if (!hardware && frame->format!=AV_PIX_FMT_YUV420P)
            throw std::runtime_error("NVDEC output requires an unsupported pixel format");
        if (frame->pts<=0 || frame->width<=0 || frame->height<=0)
            throw std::runtime_error("NVDEC decoded frame has invalid dimensions or origin");
        *origin=static_cast<uint64_t>(frame->pts); *width=frame->width; *height=frame->height;
        if (hardware) {
            const auto* frames=reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
            if (frames->sw_format!=AV_PIX_FMT_NV12) throw std::runtime_error("NVDEC CUDA output is not NV12");
            AVFrame* host=av_frame_alloc();
            if (!host) throw std::bad_alloc();
            std::unique_ptr<AVFrame, void(*)(AVFrame*)> owner(host, [](AVFrame* p){av_frame_free(&p);});
            host->format=AV_PIX_FMT_NV12;
            // Release the finite decoder surface immediately. The retained
            // host pixels remain independent of later receive calls.
            // https://ffmpeg.org/doxygen/trunk/hwcontext_8h.html
            decoder->check(av_hwframe_transfer_data(host, frame, 0));
            av_frame_unref(decoder->frame);
            av_frame_move_ref(decoder->frame, host);
        }
        *retained_frame=av_frame_clone(frame);
        if (!*retained_frame) throw std::bad_alloc();
        return 1;
    } catch (const std::exception& error) { decoder->error=error.what(); return -1; }
    catch (...) { decoder->error="unknown native decoder failure"; return -1; }
}
// Borrow host plane metadata from an owned frame. The decoder has already
// completed any hardware transfer before publishing this reference to VIC.
int nixe_video_frame_planes(const AVFrame* frame, const uint8_t** planes, int32_t* strides) noexcept {
    if (frame->format != AV_PIX_FMT_NV12 && frame->format != AV_PIX_FMT_YUV420P) {
        native_error="unsupported host video pixel format";
        return -1;
    }
    const int count=frame->format == AV_PIX_FMT_NV12 ? 2 : 3;
    for (int i=0;i<3;++i) {
        planes[i]=i<count ? frame->data[i] : nullptr;
        strides[i]=i<count ? frame->linesize[i] : 0;
    }
    return count;
}
void nixe_video_frame_free(AVFrame* frame) noexcept { av_frame_free(&frame); }
// Capability selection precedes decoder creation; an unavailable CUDA device
// preserves the existing software decoder. Actual decoding failures stay errors.
int nixe_video_hardware_supported(uint32_t codec_id, const uint8_t* uuid) noexcept {
#ifdef __linux__
    if (!uuid) return 0;
    void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) return 0;
    dlclose(library);
    try {
        if (CudaDevice::get().find_device(uuid) < 0) return 0;
        const auto id = codec_id == 3 ? AV_CODEC_ID_H264 : codec_id == 9 ? AV_CODEC_ID_VP9 : AV_CODEC_ID_NONE;
        const auto* codec = avcodec_find_decoder(id);
        if (!codec) throw std::runtime_error("requested FFmpeg decoder unavailable");
        for (int i=0;; ++i) {
            const auto* config = avcodec_get_hw_config(codec, i);
            if (!config) return 0;
            if (config->device_type == AV_HWDEVICE_TYPE_CUDA
                && (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) return 1;
        }
    } catch (const std::exception& error) { native_error=error.what(); return -1; }
#else
    (void)codec_id; (void)uuid; return 0;
#endif
}
}
