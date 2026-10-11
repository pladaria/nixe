// SPDX-License-Identifier: GPL-3.0-or-later
// FFmpeg owns the codec state and reference frames; the bridge hides its ABI from Rust.
// https://ffmpeg.org/doxygen/trunk/group__lavc__encdec.html
#include "vp9.h"
#include <cstring>
#include <limits>
#include <string>
#include <memory>
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
Decoder* nixe_nvdec_create() noexcept {
    try {
        auto decoder=std::make_unique<Decoder>();
        const AVCodec* codec=avcodec_find_decoder(AV_CODEC_ID_VP9);
        if (!codec) return nullptr;
        decoder->context=avcodec_alloc_context3(codec);
        decoder->frame=av_frame_alloc();
        if (!decoder->context || !decoder->frame) return nullptr;
        // Slice threading preserves packet/surface order without frame-thread latency.
        decoder->context->thread_type=FF_THREAD_SLICE;
        decoder->context->thread_count=0;
        decoder->context->err_recognition=AV_EF_EXPLODE;
        decoder->check(avcodec_open2(decoder->context, codec, nullptr));
        return decoder.release();
    } catch (...) { return nullptr; }
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
        auto bytes=decoder->headers.ComposeFrame(info,probs,{tiles,length},refs,origin);
        if (bytes.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("NVDEC reconstructed VP9 packet exceeds codec size limit");
        AVPacket* packet=av_packet_alloc();
        if (!packet) throw std::bad_alloc();
        std::unique_ptr<AVPacket, void(*)(AVPacket*)> owner(packet, [](AVPacket* p){av_packet_free(&p);});
        decoder->check(av_new_packet(packet,static_cast<int>(bytes.size())));
        std::memcpy(packet->data,bytes.data(),bytes.size());
        packet->pts=packet->dts=static_cast<int64_t>(origin);
        decoder->check(avcodec_send_packet(decoder->context,packet));
        return 1;
    } catch (const std::exception& error) { decoder->error=error.what(); return -1; }
    catch (...) { decoder->error="unknown native decoder failure"; return -1; }
}
// Returns a borrowed frame until the next receive or destruction. No AVFrame layout crosses FFI.
int nixe_nvdec_receive(Decoder* decoder, uint64_t* origin, uint32_t* width, uint32_t* height,
                      const uint8_t** planes, int32_t* strides) noexcept {
    try {
        av_frame_unref(decoder->frame);
        const int result=avcodec_receive_frame(decoder->context,decoder->frame);
        if (result==AVERROR(EAGAIN) || result==AVERROR_EOF) return 0;
        decoder->check(result);
        const AVFrame* frame=decoder->frame;
        if (frame->format!=AV_PIX_FMT_YUV420P)
            throw std::runtime_error("NVDEC VP9 output requires an unsupported pixel format");
        if (frame->pts<=0 || frame->width<=0 || frame->height<=0)
            throw std::runtime_error("NVDEC decoded frame has invalid dimensions or origin");
        *origin=static_cast<uint64_t>(frame->pts); *width=frame->width; *height=frame->height;
        for (int i=0;i<3;++i) { planes[i]=frame->data[i]; strides[i]=frame->linesize[i]; }
        return 1;
    } catch (const std::exception& error) { decoder->error=error.what(); return -1; }
    catch (...) { decoder->error="unknown native decoder failure"; return -1; }
}
}
