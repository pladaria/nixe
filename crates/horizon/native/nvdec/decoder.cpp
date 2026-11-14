// SPDX-License-Identifier: GPL-3.0-or-later
#include "vp9.h"
#include "../video/decoder.h"
#include <cstring>
#include <memory>
extern "C" {
Tegra::Decoders::VP9* nixe_vp9_create() noexcept {
    try { return new Tegra::Decoders::VP9; } catch (...) { return nullptr; }
}
void nixe_vp9_destroy(Tegra::Decoders::VP9* headers) noexcept { delete headers; }
int nixe_nvdec_send(Decoder* decoder, Tegra::Decoders::VP9* headers, const uint8_t* picture, const uint8_t* entropy,
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
        decoder->check(headers->ComposeFrame(info,probs,{tiles,length},refs,packet));
        packet->pts=packet->dts=static_cast<int64_t>(origin);
        decoder->check(avcodec_send_packet(decoder->context,packet));
        // Hidden pictures update the codec references without a VIC frame.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/host1x/ffmpeg.cpp#L443-L450
        return headers->IsFrameVisible() ? 1 : 0;
    } catch (const std::exception& error) { decoder->error=error.what(); return -1; }
    catch (...) { decoder->error="unknown native decoder failure"; return -1; }
}
}
