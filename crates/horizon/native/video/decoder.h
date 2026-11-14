// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <stdexcept>
#include <string>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
}
struct Decoder {
    AVCodecContext* context{};
    AVFrame* frame{};
    std::string error;
    ~Decoder() {
        av_frame_free(&frame);
        avcodec_free_context(&context);
    }
    void check(int code) {
        if (code < 0) {
            char text[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(code, text, sizeof(text));
            throw std::runtime_error(text);
        }
    }
};
