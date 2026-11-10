// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// https://en.wikipedia.org/wiki/Lanczos_resampling

// Ported from Eden present_lanczos.frag; the reconstruction kernel is unchanged.
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/host_shaders/present_lanczos.frag

#define PI 3.1415926535897932384626433
float sinc(float x) {
    return x == 0.0f ? 1.0f : sin(PI * x) / (PI * x);
}
float lanczos(vec2 v, float a) {
    float d = length(v);
    return sinc(d) / sinc(d / a);
}
vec4 textureLanczos(vec2 p) {
    vec3 c_sum = vec3(0.0f);
    float w_sum = 0.0f;
    vec2 res = vec2(source_size());
    vec2 cc = floor(p * res) / res;
    // kernel size = (2r + 1)^2
    const int r = 3; //radius (1 = 3 steps)
#define LANCZOS_LOOP_STEP(x, y) \
    { \
        vec2 kp = 0.5f * (vec2(x, y) / res); /* 0.5 = half-pixel level resampling */ \
        vec2 uv = cc + kp; \
        float w = lanczos(kp, float(r)); \
        c_sum += w * sample_source(p + kp).rgb; \
        w_sum += w; \
    }

    for (int y = -r; y <= r; ++y) {
        LANCZOS_LOOP_STEP(-3, y);
        LANCZOS_LOOP_STEP(-2, y);
        LANCZOS_LOOP_STEP(-1, y);
        LANCZOS_LOOP_STEP(-0, y);
        LANCZOS_LOOP_STEP(+1, y);
        LANCZOS_LOOP_STEP(+2, y);
        LANCZOS_LOOP_STEP(+3, y);
    }
    return vec4(c_sum / w_sum, 1.0f);
}

void main() {
    color = textureLanczos(source_uv());
    encode_target(color);
}
