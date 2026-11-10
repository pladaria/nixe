// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Code adapted from the following sources:
// - https://learnopengl.com/Advanced-Lighting/Bloom
// - https://www.rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/

// Ported from Eden present_gaussian.frag; the reconstruction kernel is unchanged.
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/host_shaders/present_gaussian.frag

const float offset[3] = float[](0.0, 1.3846153846, 3.2307692308);
const float weight[3] = float[](0.2270270270, 0.3162162162, 0.0702702703);

vec4 blurVertical(vec2 coord, vec2 norm) {
    vec4 result = vec4(0.0f);
    for (int i = 1; i < 3; i++) {
        result += sample_source(vec2(coord) + (vec2(0.0, offset[i]) * norm)) * weight[i];
        result += sample_source(vec2(coord) - (vec2(0.0, offset[i]) * norm)) * weight[i];
    }
    return result;
}

vec4 blurHorizontal(vec2 coord, vec2 norm) {
    vec4 result = vec4(0.0f);
    for (int i = 1; i < 3; i++) {
        result += sample_source(vec2(coord) + (vec2(offset[i], 0.0) * norm)) * weight[i];
        result += sample_source(vec2(coord) - (vec2(offset[i], 0.0) * norm)) * weight[i];
    }
    return result;
}

vec4 blurDiagonal(vec2 coord, vec2 norm) {
    vec4 result = vec4(0.0f);
    for (int i = 1; i < 3; i++) {
        result +=
            sample_source(vec2(coord) + (vec2(offset[i], offset[i]) * norm)) * weight[i];
        result +=
            sample_source(vec2(coord) - (vec2(offset[i], offset[i]) * norm)) * weight[i];
    }
    return result;
}

void main() {

    vec4 base = sample_source(vec2(source_uv())) * weight[0];
    vec2 tex_offset = 1.0f / vec2(source_size());

    // TODO(Blinkhawk): This code can be optimized through shader group instructions.
    vec4 horizontal = blurHorizontal( source_uv(), tex_offset);
    vec4 vertical = blurVertical( source_uv(), tex_offset);
    vec4 diagonalA = blurDiagonal( source_uv(), tex_offset);
    vec4 diagonalB = blurDiagonal( source_uv(), tex_offset * vec2(1.0, -1.0));
    vec4 combination = mix(mix(horizontal, vertical, 0.5f), mix(diagonalA, diagonalB, 0.5f), 0.5f);
    color = combination + base;

    encode_target(color);
}
