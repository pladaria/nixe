// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
// Ported from Eden present_bicubic.frag; the reconstruction kernel is unchanged.
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/host_shaders/present_bicubic.frag

vec4 cubic(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    return vec4(1.0, x, x2, x3) * transpose(mat4x4(
         0.0,   2.0,  0.0,  0.0,
        -1.0,   0.0,  1.0,  0.0,
         2.0,  -5.0,  4.0, -1.0,
        -1.0,   3.0, -3.0,  1.0
    ) * (1.0 / 2.0));
}
vec4 textureBicubic(vec2 uv) {
    vec2 tex_size = vec2(source_size());
    vec2 cc_tex = uv * tex_size - 0.5f;
    vec2 fex = cc_tex - floor(cc_tex);
    vec4 xcubic = cubic(fex.x);
    vec4 ycubic = cubic(fex.y);
    vec4 c = floor(cc_tex).xxyy + vec2(-0.5f, 1.5f).xyxy;
    vec4 z = vec4(xcubic.yw, ycubic.yw);
    vec4 s = vec4(xcubic.xz, ycubic.xz) + z;
    vec4 offset = (c + z / s) * (1.0f / tex_size).xxyy;
    vec2 n = vec2(s.x / (s.x + s.y), s.z / (s.z + s.w));
    return mix(
        mix(sample_source(offset.yw), sample_source(offset.xw), n.x),
        mix(sample_source(offset.yz), sample_source(offset.xz), n.x),
    n.y);
}

void main() {
    color = textureBicubic(source_uv());
    encode_target(color);
}
