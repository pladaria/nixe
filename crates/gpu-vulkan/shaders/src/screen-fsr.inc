// FSR 1 integration follows Eden's fidelityfx_fsr.frag (GPL-2.0-or-later).
// Copyright 2021 yuzu Emulator Project.
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/host_shaders/fidelityfx_fsr.frag
#extension GL_EXT_shader_explicit_arithmetic_types : require
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
layout(push_constant) uniform Parameters {
    uvec4 con0; uvec4 con1; uvec4 con2; uvec4 con3;
    vec4 crop; uvec4 flags;
} p;
#include "screen-sampling.inc"
#define A_GPU 1
#define A_GLSL 1
#define FSR_RCAS_PASSTHROUGH_ALPHA 1
#if HALF
#define A_HALF 1
#endif
#include "../../../../vendor/fidelityfx-fsr/ffx_a.h"
#if EASU
#if HALF
#define FSR_EASU_H 1
AH4 FsrEasuRH(AF2 q) { return AH4(gather_source(q, 0)); }
AH4 FsrEasuGH(AF2 q) { return AH4(gather_source(q, 1)); }
AH4 FsrEasuBH(AF2 q) { return AH4(gather_source(q, 2)); }
#else
#define FSR_EASU_F 1
AF4 FsrEasuRF(AF2 q) { return gather_source(q, 0); }
AF4 FsrEasuGF(AF2 q) { return gather_source(q, 1); }
AF4 FsrEasuBF(AF2 q) { return gather_source(q, 2); }
#endif
#else
#if HALF
#define FSR_RCAS_H 1
AH4 FsrRcasLoadH(ASW2 q) { return AH4(fetch_source(ivec2(q))); }
void FsrRcasInputH(inout AH1 r, inout AH1 g, inout AH1 b) {}
#else
#define FSR_RCAS_F 1
AF4 FsrRcasLoadF(ASU2 q) { return fetch_source(q); }
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}
#endif
#endif
#include "../../../../vendor/fidelityfx-fsr/ffx_fsr1.h"
void main() {
    AU2 pos = AU2(gl_FragCoord.xy);
#if EASU
#if HALF
    AH3 result;
    FsrEasuH(result, pos, p.con0, p.con1, p.con2, p.con3);
#else
    AF3 result;
    FsrEasuF(result, pos, p.con0, p.con1, p.con2, p.con3);
#endif
    color = vec4(result, sample_source(p.crop.xy + uv * p.crop.zw).a);
#else
#if HALF
    AH4 result;
    FsrRcasH(result.r, result.g, result.b, result.a, pos, p.con0);
#else
    AF4 result;
    FsrRcasF(result.r, result.g, result.b, result.a, pos, p.con0);
#endif
    color = vec4(result);
#endif
    if (p.flags.z != 0u) color.a = 1.0;
}
