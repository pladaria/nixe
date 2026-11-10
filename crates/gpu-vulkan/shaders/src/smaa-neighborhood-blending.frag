// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 460
#extension GL_GOOGLE_include_directive : enable
layout(push_constant) uniform Parameters { vec4 metrics; } p;
layout(set=0,binding=0) uniform sampler2D input_tex;
layout(set=0,binding=1) uniform sampler2D blend_tex;

layout (location = 0) in vec2 tex_coord;
layout (location = 1) in vec4 offset;

layout (location = 0) out vec4 frag_color;

vec4 metrics = p.metrics;
#define SMAA_RT_METRICS metrics
#define SMAA_GLSL_4
#define SMAA_PRESET_ULTRA
#define SMAA_INCLUDE_VS 0
#define SMAA_INCLUDE_PS 1

#include "smaa.inc"

void main() {
    frag_color = SMAANeighborhoodBlendingPS(tex_coord,
                                  offset,
                                  input_tex,
                                  blend_tex
                                  );
}
