// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 460
#extension GL_GOOGLE_include_directive : enable
layout(push_constant) uniform Parameters { vec4 metrics; } p;

#define VERTEX_ID gl_VertexIndex

out gl_PerVertex {
    vec4 gl_Position;
};

const vec2 vertices[3] =
    vec2[3](vec2(-1,-1), vec2(3,-1), vec2(-1, 3));

layout (location = 0) out vec2 tex_coord;
layout (location = 1) out vec4 offset[3];

vec4 metrics = p.metrics;
#define SMAA_RT_METRICS metrics
#define SMAA_GLSL_4
#define SMAA_PRESET_ULTRA
#define SMAA_INCLUDE_VS 1
#define SMAA_INCLUDE_PS 0

#include "smaa.inc"

void main() {
    vec2 vertex = vertices[VERTEX_ID];
    gl_Position = vec4(vertex, 0.0, 1.0);
    tex_coord = (vertex + 1.0) / 2.0;
    SMAAEdgeDetectionVS(tex_coord, offset);
}
