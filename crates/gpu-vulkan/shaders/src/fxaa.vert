// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 460
layout(push_constant) uniform Parameters { vec4 metrics; } p;

out gl_PerVertex {
    vec4 gl_Position;
};

const vec2 vertices[3] =
    vec2[3](vec2(-1,-1), vec2(3,-1), vec2(-1, 3));

layout (location = 0) out vec4 posPos;

#define VERTEX_ID gl_VertexIndex

const float FXAA_SUBPIX_SHIFT = 0;

void main() {
  vec2 vertex = vertices[VERTEX_ID];
  gl_Position = vec4(vertex, 0.0, 1.0);
  vec2 vert_tex_coord = (vertex + 1.0) / 2.0;
  posPos.xy = vert_tex_coord;
  posPos.zw = vert_tex_coord - (0.5 + FXAA_SUBPIX_SHIFT) / p.metrics.zw;
}
