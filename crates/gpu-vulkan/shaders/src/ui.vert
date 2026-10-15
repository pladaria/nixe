#version 450
layout(location=0) in vec2 position;
layout(location=1) in vec2 texture_uv;
layout(location=2) in vec4 vertex_color;
layout(location=0) out vec2 uv;
layout(location=1) out vec4 tint;
layout(push_constant) uniform Parameters { vec2 size; } p;
void main() {
    gl_Position = vec4(position / p.size * 2.0 - 1.0, 0.0, 1.0);
    uv = texture_uv;
    tint = vertex_color;
}
