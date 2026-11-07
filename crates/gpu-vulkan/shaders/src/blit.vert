#version 450
// Eden full-screen blit vertex coordinates and source scale/offset.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/blit_image.cpp
layout(push_constant) uniform Parameters { vec2 scale; vec2 offset; } p;
layout(location=0) out vec2 texcoord;
void main() {
    vec2 vertex = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(vertex * 2.0 - 1.0, 0.0, 1.0);
    texcoord = p.offset + vertex * p.scale;
}
