#version 450
// Host-only interoperability fixture, not a replacement for guest shaders.
void main() {
    vec2 positions[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(positions[gl_VertexIndex], 0.25, 1);
}
