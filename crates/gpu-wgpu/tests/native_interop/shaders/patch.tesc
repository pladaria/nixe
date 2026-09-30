#version 450
layout(vertices = 3) out;
layout(set = 0, binding = 0, std430) readonly buffer Parameters { vec4 value; } parameters;
layout(location = 0) patch out vec4 tint;
void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    if (gl_InvocationID == 0) {
        tint = parameters.value;
        gl_TessLevelInner[0] = parameters.value.a + 1;
        gl_TessLevelOuter[0] = parameters.value.a + 1;
        gl_TessLevelOuter[1] = parameters.value.a + 1;
        gl_TessLevelOuter[2] = parameters.value.a + 1;
    }
}
