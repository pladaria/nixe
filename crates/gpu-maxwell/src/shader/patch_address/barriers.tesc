#version 450 core
layout(vertices=3) out;
layout(location=0) out vec4 result[];
layout(location=1) patch out float sharedColor;
void main() {
    if (gl_InvocationID == 0) sharedColor = 0.25;
    barrier();
    float first = sharedColor;
    barrier();
    if (gl_InvocationID == 0) sharedColor = 0.75;
    barrier();
    result[gl_InvocationID] = vec4(first, sharedColor, first, 1.0);
    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = 1.0;
        gl_TessLevelOuter[1] = 1.0;
        gl_TessLevelOuter[2] = 1.0;
        gl_TessLevelInner[0] = 1.0;
    }
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
