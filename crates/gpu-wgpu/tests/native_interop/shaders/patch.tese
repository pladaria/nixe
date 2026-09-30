#version 450
layout(triangles, equal_spacing, ccw) in;
layout(location = 0) patch in vec4 tint;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 1) uniform sampler2D source_image;
void main() {
    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position
                + gl_TessCoord.y * gl_in[1].gl_Position
                + gl_TessCoord.z * gl_in[2].gl_Position;
    color = tint * texelFetch(source_image, ivec2(0), 0);
}
