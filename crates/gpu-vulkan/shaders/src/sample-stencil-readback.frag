#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(set=0, binding=0) uniform utexture2DMS sourceImage;
layout(push_constant) uniform Parameters { uvec4 gridRowKind; } params;
layout(location=0) out uint encoded;
void main() {
    uvec2 grid = params.gridRowKind.xy;
    uvec2 at = uvec2(gl_FragCoord.xy) + uvec2(0, params.gridRowKind.z);
    int sampleIndex = int((at.y % grid.y) * grid.x + at.x % grid.x);
    encoded = texelFetch(sourceImage, ivec2(at / grid), sampleIndex).r;
}
