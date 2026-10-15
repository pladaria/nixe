#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(set=0, binding=0) uniform texture2DMS sourceImage;
layout(push_constant) uniform Parameters { uvec4 gridRowKind; } params;
layout(location=0) out uint encoded;
// Recover the exact D24 integer from its normalized binary32 sample. A
// binary32 multiply may round to a half-integer before round() sees it.
// The 24x24-bit integer product keeps the normalization inverse exact.
uint depth24(float value) {
    uint bits = floatBitsToUint(value);
    uint exponent = (bits >> 23) & 255u;
    if (exponent < 103u) return 0u;
    uint hi, lo;
    umulExtended((bits & 0x7fffffu) | 0x800000u, 16777215u, hi, lo);
    uint shift = 150u - exponent;
    if (shift > 32u) return (hi >> (shift - 32u)) + ((hi >> (shift - 33u)) & 1u);
    if (shift == 32u) return hi + (lo >> 31);
    return ((lo >> shift) | (hi << (32u - shift))) + ((lo >> (shift - 1u)) & 1u);
}
void main() {
    uvec2 grid = params.gridRowKind.xy;
    uvec2 at = uvec2(gl_FragCoord.xy) + uvec2(0, params.gridRowKind.z);
    int sampleIndex = int((at.y % grid.y) * grid.x + at.x % grid.x);
    vec4 value = texelFetch(sourceImage, ivec2(at / grid), sampleIndex);
    uint kind = params.gridRowKind.w;
    if (kind == 1u) {
        encoded = depth24(value.r);
    } else if (kind == 2u) {
        encoded = floatBitsToUint(value.r);
    } else {
        if (kind == 3u) value = value.bgra;
        encoded = packUnorm4x8(value);
    }
}
