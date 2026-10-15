#version 450
#extension GL_GOOGLE_include_directive : require
#include "screen.inc"
layout(set=0,binding=0) uniform utexture2D source_image;
layout(set=0,binding=1) uniform sampler source_sampler;
vec4 fetch_pixel(ivec2 at) {
    ivec2 size = textureSize(usampler2D(source_image, source_sampler), 0);
    uint raw = texelFetch(usampler2D(source_image, source_sampler), clamp(at, ivec2(0), size-1), 0).r;
    vec4 decoded = p.flags.w != 0u
        ? vec4(float((raw >> 11) & 31u) / 31.0, float((raw >> 5) & 63u) / 63.0, float(raw & 31u) / 31.0, 1.0)
        : vec4(raw & 15u, (raw >> 4) & 15u, (raw >> 8) & 15u, (raw >> 12) & 15u) / 15.0;
    // Framebuffer conversion defines UNORM8 texels before display filtering.
    // Preserve packed bits and that quantization independently of normalized
    // texture expansion precision (covered by real-GPU exact-byte tests).
    return roundEven(decoded * 255.0) / 255.0;
}
void main() {
    vec2 q = source_uv() * vec2(textureSize(usampler2D(source_image, source_sampler), 0)) - 0.5;
    ivec2 at = ivec2(floor(q)); vec2 f = fract(q);
    color = mix(mix(fetch_pixel(at), fetch_pixel(at+ivec2(1,0)), f.x), mix(fetch_pixel(at+ivec2(0,1)), fetch_pixel(at+ivec2(1,1)), f.x), f.y);
    encode_target(color);
}
