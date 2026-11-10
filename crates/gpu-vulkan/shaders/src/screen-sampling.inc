// Decode packed framebuffers before filtering, with the existing UNORM8 rounding.
#ifndef NEAREST
#define NEAREST 0
#endif
#if PACKED
layout(set=0,binding=0) uniform utexture2D source_image;
#else
layout(set=0,binding=0) uniform texture2D source_image;
#endif
layout(set=0,binding=1) uniform sampler source_sampler;
ivec2 source_size() {
#if PACKED
    return textureSize(usampler2D(source_image, source_sampler), 0);
#else
    return textureSize(sampler2D(source_image, source_sampler), 0);
#endif
}
vec4 fetch_source(ivec2 at) {
    at = clamp(at, ivec2(0), source_size()-1);
#if PACKED
    uint raw = texelFetch(usampler2D(source_image, source_sampler), at, 0).r;
    vec4 decoded = p.flags.w != 0u
        ? vec4(float((raw >> 11) & 31u) / 31.0, float((raw >> 5) & 63u) / 63.0, float(raw & 31u) / 31.0, 1.0)
        : vec4(raw & 15u, (raw >> 4) & 15u, (raw >> 8) & 15u, (raw >> 12) & 15u) / 15.0;
    return roundEven(decoded * 255.0) / 255.0;
#else
    return texelFetch(sampler2D(source_image, source_sampler), at, 0);
#endif
}
vec4 sample_source(vec2 coord) {
#if PACKED
#if NEAREST
    return fetch_source(ivec2(floor(coord * vec2(source_size()))));
#else
    vec2 q = coord * vec2(source_size()) - 0.5;
    ivec2 at = ivec2(floor(q));
    vec2 f = fract(q);
    return mix(mix(fetch_source(at), fetch_source(at+ivec2(1,0)), f.x),
               mix(fetch_source(at+ivec2(0,1)), fetch_source(at+ivec2(1,1)), f.x), f.y);
#endif
#else
    return texture(sampler2D(source_image, source_sampler), coord);
#endif
}
vec4 gather_source(vec2 coord, int component) {
#if PACKED
    ivec2 at = ivec2(floor(coord * vec2(source_size()) - 0.5));
    return vec4(fetch_source(at+ivec2(0,1))[component],
                fetch_source(at+ivec2(1,1))[component],
                fetch_source(at+ivec2(1,0))[component], fetch_source(at)[component]);
#else
    // textureGather requires a compile-time component.
    if (component == 0) return textureGather(sampler2D(source_image, source_sampler), coord, 0);
    if (component == 1) return textureGather(sampler2D(source_image, source_sampler), coord, 1);
    return textureGather(sampler2D(source_image, source_sampler), coord, 2);
#endif
}
