#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_samplerless_texture_functions : require
layout(set=0,binding=0) uniform texture2D source_image;
layout(push_constant) uniform Parameters {
    ivec4 source_rect; // lower inclusive, upper exclusive
    ivec2 destination_origin;
    uint linear_filter;
    uint padding;
    i64vec2 first;
    i64vec2 step;
} p;
layout(location=0) out vec4 color;
vec4 fetch(i64vec2 at) {
    return texelFetch(source_image,ivec2(clamp(at,i64vec2(p.source_rect.xy),i64vec2(p.source_rect.zw)-1)),0);
}
void main() {
    i64vec2 pixel=i64vec2(ivec2(gl_FragCoord.xy)-p.destination_origin);
    i64vec2 at=p.first+pixel*p.step;
    if (p.linear_filter==0) {color=fetch(at>>32);return;}
    at-=int64_t(1)<<31;
    i64vec2 base=at>>32;
    vec2 fraction=vec2(uvec2(at))* (1.0/4294967296.0);
    color=mix(mix(fetch(base),fetch(base+i64vec2(1,0)),fraction.x),
              mix(fetch(base+i64vec2(0,1)),fetch(base+i64vec2(1,1)),fraction.x),fraction.y);
}
