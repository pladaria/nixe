layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
layout(push_constant) uniform Parameters { vec4 crop; uvec4 flags; } p;
vec2 source_uv() {
    vec2 q = uv;
    if ((p.flags.x & 4u) != 0u) q = vec2(q.y, 1.0-q.x);
    if ((p.flags.x & 1u) != 0u) q.x = 1.0-q.x;
    if ((p.flags.x & 2u) != 0u) q.y = 1.0-q.y;
    return p.crop.xy + q * p.crop.zw;
}
void encode_target(inout vec4 c) {
    if (p.flags.z != 0u) c.a = 1.0;
}
