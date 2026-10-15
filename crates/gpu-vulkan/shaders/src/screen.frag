#version 450
#extension GL_GOOGLE_include_directive : require
#include "screen.inc"
layout(set=0,binding=0) uniform texture2D source_image;
layout(set=0,binding=1) uniform sampler source_sampler;
void main() {
    color = texture(sampler2D(source_image, source_sampler), source_uv());
    encode_target(color);
}
