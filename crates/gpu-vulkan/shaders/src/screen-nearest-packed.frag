#version 450
#extension GL_GOOGLE_include_directive : require
#define PACKED 1
#define NEAREST 1
#include "screen.inc"
#include "screen-sampling.inc"
void main() {
    color = sample_source(source_uv());
    encode_target(color);
}
