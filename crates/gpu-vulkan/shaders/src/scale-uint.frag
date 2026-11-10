#version 450
#extension GL_GOOGLE_include_directive : require
#define SCALE_MULTISAMPLE 0
#define SCALE_DEPTH 0
#define SCALE_STENCIL 0
#define SCALE_INTEGER 1
#define SCALE_VALUE uvec4
#define SCALE_TEXTURE usampler2D
#include "scale.inc"
