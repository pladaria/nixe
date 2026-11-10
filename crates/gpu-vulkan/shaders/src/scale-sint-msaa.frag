#version 450
#extension GL_GOOGLE_include_directive : require
#define SCALE_MULTISAMPLE 1
#define SCALE_DEPTH 0
#define SCALE_STENCIL 0
#define SCALE_INTEGER 1
#define SCALE_VALUE ivec4
#define SCALE_TEXTURE itexture2DMS
#include "scale.inc"
