#version 450
#extension GL_GOOGLE_include_directive : require
#define SCALE_MULTISAMPLE 1
#define SCALE_DEPTH 1
#define SCALE_STENCIL 0
#define SCALE_INTEGER 0
#define SCALE_VALUE float
#define SCALE_TEXTURE texture2DMS
#include "scale.inc"
