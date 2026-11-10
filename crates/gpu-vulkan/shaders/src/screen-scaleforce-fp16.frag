#version 450
#extension GL_GOOGLE_include_directive : require
#define PACKED 0
#define YUZU_USE_FP16 1
#include "screen.inc"
#include "screen-sampling.inc"
#include "screen-scaleforce.inc"
