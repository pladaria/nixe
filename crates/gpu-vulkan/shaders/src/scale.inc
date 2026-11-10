// Eden BlitScaleHelper: preserve individual MSAA samples and depth/stencil.
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_texture_cache.cpp#L2463
#extension GL_EXT_samplerless_texture_functions : require
#if SCALE_STENCIL
#extension GL_ARB_shader_stencil_export : require
#endif
layout(set=0, binding=0) uniform SCALE_TEXTURE sourceImage;
#if SCALE_STENCIL
#if SCALE_MULTISAMPLE
layout(set=0, binding=1) uniform utexture2DMS stencilImage;
#else
layout(set=0, binding=1) uniform usampler2D stencilImage;
#endif
#endif
layout(push_constant) uniform Parameters { vec2 destinationSize; } params;
#if !SCALE_DEPTH
layout(location=0) out SCALE_VALUE outputColor;
#endif
void main() {
#if SCALE_MULTISAMPLE
    ivec2 size = textureSize(sourceImage);
#else
    ivec2 size = textureSize(sourceImage, 0);
#endif
    vec2 uv = gl_FragCoord.xy / params.destinationSize;
    ivec2 at = clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1);
#if SCALE_MULTISAMPLE
    // Referencing SampleID requests per-sample execution, without averaging.
    #if SCALE_DEPTH
    float value = texelFetch(sourceImage, at, gl_SampleID).r;
    #else
    SCALE_VALUE value = texelFetch(sourceImage, at, gl_SampleID);
    #endif
#elif SCALE_DEPTH
    float value = texelFetch(sourceImage, at, 0).r;
#elif SCALE_INTEGER
    SCALE_VALUE value = texelFetch(sourceImage, at, 0);
#else
    SCALE_VALUE value = textureLod(sourceImage, uv, 0.0);
#endif
#if SCALE_DEPTH
    gl_FragDepth = value;
#else
    outputColor = value;
#endif
#if SCALE_STENCIL
    #if SCALE_MULTISAMPLE
    gl_FragStencilRefARB = int(texelFetch(stencilImage, at, gl_SampleID).r);
    #else
    gl_FragStencilRefARB = int(texelFetch(stencilImage, at, 0).r);
    #endif
#endif
}
