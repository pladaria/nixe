#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(set=0,binding=0) uniform texture2D luma;
layout(set=0,binding=1) uniform texture2D chroma;
layout(push_constant) uniform Parameters {
    ivec4 red; ivec4 green; ivec4 blue;
    uvec4 control; ivec4 clamp_alpha;
} parameters;
layout(location=0) out vec4 color;
int channel(ivec3 value,ivec4 row) {
    return (((value.x*row.x + value.y*row.y + value.z*row.z) >> parameters.control.z) + row.w) >> 8;
}
void main() {
    ivec2 position=ivec2(gl_FragCoord.xy)+ivec2(parameters.control.xy);
    int y=int(round(texelFetch(luma,position,0).r*255.0));
    ivec2 uv=ivec2(round(texelFetch(chroma,position/2,0).rg*255.0));
    ivec3 value=ivec3(y,uv)*4;
    ivec3 rgb=value;
    if(parameters.control.w!=0u) rgb=ivec3(channel(value,parameters.red),channel(value,parameters.green),channel(value,parameters.blue));
    rgb=clamp(clamp(rgb,ivec3(parameters.clamp_alpha.x),ivec3(parameters.clamp_alpha.y)),ivec3(0),ivec3(1023)) >> 2;
    color=vec4(vec3(rgb),float(parameters.clamp_alpha.z))/255.0;
}
