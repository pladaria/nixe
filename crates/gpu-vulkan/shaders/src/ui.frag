#version 450
layout(location=0) in vec2 uv;
layout(location=1) in vec4 tint;
layout(location=0) out vec4 color;
layout(set=0,binding=0) uniform texture2D atlas;
layout(set=0,binding=1) uniform sampler atlas_sampler;
void main() {
    color = tint * texture(sampler2D(atlas, atlas_sampler), uv);
}
