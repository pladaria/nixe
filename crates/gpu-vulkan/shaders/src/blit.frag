#version 450
// Eden BlitColor samples a selected view through its native sampler.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/host_shaders/blit_color_float.frag
layout(set=0,binding=0) uniform sampler2D source_image;
layout(location=0) in vec2 texcoord;
layout(location=0) out vec4 color;
void main() { color = textureLod(source_image, texcoord, 0.0); }
