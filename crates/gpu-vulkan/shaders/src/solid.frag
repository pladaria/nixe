#version 450
layout(location = 0) out vec4 color;
layout(push_constant) uniform Args { vec4 value; } args;
void main() { color = args.value; }
