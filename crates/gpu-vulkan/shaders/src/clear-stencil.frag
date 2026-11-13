#version 450
layout(push_constant) uniform Clear {
    float depth;
};
void main() {
    gl_FragDepth = depth;
}
