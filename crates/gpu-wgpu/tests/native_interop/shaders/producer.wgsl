@group(0) @binding(0) var<storage, read_write> parameters: vec4f;
@group(0) @binding(1) var destination: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1) fn main() {
    // The copy path uploads white parameters and green texels. This path
    // independently changes both resources, so a stale buffer OR image is visible.
    parameters = vec4f(0, 0, 1, 1);
    textureStore(destination, vec2i(0), vec4f(1, 1, 1, 1));
}
