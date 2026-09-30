fn position(index: u32, depth: f32) -> vec4f {
    let positions = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    return vec4f(positions[index], depth, 1);
}
@vertex fn before(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return position(i, 0.125); }
@vertex fn behind(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return position(i, 0.5); }
@vertex fn front(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return position(i, 0); }
@fragment fn red() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
@fragment fn blue() -> @location(0) vec4f { return vec4f(0, 0, 1, 1); }

@group(0) @binding(0) var source_color: texture_2d<f32>;
@group(0) @binding(1) var source_depth: texture_depth_2d;
@fragment fn sample_result(@builtin(position) p: vec4f) -> @location(0) vec4f {
    return vec4f(textureLoad(source_color, vec2i(p.xy), 0).rgb,
                 textureLoad(source_depth, vec2i(p.xy), 0));
}
