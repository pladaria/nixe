struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texture_coordinates: vec2<f32>,
}

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let position = positions[vertex_index];
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.texture_coordinates = position * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    return output;
}

@group(0) @binding(0)
var frame_texture: texture_2d<f32>;

@group(0) @binding(1)
var frame_sampler: sampler;

struct SamplingParameters {
    crop: vec4<f32>,
    transform: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
}

@group(0) @binding(2)
var<uniform> sampling: SamplingParameters;

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var coordinates = input.texture_coordinates;
    if ((sampling.transform & 4u) != 0u) {
        coordinates = vec2<f32>(coordinates.y, 1.0 - coordinates.x);
    }
    if ((sampling.transform & 1u) != 0u) {
        coordinates.x = 1.0 - coordinates.x;
    }
    if ((sampling.transform & 2u) != 0u) {
        coordinates.y = 1.0 - coordinates.y;
    }
    coordinates = sampling.crop.xy + coordinates * sampling.crop.zw;
    let color = textureSample(frame_texture, frame_sampler, coordinates);
    return vec4<f32>(color.rgb, 1.0);
}
