struct ImportParameters {
    width: u32,
    height: u32,
    row_pitch: u32,
    source_size: u32,
    format: u32,
    memory_layout: u32,
    block_height_log2: u32,
    bytes_per_texel: u32,
}

@group(0) @binding(0)
var<storage, read> source: array<u32>;

@group(0) @binding(1)
var output: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> parameters: ImportParameters;

fn source_byte(offset: u32) -> u32 {
    let word = source[offset / 4u];
    return (word >> ((offset & 3u) * 8u)) & 0xffu;
}

fn source_offset(x: u32, y: u32) -> u32 {
    let byte_x = x * parameters.bytes_per_texel;
    if parameters.memory_layout == 0u {
        return y * parameters.row_pitch + byte_x;
    }

    // Tegra's generic 16Bx2 GOB addressing, matching pinned libnx's
    // framebuffer conversion:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/framebuffer.c
    let width_in_gobs = parameters.row_pitch / 64u;
    let block_height_gobs = 1u << parameters.block_height_log2;
    let block_rows = 8u * block_height_gobs;
    return (y / block_rows) * 512u * block_height_gobs * width_in_gobs
        + (byte_x / 64u) * 512u * block_height_gobs
        + ((y % block_rows) / 8u) * 512u
        + ((byte_x % 64u) / 32u) * 256u
        + ((y % 8u) / 2u) * 64u
        + ((byte_x % 32u) / 16u) * 32u
        + (y % 2u) * 16u
        + byte_x % 16u;
}

fn normalized(value: u32) -> f32 {
    return f32(value) / 255.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) position: vec3<u32>) {
    if position.x >= parameters.width || position.y >= parameters.height {
        return;
    }

    let offset = source_offset(position.x, position.y);
    if offset + parameters.bytes_per_texel > parameters.source_size {
        return;
    }

    let low = source_byte(offset);
    let high = source_byte(offset + 1u);
    var color: vec4<f32>;
    switch parameters.format {
        case 0u: {
            color = vec4<f32>(
                normalized(low),
                normalized(high),
                normalized(source_byte(offset + 2u)),
                normalized(source_byte(offset + 3u)),
            );
        }
        case 1u: {
            color = vec4<f32>(
                normalized(low),
                normalized(high),
                normalized(source_byte(offset + 2u)),
                1.0,
            );
        }
        case 2u: {
            color = vec4<f32>(
                normalized(source_byte(offset + 2u)),
                normalized(high),
                normalized(low),
                normalized(source_byte(offset + 3u)),
            );
        }
        case 3u: {
            let packed = low | (high << 8u);
            color = vec4<f32>(
                f32((packed >> 11u) & 0x1fu) / 31.0,
                f32((packed >> 5u) & 0x3fu) / 63.0,
                f32(packed & 0x1fu) / 31.0,
                1.0,
            );
        }
        default: {
            let packed = low | (high << 8u);
            color = vec4<f32>(
                f32(packed & 0x0fu) / 15.0,
                f32((packed >> 4u) & 0x0fu) / 15.0,
                f32((packed >> 8u) & 0x0fu) / 15.0,
                f32((packed >> 12u) & 0x0fu) / 15.0,
            );
        }
    }
    textureStore(output, vec2<i32>(position.xy), color);
}
