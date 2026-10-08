// One invocation owns one destination word. Partial bytes and row/tile padding survive.
// Tegra block-linear GOB addressing:
// https://github.com/switchbrew/libnx/blob/master/nx/source/runtime/devices/console.c
struct Parameters { words: array<vec4<u32>, 8> }
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;
fn p(index: u32) -> u32 { return parameters.words[index / 4u][index % 4u]; }
fn position(base: u32, byte_x: u32, row_y: u32) -> u32 {
    if p(base + 5u) == 0u { return row_y * p(base) + byte_x; }
    let x = byte_x + p(base + 2u); let y = row_y + p(base + 3u);
    let height = 1u << p(base + 4u); let width_gobs = (p(base + 1u) + 63u) / 64u;
    return (y / (8u * height)) * 512u * height * width_gobs
        + (x / 64u) * 512u * height + ((y % (8u * height)) / 8u) * 512u
        + ((x % 64u) / 32u) * 256u + ((y % 8u) / 2u) * 64u
        + ((x % 32u) / 16u) * 32u + (y % 2u) * 16u + x % 16u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = id.x + id.y * groups.x * 64u;
    let word = p(2u) / 4u + index;
    if word * 4u >= p(2u) + p(3u) { return; }
    var value = destination[word];
    for (var lane = 0u; lane < 4u; lane += 1u) {
        let absolute = word * 4u + lane;
        if absolute < p(2u) || absolute >= p(2u) + p(3u) { continue; }
        let offset = absolute - p(2u);
        var x_byte: u32; var y: u32;
        if p(27u) == 0u {
            x_byte = offset % p(22u); y = offset / p(22u);
        } else {
            let height = 1u << p(26u); let row = ((p(23u) + 63u) / 64u) * 512u * height;
            let gob = offset % 512u;
            let bx = (offset % row) / (512u * height) * 64u + gob / 256u * 32u + gob % 64u / 32u * 16u + gob % 16u;
            let by = offset / row * 8u * height + offset % (512u * height) / 512u * 8u + gob % 256u / 64u * 2u + gob % 32u / 16u;
            if bx < p(24u) || by < p(25u) { continue; }
            x_byte = bx - p(24u); y = by - p(25u);
        }
        let x = x_byte / p(7u);
        if x >= p(4u) || y >= p(5u) { continue; }
        let within = x_byte % p(7u); let component_byte = within % p(8u);
        let component = p(12u + within / p(8u));
        if component == 6u { continue; }
        var byte: u32;
        if component < 4u {
            let address = p(0u) + position(16u, x * p(6u), y) + component * p(8u) + component_byte;
            byte = (source[address / 4u] >> ((address % 4u) * 8u)) & 255u;
        } else {
            byte = (p(select(10u, 11u, component == 5u)) >> (component_byte * 8u)) & 255u;
        }
        let shift = lane * 8u; value = (value & ~(255u << shift)) | (byte << shift);
    }
    destination[word] = value;
}
