// Ported from Eden present_area.frag; the reconstruction kernel is unchanged.
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/host_shaders/present_area.frag
/***** Area Sampling *****/

// By Sam Belliveau and Filippo Tarpini. Public Domain license.
// Effectively a more accurate sharp bilinear filter when upscaling,
// that also works as a mathematically perfect downscale filter.
// https://entropymine.com/imageworsener/pixelmixing/
// https://github.com/obsproject/obs-studio/pull/1715
// https://legacy.imagemagick.org/Usage/filter/
vec4 AreaSampling(vec2 texCoords, vec2 source_size, vec2 target_size) {
    // Determine the sizes of the source and target images.
    vec2 inverted_target_size = vec2(1.0) / target_size;

    // Determine the range of the source image that the target pixel will cover.
    vec2 range = source_size * inverted_target_size;
    vec2 beg = (texCoords.xy * source_size) - (range * 0.5);
    vec2 end = beg + range;

    // Compute the top-left and bottom-right corners of the pixel box.
    ivec2 f_beg = ivec2(floor(beg));
    ivec2 f_end = ivec2(floor(end));

    // Compute how much of the start and end pixels are covered horizontally & vertically.
    float area_w = 1.0 - fract(beg.x);
    float area_n = 1.0 - fract(beg.y);
    float area_e = fract(end.x);
    float area_s = fract(end.y);

    // Compute the areas of the corner pixels in the pixel box.
    float area_nw = area_n * area_w;
    float area_ne = area_n * area_e;
    float area_sw = area_s * area_w;
    float area_se = area_s * area_e;

    // Initialize the color accumulator.
    vec4 avg_color = vec4(0.0, 0.0, 0.0, 0.0);

    // Accumulate corner pixels.
    avg_color += area_nw * fetch_source(ivec2(f_beg.x, f_beg.y));
    avg_color += area_ne * fetch_source(ivec2(f_end.x, f_beg.y));
    avg_color += area_sw * fetch_source(ivec2(f_beg.x, f_end.y));
    avg_color += area_se * fetch_source(ivec2(f_end.x, f_end.y));

    // Determine the size of the pixel box.
    int x_range = int(f_end.x - f_beg.x - 0.5);
    int y_range = int(f_end.y - f_beg.y - 0.5);

    // Accumulate top and bottom edge pixels.
    for (int x = f_beg.x + 1; x <= f_beg.x + x_range; ++x) {
        avg_color += area_n * fetch_source(ivec2(x, f_beg.y));
        avg_color += area_s * fetch_source(ivec2(x, f_end.y));
    }

    // Accumulate left and right edge pixels and all the pixels in between.
    for (int y = f_beg.y + 1; y <= f_beg.y + y_range; ++y) {
        avg_color += area_w * fetch_source(ivec2(f_beg.x, y));
        avg_color += area_e * fetch_source(ivec2(f_end.x, y));

        for (int x = f_beg.x + 1; x <= f_beg.x + x_range; ++x) {
            avg_color += fetch_source(ivec2(x, y));
        }
    }

    // Compute the area of the pixel box that was sampled.
    float area_corners = area_nw + area_ne + area_sw + area_se;
    float area_edges = float(x_range) * (area_n + area_s) + float(y_range) * (area_w + area_e);
    float area_center = float(x_range) * float(y_range);

    // Return the normalized average color.
    return avg_color / (area_corners + area_edges + area_center);
}

void main() {
    color = AreaSampling(source_uv(), vec2(source_size()), p.output_size);
    encode_target(color);
}
