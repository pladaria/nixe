fn main() {
    let codec = pkg_config::Config::new()
        .atleast_version("58")
        .probe("libavcodec")
        .expect("NVDEC emulation requires the FFmpeg libavcodec development package");
    let util = pkg_config::Config::new()
        .probe("libavutil")
        .expect("NVDEC emulation requires the FFmpeg libavutil development package");
    let mut build = cc::Build::new();
    build.cpp(true).std("c++20").warnings(true);
    for path in codec.include_paths.iter().chain(&util.include_paths) {
        build.include(path);
    }
    build
        .file("native/nvdec/vp9.cpp")
        .file("native/nvdec/decoder.cpp")
        .compile("nixe_nvdec");
    for file in [
        "vp9.cpp",
        "vp9.h",
        "types.h",
        "decoder.cpp",
        "cuda_device.h",
        "dynlink_cuda.h",
    ] {
        println!("cargo:rerun-if-changed=native/nvdec/{file}");
    }
}
