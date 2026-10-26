fn main() {
    pkg_config::Config::new()
        .atleast_version("1.1")
        .probe("opus")
        .expect("Opus decoding requires the libopus development package");
}
