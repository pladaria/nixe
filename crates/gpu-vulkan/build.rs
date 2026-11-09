fn main() {
    println!("cargo:rerun-if-changed=native/present_counter.cpp");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let compiler = cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(2)
        .pic(true)
        .flag("-fvisibility=hidden")
        .get_compiler();
    let status = compiler
        .to_command()
        .args(["-shared", "-pthread", "native/present_counter.cpp", "-o"])
        .arg(output.join("libnixe_present_counter.so"))
        .status()
        .expect("cannot build the Vulkan presentation counter");
    assert!(status.success(), "Vulkan presentation counter build failed");
}
