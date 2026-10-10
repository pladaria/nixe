fn main() {
    println!("cargo:rerun-if-changed=native");
    println!("cargo:rerun-if-changed=../../vendor/dynarmic");
    let dst = cmake::Config::new("native")
        .profile("Release")
        .define(
            "NIXE_PERFORMANCE_COUNTERS",
            if std::env::var_os("CARGO_FEATURE_PERFORMANCE_COUNTERS").is_some() {
                "ON"
            } else {
                "OFF"
            },
        )
        .define("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    for library in ["nixe_dynarmic", "dynarmic", "fmt", "mcl"] {
        println!("cargo:rustc-link-lib=static={library}");
    }
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64") {
        for library in ["Zydis", "Zycore"] {
            println!("cargo:rustc-link-lib=static={library}");
        }
    }
    println!("cargo:rustc-link-lib=stdc++");
}
