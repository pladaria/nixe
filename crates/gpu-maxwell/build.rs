fn main() {
    println!("cargo:rerun-if-changed=native");
    println!("cargo:rerun-if-changed=../../vendor/asmjit");
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64") {
        return;
    }
    let dst = cmake::Config::new("native").profile("Release").build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    for library in ["nixe_mme", "asmjit"] {
        println!("cargo:rustc-link-lib=static={library}");
    }
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => println!("cargo:rustc-link-lib=c++"),
        Ok("windows") if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") => {}
        _ => println!("cargo:rustc-link-lib=stdc++"),
    }
}
