use std::{fs, path::Path};

#[test]
fn jit_owns_cranelift_and_cold_interpretation_without_importing_runtime() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let production_dependencies = manifest
        .split("[dev-dependencies]")
        .next()
        .expect("the manifest has a production dependency section");

    for required in [
        "cranelift-codegen = { version = \"=0.134.3\"",
        "cranelift-frontend = \"=0.134.3\"",
        "cranelift-native = \"=0.134.3\"",
        "nixe-cpu.workspace = true",
        "nixe-cpu-engine.workspace = true",
        "nixe-cpu-engine-interpreter.workspace = true",
        "nixe-memory.workspace = true",
        "libc = \"0.2\"",
        "windows-sys = { version = \"0.61\"",
    ] {
        assert!(
            manifest.contains(required),
            "missing dependency: {required}"
        );
    }
    for forbidden in [
        "nixe-runtime.workspace",
        "nixe-horizon.workspace",
        "nixe-scheduler.workspace",
        "nixe-gpu.workspace",
    ] {
        assert!(
            !production_dependencies.contains(forbidden),
            "forbidden production dependency: {forbidden}"
        );
    }
    for forbidden in ["cranelift-jit", "cranelift-module", "dynasm", "iced-x86"] {
        assert!(
            !manifest.contains(forbidden),
            "unexpected native-code dependency: {forbidden}"
        );
    }
}

#[test]
fn shared_crates_do_not_depend_on_cranelift_or_the_jit_provider() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for relative in [
        "crates/cpu/Cargo.toml",
        "crates/cpu-engine/Cargo.toml",
        "crates/memory/Cargo.toml",
        "crates/runtime/Cargo.toml",
    ] {
        let manifest = fs::read_to_string(workspace.join(relative)).unwrap();
        assert!(
            !manifest.contains("cranelift"),
            "{relative} imports Cranelift"
        );
        assert!(
            !manifest.contains("nixe-cpu-engine-jit"),
            "{relative} imports the concrete JIT provider"
        );
    }
}
