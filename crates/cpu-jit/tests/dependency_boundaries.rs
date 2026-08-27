use std::{fs, path::Path};

#[test]
fn jit_owns_cranelift_without_a_production_interpreter_or_runtime_dependency() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let production_dependencies = manifest
        .split("[dev-dependencies]")
        .next()
        .expect("the manifest has a production dependency section");

    for required in [
        "cranelift-codegen = { version = \"=0.134.3\"",
        "cranelift-frontend = \"=0.134.3\"",
        "cranelift-jit = \"=0.134.3\"",
        "cranelift-module = \"=0.134.3\"",
        "cranelift-native = \"=0.134.3\"",
        "nixe-cpu.workspace = true",
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
        "nixe-cpu-interpreter.workspace",
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
    assert!(
        manifest.contains("[dev-dependencies]\nnixe-cpu-interpreter.workspace = true"),
        "the reference interpreter must remain test-only"
    );
    for forbidden in ["dynasm", "iced-x86"] {
        assert!(
            !manifest.contains(forbidden),
            "unexpected native-code dependency: {forbidden}"
        );
    }

    let library = fs::read_to_string(root.join("src/lib.rs")).unwrap();
    assert!(
        library.contains("#[cfg(test)]\nmod direct;"),
        "the replacement JIT must remain internal until the atomic S09 cutover"
    );
    assert!(
        !library.contains("pub mod direct;"),
        "the replacement JIT must not be exposed as a production selector"
    );

    let direct = root.join("src/direct");
    for path in fs::read_dir(direct)
        .unwrap()
        .map(|entry| entry.unwrap().path())
    {
        if path.extension().is_some_and(|extension| extension == "rs") {
            let source = fs::read_to_string(&path).unwrap();
            for forbidden in [
                "nixe_cpu::ir",
                "IrOperation",
                "OperationKind",
                "translate_region",
                "CompilationPool",
                "CodeTier",
                "Promotion",
                "ExecutionFrame",
                "HelperScratch",
                "SemanticHelper",
            ] {
                assert!(
                    !source.contains(forbidden),
                    "{} imports the superseded JIT architecture through {forbidden}",
                    path.display()
                );
            }
        }
    }
}

#[test]
fn neutral_crates_do_not_depend_on_cranelift_or_the_jit_backend() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for relative in ["crates/cpu/Cargo.toml", "crates/memory/Cargo.toml"] {
        let manifest = fs::read_to_string(workspace.join(relative)).unwrap();
        assert!(
            !manifest.contains("cranelift"),
            "{relative} imports Cranelift"
        );
        assert!(
            !manifest.contains("nixe-cpu-jit"),
            "{relative} imports the concrete JIT backend"
        );
    }

    let runtime = fs::read_to_string(workspace.join("crates/runtime/Cargo.toml")).unwrap();
    assert!(runtime.contains("nixe-cpu-interpreter.workspace = true"));
    assert!(runtime.contains("nixe-cpu-jit.workspace = true"));
}
