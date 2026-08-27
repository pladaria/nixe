use std::{fs, path::Path};

#[test]
fn interpreter_layers_on_neutral_contracts_only() {
    let manifest =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")).unwrap();
    for forbidden in [
        "nixe-runtime",
        "nixe-cpu-jit",
        "nixe-horizon",
        "nixe-scheduler",
        "nixe-gpu",
    ] {
        assert!(!manifest.contains(&format!("{forbidden}.workspace")));
    }
}

#[test]
fn crate_physically_owns_the_reference_interpreter() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(root.join("src/interpreter/mod.rs").is_file());
    assert!(root.join("src/interpreter/a64/mod.rs").is_file());
    assert!(root.join("src/process.rs").is_file());
    assert!(root.join("tests/differential.rs").is_file());
    assert!(!root.join("tests/conformance.rs").exists());
    let library = fs::read_to_string(root.join("src/lib.rs")).unwrap();
    assert!(!library.contains("pub(crate) use nixe_cpu"));
}
