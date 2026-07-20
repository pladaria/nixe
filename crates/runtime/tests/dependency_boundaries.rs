use std::fs;
use std::path::Path;

#[test]
fn cpu_remains_independent_of_loader_and_runtime_crates() {
    let runtime_manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cpu_manifest = runtime_manifest.join("../cpu/Cargo.toml");
    let manifest = fs::read_to_string(cpu_manifest).expect("CPU manifest must be readable");
    let dependencies = dependency_names(&manifest);

    assert!(!dependencies.iter().any(|dependency| {
        matches!(*dependency, "swiitx-runtime" | "swiitx-loader-executable")
    }));
}

#[test]
fn runtime_owns_the_cpu_and_executable_loader_dependency_direction() {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest = fs::read_to_string(manifest_path).expect("runtime manifest must be readable");
    let dependencies = dependency_names(&manifest);

    assert!(dependencies.contains(&"swiitx-cpu"));
    assert!(dependencies.contains(&"swiitx-loader-executable"));
}

fn dependency_names(manifest: &str) -> Vec<&str> {
    let mut in_dependency_table = false;
    let mut dependencies = Vec::new();
    for raw_line in manifest.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') {
            in_dependency_table = line == "[dependencies]";
            continue;
        }
        if in_dependency_table
            && !line.is_empty()
            && !line.starts_with('#')
            && let Some((name, _)) = line.split_once('=')
        {
            let name = name.trim().trim_matches(['\'', '"']);
            dependencies.push(name.strip_suffix(".workspace").unwrap_or(name));
        }
    }
    dependencies
}
