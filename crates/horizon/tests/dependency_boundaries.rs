use std::fs;
use std::path::Path;

#[test]
fn horizon_layers_on_runtime_without_a_reverse_dependency() {
    let horizon_manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let runtime_manifest = horizon_manifest.join("../runtime/Cargo.toml");
    let horizon_contents = fs::read_to_string(horizon_manifest.join("Cargo.toml"))
        .expect("Horizon manifest must be readable");
    let runtime_contents =
        fs::read_to_string(runtime_manifest).expect("runtime manifest must be readable");
    let horizon_dependencies = dependency_names(&horizon_contents);
    let runtime_dependencies = dependency_names(&runtime_contents);

    assert!(horizon_dependencies.contains(&"swiitx-runtime"));
    assert!(!runtime_dependencies.contains(&"swiitx-horizon"));
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
