use std::fs;
use std::path::Path;

#[test]
fn neutral_memory_contract_has_no_platform_or_execution_dependencies() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let contents =
        fs::read_to_string(manifest.join("Cargo.toml")).expect("memory manifest must be readable");
    let dependencies = dependency_names(&contents);

    assert!(
        dependencies.is_empty(),
        "neutral memory crate must remain dependency-free, found {dependencies:?}"
    );
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
