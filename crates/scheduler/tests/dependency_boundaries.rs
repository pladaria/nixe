use std::{fs, path::Path};

#[test]
fn scheduler_state_machine_has_no_execution_or_platform_dependencies() {
    let manifest =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")).unwrap();
    let dependencies = dependency_names(&manifest);
    assert!(
        dependencies.is_empty(),
        "scheduler dependencies: {dependencies:?}"
    );
}

fn dependency_names(manifest: &str) -> Vec<&str> {
    let mut active = false;
    let mut result = Vec::new();
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            active = line == "[dependencies]";
            continue;
        }
        if active && let Some((name, _)) = line.split_once('=') {
            result.push(name.trim());
        }
    }
    result
}
