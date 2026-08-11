use std::{fs, path::Path};

#[test]
fn neutral_engine_contract_has_only_cpu_and_memory_dependencies() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let dependencies = dependency_names(&manifest);
    assert_eq!(dependencies, ["nixe-cpu", "nixe-memory"]);
    for forbidden in ["nixe-runtime", "nixe-horizon", "nixe-scheduler", "nixe-gpu"] {
        assert!(!dependencies.contains(&forbidden));
    }

    let protocol = fs::read_to_string(root.join("src/lib.rs")).unwrap();
    assert!(protocol.contains("pub trait EngineDomain"));
    assert!(protocol.contains("fn create_executor("));
    assert!(protocol.contains("pub trait EngineExecutor"));
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
            result.push(
                name.trim()
                    .strip_suffix(".workspace")
                    .unwrap_or(name.trim()),
            );
        }
    }
    result
}
