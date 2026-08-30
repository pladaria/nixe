use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn direct_memory_support_sits_below_both_cpu_frontends() {
    let direct = manifest(".");
    let direct_dependencies = dependency_names(&direct);
    assert!(direct_dependencies.contains(&"nixe-cpu"));
    assert!(direct_dependencies.contains(&"nixe-memory"));
    assert!(!direct_dependencies.contains(&"nixe-cpu-jit"));
    assert!(!direct_dependencies.contains(&"nixe-cpu-interpreter"));

    for component in ["../cpu", "../memory"] {
        let contents = manifest(component);
        let dependencies = dependency_names(&contents);
        assert!(!dependencies.contains(&"nixe-cpu-direct-memory"));
        assert!(!dependencies.contains(&"nixe-cpu-jit"));
        assert!(!dependencies.contains(&"nixe-cpu-interpreter"));
    }

    for frontend in ["../cpu-jit", "../cpu-interpreter"] {
        assert!(
            dependency_names(&manifest(frontend)).contains(&"nixe-cpu-direct-memory"),
            "{} does not depend on shared direct-memory support",
            frontend,
        );
    }
}

fn manifest(relative: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(normalized_join(root, relative).join("Cargo.toml")).unwrap()
}

fn normalized_join(root: &Path, relative: &str) -> PathBuf {
    root.join(relative).canonicalize().unwrap()
}

fn dependency_names(manifest: &str) -> Vec<&str> {
    let mut in_dependencies = false;
    manifest
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.starts_with('[') {
                in_dependencies = line == "[dependencies]";
                return None;
            }
            if !in_dependencies || line.is_empty() || line.starts_with('#') {
                return None;
            }
            line.split_once('=').map(|(name, _)| {
                name.trim()
                    .split_once('.')
                    .map_or(name.trim(), |(package, _)| package)
            })
        })
        .collect()
}
