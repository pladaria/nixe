use std::fs;
use std::path::Path;

// Keep this allowlist explicit. Adding an entry is an architectural review of
// the CPU crate's ownership boundary, not merely a manifest edit.
const APPROVED_DEPENDENCIES: &[&str] = &[];

#[test]
fn manifest_contains_only_architecturally_approved_dependencies() {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest = fs::read_to_string(&manifest_path).expect("CPU manifest must be readable");

    for dependency in dependency_names(&manifest) {
        assert!(
            APPROVED_DEPENDENCIES.contains(&dependency),
            "CPU crate dependency `{dependency}` has not passed an ownership-boundary review"
        );
    }
}

fn dependency_names(manifest: &str) -> Vec<&str> {
    let mut in_dependency_table = false;
    let mut dependencies = Vec::new();

    for raw_line in manifest.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') {
            in_dependency_table = matches!(
                line,
                "[dependencies]" | "[dev-dependencies]" | "[build-dependencies]"
            ) || (line.starts_with("[target.")
                && line.ends_with(".dependencies]"));
            continue;
        }

        if in_dependency_table
            && !line.is_empty()
            && !line.starts_with('#')
            && let Some((name, _)) = line.split_once('=')
        {
            dependencies.push(name.trim().trim_matches(['\'', '"']));
        }
    }

    dependencies
}

#[test]
fn dependency_table_parser_covers_target_specific_dependencies() {
    let manifest = r#"
        [dependencies]
        serde = "1"

        [target.'cfg(unix)'.dependencies]
        host_runtime = { path = "../runtime" }

        [package.metadata.example]
        ignored = "value"
    "#;

    assert_eq!(dependency_names(manifest), ["serde", "host_runtime"]);
}
