use std::fs;
use std::path::Path;

// Keep this allowlist explicit. Adding an entry is an architectural review of
// the CPU crate's ownership boundary, not merely a manifest edit.
const APPROVED_DEPENDENCIES: &[&str] = &["nixe-memory"];

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
            let name = name.trim().trim_matches(['\'', '"']);
            dependencies.push(name.strip_suffix(".workspace").unwrap_or(name));
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

#[test]
fn cpu_frontend_does_not_own_a_concrete_interpreter_engine() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(!root.join("src/interpreter").exists());
    let library = fs::read_to_string(root.join("src/lib.rs")).unwrap();
    assert!(!library.contains("mod interpreter"));

    for relative in [
        "src/coverage.rs",
        "src/decode/allocation.rs",
        "src/decode/table.rs",
    ] {
        let source = fs::read_to_string(root.join(relative)).unwrap();
        for forbidden in [
            "interpreter_coverage",
            "registration.interpreter",
            "pub interpreter:",
        ] {
            assert!(
                !source.contains(forbidden),
                "neutral frontend source `{relative}` retains concrete interpreter metadata `{forbidden}`"
            );
        }
    }
}
