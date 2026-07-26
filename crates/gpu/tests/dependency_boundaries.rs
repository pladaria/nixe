use std::fs;
use std::path::Path;

#[test]
fn neutral_gpu_contract_has_no_console_or_host_backend_dependencies() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let contents =
        fs::read_to_string(manifest.join("Cargo.toml")).expect("GPU manifest must be readable");

    for prohibited in [
        "nixe-horizon",
        "nixe-video",
        "nixe-video-winit",
        "wgpu",
        "winit",
    ] {
        assert!(
            !dependency_names(&contents).contains(&prohibited),
            "neutral GPU crate must not depend on {prohibited}"
        );
    }
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
