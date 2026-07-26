use std::fs;
use std::path::Path;

#[test]
fn maxwell_frontend_depends_only_on_neutral_gpu_contracts() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let contents =
        fs::read_to_string(manifest.join("Cargo.toml")).expect("Maxwell manifest must be readable");
    let dependencies = dependency_names(&contents);

    assert_eq!(dependencies, ["nixe-gpu"]);
    for prohibited in [
        "nixe-horizon",
        "nixe-video",
        "nixe-video-winit",
        "wgpu",
        "winit",
    ] {
        assert!(!dependencies.contains(&prohibited));
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
