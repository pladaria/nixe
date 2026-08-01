use std::fs;
use std::path::Path;

#[test]
fn headless_backend_depends_only_on_the_neutral_gpu_contract() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let contents = fs::read_to_string(manifest.join("Cargo.toml"))
        .expect("headless GPU manifest must be readable");
    let dependencies = dependency_names(&contents);

    assert_eq!(dependencies, ["nixe-gpu"]);
    for prohibited in [
        "nixe-cpu",
        "nixe-runtime",
        "nixe-horizon",
        "nixe-gpu-maxwell",
        "nixe-video",
        "nixe-video-winit",
        "wgpu",
        "winit",
    ] {
        assert!(!dependencies.contains(&prohibited));
    }
}

fn dependency_names(manifest: &str) -> Vec<&str> {
    let mut in_dependencies = false;
    let mut dependencies = Vec::new();
    for raw_line in manifest.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') {
            in_dependencies = line == "[dependencies]";
            continue;
        }
        if in_dependencies
            && !line.is_empty()
            && !line.starts_with('#')
            && let Some((name, _)) = line.split_once('=')
        {
            dependencies.push(
                name.trim()
                    .strip_suffix(".workspace")
                    .unwrap_or(name.trim()),
            );
        }
    }
    dependencies
}
