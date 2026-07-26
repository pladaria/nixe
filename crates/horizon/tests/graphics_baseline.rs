use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

fn baseline_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../roms/homebrew/graphics/opengl/simple_triangle")
}

#[test]
fn triangle_baseline_manifest_identifies_the_retained_artifact_and_public_sources() {
    let directory = baseline_directory();
    let manifest =
        fs::read(directory.join("baseline.toml")).expect("baseline manifest must be readable");
    let manifest: toml::Value = toml::from_str(
        std::str::from_utf8(&manifest).expect("baseline manifest must contain UTF-8"),
    )
    .expect("baseline manifest must be valid TOML");
    let artifact = fs::read(directory.join("simple_triangle.nro"))
        .expect("retained simple_triangle artifact must be readable");

    assert_eq!(manifest["schema_version"].as_integer(), Some(1));
    assert_eq!(
        manifest["source"]["revision"].as_str(),
        Some("669786898205b7beb25ff1731e72982e6d0397d3")
    );
    assert_eq!(
        manifest["libnx"]["revision"].as_str(),
        Some("dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb")
    );
    assert_eq!(
        manifest["service_table"]["reference"].as_str(),
        Some("https://switchbrew.org/w/index.php?title=NV_services&oldid=14790")
    );
    assert_eq!(
        manifest["artifact"]["size"].as_integer(),
        i64::try_from(artifact.len()).ok()
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&artifact)),
        manifest["artifact"]["sha256"].as_str().unwrap()
    );
}

#[test]
fn triangle_baseline_classifies_the_first_gap_and_pins_its_wire_layout() {
    let manifest = fs::read_to_string(baseline_directory().join("baseline.toml"))
        .expect("baseline manifest must be readable");
    let manifest: toml::Value =
        toml::from_str(&manifest).expect("baseline manifest must be valid TOML");
    let ioctls = manifest["ioctl"].as_array().unwrap();
    let tpc_masks = ioctls
        .iter()
        .find(|ioctl| ioctl["request"].as_str() == Some("0xc0184706"))
        .expect("TPC-mask ioctl must be recorded");

    assert_eq!(tpc_masks["device"].as_str(), Some("/dev/nvhost-ctrl-gpu"));
    assert_eq!(tpc_masks["wire_size"].as_integer(), Some(24));
    assert_eq!(tpc_masks["direction"].as_str(), Some("inout"));
    assert_eq!(manifest["baseline"]["graphics_gap"].as_str(), Some("ioctl"));
    assert_eq!(
        manifest["baseline"]["host_pointers_allowed"].as_bool(),
        Some(false)
    );
}
