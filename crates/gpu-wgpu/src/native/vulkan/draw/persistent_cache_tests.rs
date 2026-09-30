use super::*;

fn properties() -> vk::PhysicalDeviceProperties {
    vk::PhysicalDeviceProperties {
        vendor_id: 123,
        device_id: 456,
        driver_version: 789,
        pipeline_cache_uuid: [17; 16],
        ..Default::default()
    }
}

fn payload(p: &vk::PhysicalDeviceProperties) -> Vec<u8> {
    let mut bytes = Vec::new();
    for word in [32, 1, p.vendor_id, p.device_id] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes.extend_from_slice(&p.pipeline_cache_uuid);
    bytes.extend_from_slice(&[1, 2, 3, 4]);
    bytes
}

fn envelope(p: &vk::PhysicalDeviceProperties, data: &[u8]) -> Vec<u8> {
    let mut bytes = identity(p).to_vec();
    bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&Sha256::digest(data));
    bytes.extend_from_slice(data);
    bytes
}

#[test]
fn native_cache_envelope_checks_compatibility_integrity_and_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.bin");
    let p = properties();
    let data = payload(&p);
    assert!(load(&path, &p, 1024).unwrap().is_empty());
    let valid = envelope(&p, &data);
    std::fs::write(&path, &valid).unwrap();
    assert_eq!(load(&path, &p, data.len() as u64).unwrap(), data);
    assert!(load(&path, &p, data.len() as u64 - 1).unwrap().is_empty());
    // Every header field, checksum, and payload byte is covered.
    for i in 0..valid.len() {
        let mut bad = valid.clone();
        bad[i] ^= 1;
        std::fs::write(&path, bad).unwrap();
        assert!(load(&path, &p, 1024).unwrap().is_empty(), "byte {i}");
    }
    for length in 0..valid.len() {
        std::fs::write(&path, &valid[..length]).unwrap();
        assert!(load(&path, &p, 1024).unwrap().is_empty());
    }
    let mut extended = valid.clone();
    extended.push(0);
    std::fs::write(&path, extended).unwrap();
    assert!(load(&path, &p, 1024).unwrap().is_empty());
    std::fs::write(&path, valid).unwrap();
    for changed in [
        vk::PhysicalDeviceProperties {
            vendor_id: 124,
            ..p
        },
        vk::PhysicalDeviceProperties {
            device_id: 457,
            ..p
        },
        vk::PhysicalDeviceProperties {
            driver_version: 790,
            ..p
        },
        vk::PhysicalDeviceProperties {
            pipeline_cache_uuid: [18; 16],
            ..p
        },
    ] {
        assert!(load(&path, &changed, 1024).unwrap().is_empty());
    }
}

#[test]
fn native_cache_rejects_invalid_vulkan_header_even_with_valid_envelope() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.bin");
    let p = properties();
    for i in 0..32 {
        let mut data = payload(&p);
        data[i] ^= 1;
        std::fs::write(&path, envelope(&p, &data)).unwrap();
        assert!(load(&path, &p, 1024).unwrap().is_empty(), "Vulkan byte {i}");
    }
    std::fs::write(&path, envelope(&p, &[])).unwrap();
    assert!(load(&path, &p, 1024).unwrap().is_empty());
    // Filesystem errors are not hidden as guest execution success.
    assert!(load(dir.path(), &p, 1024).is_err());
}

#[test]
#[ignore = "requires Vulkan; exercises real bounded cache extraction and reload"]
fn native_pipeline_cache_roundtrip_and_small_bound() {
    let Some(backend) = crate::test_hardware::initialize_backend(
        nixe_gpu::BackendInstanceId::new(812),
        nixe_memory::NonCpuDeviceId::new(812),
        crate::WgpuBackendConfiguration {
            pipeline_cache_directory: None,
            ..Default::default()
        },
    ) else {
        return;
    };
    let ctx = backend.presentation_context();
    let dir = tempfile::tempdir().unwrap();
    let cache = NativePipelineCache::new(ctx.device(), Some(dir.path()), 4096).unwrap();
    cache.persist().unwrap();
    let path = cache.path.as_ref().unwrap();
    let first = load(path, &cache.properties, 4096).unwrap();
    assert!(valid_payload(&first, &cache.properties));
    assert!(first.len() <= 4096);
    drop(cache);
    let reloaded = NativePipelineCache::new(ctx.device(), Some(dir.path()), 4096).unwrap();
    assert_eq!(reloaded.data().unwrap(), first);
    // vkGetPipelineCacheData must not overallocate for an undersized budget.
    let small = NativePipelineCache::new(ctx.device(), None, 31).unwrap();
    assert!(small.data().unwrap().is_empty());
    small.persist().unwrap();
    // No-file configuration retains an in-memory cache without disk work.
    assert!(small.path.is_none());
}
