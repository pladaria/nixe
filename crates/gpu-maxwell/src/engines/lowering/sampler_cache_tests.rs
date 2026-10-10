//! Immutable sampler states survive binding changes and bounded cache retirement.
use super::*;
use nixe_gpu::{AddressMode, CacheMaintenanceOperation, FilterMode};

fn description(address: AddressMode) -> SamplerDescription {
    SamplerDescription::new(
        FilterMode::Linear,
        FilterMode::Linear,
        FilterMode::Nearest,
        [address; 3],
        0.0,
        15.0,
        1.0,
    )
    .unwrap()
}

fn table(cache: &mut MaxwellLoweringCache, sampler: SamplerId) -> DescriptorTableId {
    prepare_descriptors(
        vec![DescriptorTableBinding {
            binding: 0,
            resource: ResourceDependency::Sampler(sampler),
        }],
        vec![DescriptorKind::Sampler],
        cache,
        &mut Vec::new(),
    )
    .unwrap()[0]
}

fn use_table(table: DescriptorTableId) -> GpuOperation {
    GpuOperation::new(
        GpuCommand::CacheMaintenance(CacheMaintenanceOperation::InvalidateSamplerCaches),
        [],
        [ResourceDependency::DescriptorTable(table)],
        CapabilityRequirements::none(),
    )
}

#[test]
fn alternating_sampler_states_preserve_existing_descriptor_tables() {
    let mut cache = MaxwellLoweringCache::default();
    let states = [
        AddressMode::ClampToEdge,
        AddressMode::Repeat,
        AddressMode::MirroredRepeat,
    ];
    let mut creations = Vec::new();
    let ids = states
        .map(|address| prepare_sampler(description(address), &mut cache, &mut creations).unwrap());
    let tables = ids.map(|id| table(&mut cache, id));
    for frame in 0..100 {
        cache.revision = frame;
        for ((address, id), expected_table) in states.into_iter().zip(ids).zip(tables) {
            let selected =
                prepare_sampler(description(address), &mut cache, &mut creations).unwrap();
            assert_eq!(selected, id);
            assert_eq!(table(&mut cache, selected), expected_table);
        }
    }
    assert_eq!(creations.len(), 3);
    assert_eq!(cache.samplers.len(), 3);
    assert_eq!(cache.descriptors.len(), 3);
}

#[test]
fn sampler_identity_includes_every_effective_filter_lod_and_address_field() {
    let mut cache = MaxwellLoweringCache::default();
    let base = description(AddressMode::Repeat);
    let mut variants = vec![base];
    let mut changed = base;
    changed.min_filter = FilterMode::Nearest;
    variants.push(changed);
    changed = base;
    changed.mag_filter = FilterMode::Nearest;
    variants.push(changed);
    changed = base;
    changed.mip_filter = FilterMode::Linear;
    variants.push(changed);
    for axis in 0..3 {
        changed = base;
        changed.address_modes[axis] = AddressMode::ClampToEdge;
        variants.push(changed);
    }
    changed = base;
    changed.lod_min = 1.0;
    variants.push(changed);
    changed = base;
    changed.lod_max = 14.0;
    variants.push(changed);
    changed = base;
    changed.max_anisotropy = 2.0;
    variants.push(changed);
    let mut creations = Vec::new();
    let ids = variants
        .iter()
        .map(|&state| prepare_sampler(state, &mut cache, &mut creations).unwrap())
        .collect::<Vec<_>>();
    let unique = ids
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), variants.len());
    for (state, expected) in variants.into_iter().zip(ids) {
        assert_eq!(
            prepare_sampler(state, &mut cache, &mut creations).unwrap(),
            expected
        );
    }
    assert_eq!(creations.len(), unique.len());
}

#[test]
fn sampler_lru_preserves_current_tables_and_retires_only_unused_states() {
    let defaults = GpuCacheConfiguration::default();
    let configuration = GpuCacheConfiguration::new(
        defaults.shader_entries(),
        2,
        defaults.pipeline_variants_per_resource(),
        defaults.bind_groups_per_descriptor_table(),
        defaults.persistent_pipeline_cache_bytes(),
    )
    .unwrap();
    let mut cache = MaxwellLoweringCache::new(configuration);
    let mut creations = Vec::new();
    let old =
        prepare_sampler(description(AddressMode::Repeat), &mut cache, &mut creations).unwrap();
    let old_table = table(&mut cache, old);
    cache.revision = 1;
    let current = prepare_sampler(
        description(AddressMode::ClampToEdge),
        &mut cache,
        &mut creations,
    )
    .unwrap();
    let current_table = table(&mut cache, current);
    cache.revision = 2;
    let recent = prepare_sampler(
        description(AddressMode::MirroredRepeat),
        &mut cache,
        &mut creations,
    )
    .unwrap();
    let recent_table = table(&mut cache, recent);
    let operations = [
        use_table(old_table),
        use_table(current_table),
        use_table(recent_table),
    ];
    cache.prepared_draw = Some(PreparedDrawRecord {
        indexed: false,
        state: MaxwellThreeDState::default().draw_state_identity(),
        resources: Arc::new(()),
        shaders: Arc::new(()),
        operations: [
            use_table(old_table),
            use_table(old_table),
            use_table(old_table),
        ],
        dirty_images: Arc::from([]),
        sampled_aliases: Box::new([]),
    });
    let mut invalidations = Vec::new();
    trim_samplers(&mut cache, &operations, &mut invalidations);
    assert!(
        invalidations.is_empty(),
        "all three states are in use despite a budget of two"
    );
    assert_eq!(cache.samplers.len(), 3);
    assert!(cache.prepared_draw.is_some());
    trim_samplers(&mut cache, &operations[1..], &mut invalidations);
    assert_eq!(cache.samplers.len(), 2);
    assert_eq!(cache.descriptors.len(), 2);
    assert!(
        cache.prepared_draw.is_none(),
        "an evicted table cannot remain in a prepared draw"
    );
    assert!(invalidations.contains(&ResourceDependency::Sampler(old)));
    assert!(invalidations.contains(&ResourceDependency::DescriptorTable(old_table)));
    for dependency in [
        ResourceDependency::Sampler(current),
        ResourceDependency::Sampler(recent),
        ResourceDependency::DescriptorTable(current_table),
        ResourceDependency::DescriptorTable(recent_table),
    ] {
        assert!(!invalidations.contains(&dependency));
    }
    let recreated =
        prepare_sampler(description(AddressMode::Repeat), &mut cache, &mut creations).unwrap();
    assert_ne!(recreated, old);
}
