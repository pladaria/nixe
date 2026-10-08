use super::*;
use nixe_cpu::{
    memory::{MemoryPermissions, SyntheticMemory},
    platform::TargetPlatform,
    profile::ProcessCpuContext,
};
use nixe_memory::{AddressSpaceId, GuestPhysicalPageId};
fn modules(base: u64) -> Vec<WarmupModule> {
    vec![WarmupModule {
        content: [7; 32],
        base,
        extent: 4096,
    }]
}
fn cpu() -> ProcessCpuContext {
    ProcessCpuContext::new(TargetPlatform::Switch1, AddressSpaceId::new(1))
}
fn record() -> Record {
    Record {
        module: 0,
        offset: 0,
        words: vec![0xd503201f, 0xd65f03c0],
    }
}
#[test]
fn profile_round_trip_relocates_offsets_without_process_pointers() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("test.warmup");
    write(&path, &[3; 32], &[record()]).unwrap();
    assert_eq!(
        read(&path, &[3; 32], &modules(0x1000)).unwrap(),
        vec![record()]
    );
    assert_eq!(
        read(&path, &[3; 32], &modules(0x90000000)).unwrap(),
        vec![record()]
    );
    assert!(read(&path, &[4; 32], &modules(0x1000)).is_err());
    assert!(read(&path, &[3; 32], &[]).is_err());
}
#[test]
fn profile_rejects_truncation_duplicates_invalid_offsets_and_boundaries() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("test.warmup");
    write(&path, &[3; 32], &[record()]).unwrap();
    let data = std::fs::read(&path).unwrap();
    for len in 0..data.len() {
        std::fs::write(&path, &data[..len]).unwrap();
        assert!(read(&path, &[3; 32], &modules(0)).is_err(), "{len}");
    }
    for bad in [
        Record {
            offset: 1,
            ..record()
        },
        Record {
            offset: 4092,
            ..record()
        },
        Record {
            module: 1,
            ..record()
        },
        Record {
            words: vec![],
            ..record()
        },
        Record {
            words: vec![0; 513],
            ..record()
        },
    ] {
        write(&path, &[3; 32], &[bad]).unwrap();
        assert!(read(&path, &[3; 32], &modules(0)).is_err());
    }
    write(&path, &[3; 32], &[record(), record()]).unwrap();
    assert!(read(&path, &[3; 32], &modules(0)).is_err());
}
#[test]
fn profile_identity_includes_content_context_host_policy_but_not_placement() {
    let directory = tempfile::tempdir().unwrap();
    let load = |cpu, modules| {
        Warmup::load(
            cpu,
            WarmupConfig {
                directory: directory.path().to_owned(),
                modules,
            },
        )
        .unwrap()
    };
    let a = load(cpu(), modules(0x1000));
    assert_eq!(a.path, load(cpu(), modules(0x90000000)).path);
    let mut other = modules(0x1000);
    other[0].content[3] ^= 1;
    assert_ne!(a.path, load(cpu(), other).path);
    assert_ne!(
        a.path,
        load(
            ProcessCpuContext::new(TargetPlatform::Switch2, AddressSpaceId::new(1)),
            modules(0x1000)
        )
        .path
    );
    assert_ne!(
        a.path,
        load(
            ProcessCpuContext::new(TargetPlatform::Switch1, AddressSpaceId::new(2)),
            modules(0x1000)
        )
        .path
    );
}
#[test]
fn cached_words_never_override_live_bytes_or_execute_permissions() {
    let mut memory = SyntheticMemory::new();
    let page = GuestPhysicalPageId::new(1);
    let space = AddressSpaceId::new(1);
    assert!(memory.add_ram_page(page));
    assert!(memory.initialize_ram(
        page,
        0,
        &[0xd503201fu32.to_le_bytes(), 0xd65f03c0u32.to_le_bytes()].concat()
    ));
    assert!(memory.map_page(
        space,
        GuestVirtualAddress::new(0),
        page,
        MemoryPermissions::READ_EXECUTE
    ));
    let key = BlockKey::new(
        cpu(),
        GuestVirtualAddress::new(0),
        FpSpecialization::Dynamic,
    )
    .unwrap();
    let fragment = Fragment::capture(&memory, key).unwrap();
    assert!(matches_words(&fragment, &record().words));
    assert!(!matches_words(&fragment, &[0xd503201f]));
    assert!(memory.initialize_ram(page, 0, &0x14000000u32.to_le_bytes()));
    assert!(!matches_words(
        &Fragment::capture(&memory, key).unwrap(),
        &record().words
    ));
    // Execute permission is consumed by fresh capture; stored words are hints.
    let mut memory = SyntheticMemory::new();
    assert!(memory.add_ram_page(page));
    assert!(memory.map_page(
        space,
        GuestVirtualAddress::new(0),
        page,
        MemoryPermissions::READ
    ));
    assert!(!matches_words(
        &Fragment::capture(&memory, key).unwrap(),
        &record().words
    ));
}

fn execution_memory(
    base: u64,
    words: &[u32],
    permissions: MemoryPermissions,
) -> Arc<ExecutionMemory> {
    let mut memory = ExecutionMemory::new();
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    memory.initialize_ram(page, 0, &bytes).unwrap();
    assert!(memory.map_page(
        AddressSpaceId::new(1),
        GuestVirtualAddress::new(base),
        page,
        permissions
    ));
    memory
        .bind_cpu_memory_backend(
            AddressSpaceId::new(1),
            0x10000,
            nixe_memory::DirectBackendPolicy::Required,
        )
        .unwrap();
    Arc::new(memory)
}
#[test]
fn warmup_uses_normal_publication_and_invalidation_at_a_new_module_base() {
    let directory = tempfile::tempdir().unwrap();
    let profile = Warmup::load(
        cpu(),
        WarmupConfig {
            directory: directory.path().to_owned(),
            modules: modules(0x1000),
        },
    )
    .unwrap();
    let words = vec![0xd503201f, 0xd4200021];
    profile.observe(Record {
        words: words.clone(),
        ..record()
    });
    profile.save();
    let profile = Warmup::load(
        cpu(),
        WarmupConfig {
            directory: directory.path().to_owned(),
            modules: modules(0x3000),
        },
    )
    .unwrap();
    let memory = execution_memory(0x3000, &words, MemoryPermissions::READ_EXECUTE);
    let lifetime = Arc::new(Lifetime::new(crate::executable::Cache::new().unwrap()).unwrap());
    memory.set_mutation_observer(lifetime.clone()).unwrap();
    profile.start(true);
    Task {
        profile,
        memory: memory.clone(),
        cpu: cpu(),
        arena_size: 0x10000,
    }
    .run(&lifetime, || Ok(()))
    .unwrap();
    let key = BlockKey::new(
        cpu(),
        GuestVirtualAddress::new(0x3000),
        FpSpecialization::Dynamic,
    )
    .unwrap();
    let mut reader = lifetime.register().unwrap();
    assert!(matches!(reader.claim(key).unwrap(), Request::Ready));
    memory
        .overwrite_mapped_ram(AddressSpaceId::new(1), key.pc, &0xd4200041u32.to_le_bytes())
        .unwrap();
    assert!(matches!(reader.claim(key).unwrap(), Request::Owner(_)));
}
#[test]
fn warmup_abandons_stale_unexecutable_and_cancelled_hints() {
    for case in 0..3 {
        let directory = tempfile::tempdir().unwrap();
        let profile = Warmup::load(
            cpu(),
            WarmupConfig {
                directory: directory.path().to_owned(),
                modules: modules(0x1000),
            },
        )
        .unwrap();
        profile.observe(record());
        let memory = execution_memory(
            0x1000,
            &[0xd4200021],
            if case == 1 {
                MemoryPermissions::READ
            } else {
                MemoryPermissions::READ_EXECUTE
            },
        );
        let lifetime = Arc::new(Lifetime::new(crate::executable::Cache::new().unwrap()).unwrap());
        memory.set_mutation_observer(lifetime.clone()).unwrap();
        profile.start(true);
        let cancel = profile.clone();
        Task {
            profile,
            memory,
            cpu: cpu(),
            arena_size: 0x10000,
        }
        .run(&lifetime, || {
            if case == 2 {
                cancel.cancel();
            }
            Ok(())
        })
        .unwrap();
        let key = BlockKey::new(
            cpu(),
            GuestVirtualAddress::new(0x1000),
            FpSpecialization::Dynamic,
        )
        .unwrap();
        let mut reader = lifetime.register().unwrap();
        assert!(matches!(reader.claim(key).unwrap(), Request::Owner(_)));
    }
}

#[test]
fn eviction_is_bounded_and_preserves_unrelated_files() {
    let directory = tempfile::tempdir().unwrap();
    let current = directory.path().join(format!("{:064x}.warmup", 99));
    for n in 0..10 {
        std::fs::write(directory.path().join(format!("{n:064x}.warmup")), []).unwrap();
    }
    std::fs::write(&current, []).unwrap();
    let unrelated = directory.path().join("notes.warmup");
    std::fs::write(&unrelated, b"preserve").unwrap();
    evict_profiles(&current, 8).unwrap();
    assert!(current.exists());
    assert_eq!(std::fs::read(&unrelated).unwrap(), b"preserve");
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 9);
}

#[test]
fn terminal_shutdown_cancels_warmup_without_fabricating_a_worker_failure() {
    let directory = tempfile::tempdir().unwrap();
    let profile = Warmup::load(
        cpu(),
        WarmupConfig {
            directory: directory.path().to_owned(),
            modules: modules(0x1000),
        },
    )
    .unwrap();
    profile.observe(record());
    profile.start(true);
    let memory = execution_memory(0x1000, &record().words, MemoryPermissions::READ_EXECUTE);
    let lifetime = Arc::new(Lifetime::new(crate::executable::Cache::new().unwrap()).unwrap());
    memory.set_mutation_observer(lifetime.clone()).unwrap();
    lifetime.request_shutdown().unwrap();
    Task {
        profile,
        memory,
        cpu: cpu(),
        arena_size: 0x10000,
    }
    .run(&lifetime, || Ok(()))
    .unwrap();
    assert!(lifetime.background_failure().is_none());
}
