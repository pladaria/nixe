use std::sync::Arc;

use nixe_cpu::memory::{ExecutionMemory, ProcessMemory};
use nixe_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use nixe_cpu_engine::{
    DomainMemoryBinding, DomainRequest, EngineDomain, EngineDomainId, EngineProvider,
    run_provider_conformance,
};
use nixe_cpu_engine_testkit::FakeNceProvider;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress, MemoryPermissions};

#[test]
fn fake_nce_passes_portable_engine_conformance() {
    let provider: Arc<dyn EngineProvider> = Arc::new(FakeNceProvider::new());
    let report = run_provider_conformance(provider).unwrap();
    assert!(!report.passed.is_empty());
}

#[test]
fn fake_nce_uses_the_generic_memory_and_lifecycle_contract() {
    let provider = FakeNceProvider::new();
    let metrics = provider.metrics();
    let space = AddressSpaceId::new(9);
    let cpu = ProcessCpuContext::new(GuestCpuProfile::switch_1(), space);
    let memory = ExecutionMemory::new();
    memory
        .resize_zeroed_mapping(
            space,
            GuestVirtualAddress::new(0x4000),
            0,
            0x1000,
            MemoryPermissions::READ_WRITE,
            nixe_cpu::memory::MemoryMappingPurpose::Normal,
        )
        .unwrap();
    let binding = DomainMemoryBinding {
        address_space: space,
        end_exclusive: GuestVirtualAddress::new(1_u64 << 39),
        memory: &memory,
        invalidation_generation: memory.mapping_epoch().get(),
        dirty_generation: memory.content_mutation_epoch().get(),
    };
    let mut domain = provider.create_nce_domain(DomainRequest {
        domain: EngineDomainId::new(7),
        cpu,
    });
    domain.bind_memory(binding).unwrap();
    assert_eq!(domain.mirrored_binding_count(), 1);
    domain.activate().unwrap();
    let synchronization = domain.synchronize_memory(binding).unwrap();
    assert_eq!(synchronization.address_space, space);
    domain.shutdown().unwrap();
    domain.shutdown().unwrap();
    assert!(metrics.mapping_notifications() > 0);
    assert_eq!(metrics.reconciliations(), 1);
    assert_eq!(metrics.teardowns(), 1);
}
