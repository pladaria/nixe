use super::*;
use crate::abi::RuntimeSystemOperation;
use crate::lcq::system::{CompletionError, RuntimeServices, complete_runtime};
use nixe_cpu::exclusive::ExclusiveMonitorState;
use nixe_cpu::execution::{ArchitecturalTimer, TimerSnapshot, VcpuEventState};
use nixe_cpu::memory::DataAccessFaultReason;
use nixe_memory::{
    CanonicalRangeTranslator, CpuVisibilityRequest, DeviceAccessDeclaration, DeviceVisibilityPoint,
    DeviceVisibilityRequest, MemoryInvalidationOrigin, NonCpuDeviceId, VisibilityCoordinator,
    VisibilityCoordinatorError,
};
use std::sync::atomic::AtomicUsize;

struct NoTimer;
impl ArchitecturalTimer for NoTimer {
    fn snapshot(&self) -> TimerSnapshot {
        panic!("DC does not read the timer")
    }
}

fn native_exit(
    process: &Arc<Lifetime>,
    memory: &ExecutionMemory,
    state: &mut A64State,
) -> RuntimeSystemOperation {
    let mut reader = process.register().unwrap();
    let mut frame = NativeFrame::new(state, PollBudget::new(4096, 1000).unwrap());
    let mut worker = nixe_cpu_direct_memory::WorkerFaultContext::register().unwrap();
    let mut monitor = ExclusiveMonitorState::default();
    let exit = unsafe {
        crate::lcq::invocation::run(
            &mut reader,
            &mut frame,
            memory,
            &mut worker,
            &mut monitor,
            key(0x2000),
        )
    }
    .unwrap()
    .unwrap();
    let crate::lcq::invocation::Exit::Native { guest, .. } = exit else {
        panic!()
    };
    assert_eq!(guest.pc.get(), 0x2000);
    let unit::EdgeKind::RuntimeSystem(operation) = guest.kind else {
        panic!()
    };
    operation
}

#[test]
fn native_data_cache_on_coherent_bytes_preserves_code_claims_and_log() {
    // Existing DC IVAC/CVAU/CIVAC system-exit encodings.
    for word in [0xd5087620_u32, 0xd50b7b20, 0xd50b7e20] {
        let (process, memory) = fixture();
        memory
            .overwrite_mapped_ram(SPACE, GuestVirtualAddress::new(0x2000), &word.to_le_bytes())
            .unwrap();
        let source = publish(&process, &memory, 0x2000);
        let target = publish(&process, &memory, 0x1000);
        let mut reader = process.register().unwrap();
        let compile::Request::Owner(claim) = reader.claim(key(0x1004)).unwrap() else {
            panic!()
        };
        let captured = Compilation::capture(claim, &memory).unwrap();
        let mut state = A64State::default();
        state.set_pc(0x2000);
        state.general_register_storage_mut()[0] = 0x1000;
        let operation = native_exit(&process, &memory, &mut state);
        let cursor = memory.invalidation_cursor();
        let mut monitor = ExclusiveMonitorState::default();
        assert_eq!(
            complete_runtime(
                operation,
                &mut state,
                &mut RuntimeServices {
                    address_space: SPACE,
                    memory: &memory,
                    timer: &NoTimer,
                    events: &VcpuEventState::default(),
                    exclusive: &mut monitor,
                }
            )
            .unwrap(),
            None
        );
        assert_eq!(state.pc(), 0x2004);
        assert_eq!(memory.invalidation_cursor(), cursor);
        captured.claim.validate().unwrap();
        assert!(process.snapshot(source).is_ok());
        assert!(process.snapshot(target).is_ok());
    }
}

struct Device {
    memory: std::sync::Weak<ExecutionMemory>,
    mode: u8,
    downloads: AtomicUsize,
}
impl VisibilityCoordinator for Device {
    fn make_device_visible(
        &self,
        _: DeviceVisibilityRequest,
        _: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        Ok(())
    }
    fn make_cpu_visible(
        &self,
        _: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        assert_eq!(self.downloads.fetch_add(1, Ordering::Relaxed), 0);
        let memory = self.memory.upgrade().unwrap();
        // Both operations acquire locks formerly held across this callback.
        memory
            .read_invalidations_since(memory.invalidation_cursor(), &mut Vec::new())
            .unwrap();
        assert!(
            memory
                .mapping_info(SPACE, GuestVirtualAddress::new(0x1000))
                .is_some()
        );
        if self.mode == 1 || self.mode == 2 {
            memory
                .resize_zeroed_mapping(
                    SPACE,
                    GuestVirtualAddress::new(0x1000),
                    4096,
                    0,
                    MemoryPermissions::READ_EXECUTE,
                    MemoryMappingPurpose::Normal,
                )
                .unwrap();
            if self.mode == 1 {
                memory
                    .resize_zeroed_mapping(
                        SPACE,
                        GuestVirtualAddress::new(0x1000),
                        0,
                        4096,
                        MemoryPermissions::READ_EXECUTE,
                        MemoryMappingPurpose::Normal,
                    )
                    .unwrap();
            }
        }
        if self.mode == 3 {
            return Err(VisibilityCoordinatorError::new(
                "DC device writeback rejected",
            ));
        }
        let mut bytes = vec![0; 4096];
        bytes[..4].copy_from_slice(&0xd4200120_u32.to_le_bytes());
        Ok(bytes.into_boxed_slice())
    }
}

#[test]
fn native_data_cache_writeback_is_unlocked_retranslates_and_preserves_precise_errors() {
    for word in [0xd5087620_u32, 0xd50b7b20, 0xd50b7e20] {
        for mode in 0..4 {
            let (process, memory) = fixture();
            memory
                .overwrite_mapped_ram(SPACE, GuestVirtualAddress::new(0x2000), &word.to_le_bytes())
                .unwrap();
            let source = publish(&process, &memory, 0x2000);
            let old = publish(&process, &memory, 0x1000);
            let memory = Arc::new(memory);
            let retained = memory
                .translate_canonical_range(
                    SPACE,
                    GuestVirtualAddress::new(0x1000),
                    4,
                    MemoryPermissions::READ,
                )
                .unwrap();
            let device = Arc::new(Device {
                memory: Arc::downgrade(&memory),
                mode,
                downloads: AtomicUsize::new(0),
            });
            let write = DeviceAccessDeclaration::write(
                NonCpuDeviceId::new(1),
                DeviceVisibilityPoint::new(1),
                DeviceVisibilityPoint::new(2),
            )
            .unwrap();
            retained
                .prepare_device_access(write, device.clone())
                .unwrap();
            retained
                .publish_device_write(write, device.clone())
                .unwrap();
            assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
            let cursor = memory.invalidation_cursor();
            let mut state = A64State::default();
            state.set_pc(0x2000);
            state.general_register_storage_mut()[0] = 0x1000;
            let operation = native_exit(&process, &memory, &mut state);
            let before = state.clone();
            let mut monitor = ExclusiveMonitorState::default();
            let result = complete_runtime(
                operation,
                &mut state,
                &mut RuntimeServices {
                    address_space: SPACE,
                    memory: memory.as_ref(),
                    timer: &NoTimer,
                    events: &VcpuEventState::default(),
                    exclusive: &mut monitor,
                },
            );
            if mode < 2 {
                assert_eq!(result.unwrap(), None);
                assert_eq!(state.pc(), 0x2004);
                let new = publish(&process, &memory, 0x1000);
                assert_eq!(
                    process.snapshot(new).unwrap().instructions[0].bits,
                    if mode == 0 { 0xd4200120 } else { 0 }
                );
                let mut old_bytes = [0; 4];
                retained.read(0, &mut old_bytes).unwrap();
                assert_eq!(u32::from_le_bytes(old_bytes), 0xd4200120);
            } else {
                let Err(CompletionError::Memory(fault)) = result else {
                    panic!()
                };
                if mode == 2 {
                    assert_eq!(fault.reason, DataAccessFaultReason::Unmapped);
                } else {
                    assert!(
                        matches!(fault.reason, DataAccessFaultReason::HostBacking(detail)
                    if detail.contains("DC device writeback rejected"))
                    );
                }
                assert_eq!(state, before);
            }
            assert!(process.snapshot(source).is_ok());
            assert_eq!(process.lock().phase, Phase::Open);
            assert_eq!(device.downloads.load(Ordering::Relaxed), 1);
            let mut records = Vec::new();
            memory
                .read_invalidations_since(cursor, &mut records)
                .unwrap();
            assert!(
                records
                    .iter()
                    .all(|record| record.origin == MemoryInvalidationOrigin::Mapping)
            );
            if mode == 0 || mode == 3 {
                assert!(records.is_empty());
            }
        }
    }
}
