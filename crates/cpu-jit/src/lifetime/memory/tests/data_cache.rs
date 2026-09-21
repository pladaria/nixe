use super::*;
use crate::lcq::invocation::{Exit, MemoryExit};
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

struct UnreadableDevice;
impl nixe_cpu::memory::SyntheticMmio for UnreadableDevice {
    fn read(
        &mut self,
        _: u64,
        _: nixe_cpu::memory::MemoryAccess,
    ) -> Result<nixe_cpu::memory::MemoryValue, Box<str>> {
        panic!("CIVAC must not read MMIO");
    }
    fn write(
        &mut self,
        _: u64,
        _: nixe_cpu::memory::MemoryAccess,
        _: nixe_cpu::memory::MemoryValue,
    ) -> Result<(), Box<str>> {
        panic!("CIVAC must not write MMIO");
    }
}

#[test]
fn civac_probe_mmio_and_zero_register_preserve_the_original_fault() {
    for word in [0xd50b7e20_u32, 0xd50b7e3f] {
        let (process, mut memory) = fixture();
        assert!(memory.add_mmio_page(GuestPhysicalPageId::new(3), UnreadableDevice));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            GuestPhysicalPageId::new(3),
            MemoryPermissions::READ_WRITE
        ));
        memory
            .overwrite_mapped_ram(SPACE, GuestVirtualAddress::new(0x2000), &word.to_le_bytes())
            .unwrap();
        publish(&process, &memory, 0x2000);
        let mut state = A64State::default();
        state.set_pc(0x2000);
        state.general_register_storage_mut()[0] = 0x3000;
        *state.stack_pointer_storage_mut() = 0x1000; // XZR must not use mapped SP.
        let before = state.clone();
        let exit = native_exit(&process, &memory, &mut state);
        assert_eq!(state, before);
        let fault = complete_cache(exit, &mut state, &memory).unwrap_err();
        if word & 31 == 31 {
            assert_eq!(fault.address.get(), 0);
            assert_eq!(fault.reason, DataAccessFaultReason::Unmapped);
        } else {
            assert_eq!(fault.address.get(), 0x3000);
            assert!(matches!(fault.reason, DataAccessFaultReason::Device(_)));
        }
        assert_eq!(state, before);
    }
}
impl ArchitecturalTimer for NoTimer {
    fn snapshot(&self) -> TimerSnapshot {
        panic!("DC does not read the timer")
    }
}

fn native_exit(process: &Arc<Lifetime>, memory: &ExecutionMemory, state: &mut A64State) -> Exit {
    let mut reader = process.register().unwrap();
    let mut frame = NativeFrame::new(state, PollBudget::new(4096, 1000).unwrap());
    let mut worker = nixe_cpu_direct_memory::WorkerFaultContext::register().unwrap();
    let mut monitor = ExclusiveMonitorState::default();
    unsafe {
        crate::lcq::invocation::run(
            &mut crate::sampling::Samples::new(),
            &mut reader,
            &mut frame,
            memory,
            &mut worker,
            &mut monitor,
            key(0x2000),
        )
    }
    .unwrap()
    .unwrap()
}

fn complete_cache(
    exit: Exit,
    state: &mut A64State,
    memory: &ExecutionMemory,
) -> Result<(), nixe_cpu::memory::DataAccessFault> {
    let mut monitor = ExclusiveMonitorState::default();
    match exit {
        Exit::Native { guest, .. } => {
            let unit::EdgeKind::RuntimeSystem(operation) = guest.kind else {
                panic!()
            };
            match complete_runtime(
                operation,
                state,
                &mut RuntimeServices {
                    address_space: SPACE,
                    memory,
                    timer: &NoTimer,
                    events: &VcpuEventState::default(),
                    exclusive: &mut monitor,
                },
            ) {
                Ok(None) => Ok(()),
                Err(CompletionError::Memory(fault)) => Err(fault),
                other => panic!("{other:?}"),
            }
        }
        Exit::Memory {
            instruction,
            outcome,
            ..
        } => {
            assert!(matches!(outcome, MemoryExit::CacheCleanInvalidate { .. }));
            match outcome
                .complete(instruction, state, memory, &mut monitor, 0)
                .unwrap()
            {
                None => Ok(()),
                Some(nixe_cpu::execution::CpuExit::DataFault { fault, .. }) => Err(fault),
                other => panic!("{other:?}"),
            }
        }
    }
}

#[test]
fn native_data_cache_on_coherent_bytes_preserves_code_claims_and_log() {
    // CIVAC stays native; the other data maintenance operations stay cold.
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
        let cursor = memory.invalidation_cursor();
        let exit = native_exit(&process, &memory, &mut state);
        if word == 0xd50b7e20 {
            assert!(matches!(exit, Exit::Native { guest, .. }
                if guest.kind == unit::EdgeKind::Breakpoint(0) && guest.pc.get() == 0x2004));
        } else {
            complete_cache(exit, &mut state, &memory).unwrap();
        }
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
            let exit = native_exit(&process, &memory, &mut state);
            let before = state.clone();
            assert_eq!(device.downloads.load(Ordering::Relaxed), 0);
            let result = complete_cache(exit, &mut state, &memory);
            if mode < 2 {
                result.unwrap();
                assert_eq!(state.pc(), 0x2004);
                let new = publish(&process, &memory, 0x1000);
                assert_eq!(
                    process
                        .snapshot(new)
                        .unwrap()
                        .instructions
                        .get(0)
                        .unwrap()
                        .bits,
                    if mode == 0 { 0xd4200120 } else { 0 }
                );
                let mut old_bytes = [0; 4];
                retained.read(0, &mut old_bytes).unwrap();
                assert_eq!(u32::from_le_bytes(old_bytes), 0xd4200120);
                if word == 0xd50b7e20 && mode == 0 {
                    // The completed download restored the direct alias. The
                    // same instruction now succeeds natively without a second
                    // callback or canonical cache exit.
                    state.set_pc(0x2000);
                    let exit = native_exit(&process, &memory, &mut state);
                    assert!(matches!(exit, Exit::Native { guest, .. }
                        if guest.kind == unit::EdgeKind::Breakpoint(0)));
                    assert_eq!(state.pc(), 0x2004);
                }
            } else {
                let Err(fault) = result else { panic!() };
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
