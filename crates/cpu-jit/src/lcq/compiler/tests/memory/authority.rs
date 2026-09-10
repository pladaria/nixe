//! Delivered faults use ExecutionMemory's real backing/protection policy.
use super::*;
use nixe_cpu::memory::{
    CpuMemory, DataAccessFaultReason, DirectFaultResolution, ExecutionMemory, MemoryAccess,
    MemoryAccessSize, MemoryValue, SyntheticMmio,
};
use nixe_cpu_direct_memory::{
    CapturedFault, FaultDisposition, InvocationOutcome, NativeInvocation, WorkerFaultContext,
};
use nixe_memory::{CanonicalRangeTranslator, DirectBackendPolicy};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct Writeback;
impl nixe_memory::VisibilityCoordinator for Writeback {
    fn make_device_visible(
        &self,
        _: nixe_memory::DeviceVisibilityRequest,
        _: &[u8],
    ) -> Result<(), nixe_memory::VisibilityCoordinatorError> {
        Ok(())
    }
    fn make_cpu_visible(
        &self,
        _: nixe_memory::CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, nixe_memory::VisibilityCoordinatorError> {
        Ok(vec![0x5a; 4096].into_boxed_slice())
    }
}

pub(super) struct Device(pub(super) Arc<AtomicUsize>);
impl SyntheticMmio for Device {
    fn read(&mut self, _: u64, _: MemoryAccess) -> Result<MemoryValue, Box<str>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(MemoryValue::U64(19))
    }
    fn write(&mut self, _: u64, _: MemoryAccess, _: MemoryValue) -> Result<(), Box<str>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

pub(super) struct Dispatch<'a> {
    pub(super) frame: *const libc::c_void,
    pub(super) lookup: crate::lifetime::FaultLookup<'a>,
    pub(super) memory: &'a ExecutionMemory,
    pub(super) resolution: Option<DirectFaultResolution>,
    pub(super) count: usize,
}

pub(super) fn prepare_cold(
    memory: &ExecutionMemory,
    state: &mut A64State,
) -> crate::lcq::fault::cold::Completion {
    let arena = memory.direct_address_space_view(SPACE).unwrap();
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    let mut reader = process.register().unwrap();
    let Request::Owner(claim) = reader.claim(key()).unwrap() else {
        panic!()
    };
    let compilation = Compilation::capture(claim, memory).unwrap();
    Compiler::for_arena(native_abi(), ARENA)
        .unwrap()
        .publish(compilation, &process, &cache, memory)
        .unwrap();
    let mut frame = NativeFrame::new(state, PollBudget::new(4096, 1000).unwrap());
    let _lease = memory.acquire_execution_lease();
    let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
    let entry = invocation.payload().preferred().unwrap().canonical.get();
    let (frame, lookup) = invocation.frame_and_faults();
    let mut dispatcher = Dispatch {
        frame: std::ptr::from_ref(frame).cast(),
        lookup,
        memory,
        resolution: None,
        count: 0,
    };
    let mut call = CapturedEntry {
        frame,
        arena: arena.base as *mut u8,
        result: None,
    };
    let mut worker = WorkerFaultContext::register().unwrap();
    let outcome = unsafe {
        worker.invoke_captured(
            arena,
            [
                call.frame.host_fp.saved_control,
                call.frame.host_fp.saved_status,
            ],
            dispatch,
            std::ptr::from_mut(&mut dispatcher).cast(),
            NativeInvocation {
                gateway: captured_entry,
                context: std::ptr::from_mut(&mut call).cast(),
                entry,
            },
        )
    }
    .unwrap();
    assert_eq!(outcome, InvocationOutcome::Escaped);
    assert_eq!(dispatcher.resolution, Some(DirectFaultResolution::Cold));
    let captured = worker.escaped_fault().unwrap();
    let fault = dispatcher.lookup.find(captured.native_pc()).unwrap();
    assert!(
        unsafe { crate::lcq::fault::cold::Completion::prepare(call.frame, &fault, None) }.is_err(),
        "preparation must not precede canonical reconstruction"
    );
    let reconstructed =
        unsafe { crate::lcq::fault::reconstruct(call.frame, &captured, &fault) }.unwrap();
    if reconstructed.completed_read.is_some() {
        assert!(
            unsafe { crate::lcq::fault::cold::Completion::prepare(call.frame, &fault, None) }
                .is_err()
        );
    }
    // The returned value outlives all code, frame, worker and memory-lease
    // owners here. Completion must neither dereference nor re-enter them.
    unsafe {
        crate::lcq::fault::cold::Completion::prepare(
            call.frame,
            &fault,
            reconstructed.completed_read,
        )
    }
    .unwrap()
}

pub(super) unsafe extern "C" fn dispatch(
    opaque: *mut libc::c_void,
    captured: *mut CapturedFault,
) -> FaultDisposition {
    // Match the eventual runtime's unwind boundary; no panic crosses native code.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dispatch = unsafe { &mut *opaque.cast::<Dispatch<'_>>() };
        let captured = unsafe { &*captured };
        let Some(fault) = dispatch.lookup.find(captured.native_pc()) else {
            return FaultDisposition::Fatal;
        };
        let resolved = unsafe {
            crate::lcq::fault::access::resolve(
                &*dispatch.frame.cast::<NativeFrame<'_>>(),
                captured,
                &fault,
                dispatch.memory.direct_address_space_view(SPACE).unwrap(),
                dispatch.memory,
            )
        };
        let (_, resolution) = resolved.unwrap();
        let disposition = if resolution == DirectFaultResolution::Retry {
            FaultDisposition::Retry
        } else {
            FaultDisposition::Escape
        };
        dispatch.count += 1;
        dispatch.resolution = Some(resolution);
        disposition
    }))
    .unwrap_or(FaultDisposition::Fatal)
}

#[test]
fn delivered_lcq_faults_use_memory_authority_for_retry_fault_and_cold() {
    // Stores start on clean, read-only tracked RAM. Cross-page stores must
    // repair both pages in one dispatch before the exact native retry.
    for (case, word, address) in [
        (0, 0xf900_0020, 0x2000),
        (1, 0xf900_0020, 0x2ffc),
        (2, 0xf940_0020, 0x3000),
        (3, 0xf900_0020, 0x3000),
        (4, 0xf940_0020, 0x3000),
        (5, 0xc8df_fc20, 0x2001), // LDAR misalignment on otherwise valid RAM
        (6, 0xf940_0020, ARENA as u64 + 8),
        (7, 0xf900_0020, 0x2ffc),  // second page denies write
        (8, 0xf900_0020, 0x3000),  // MMIO store: classification is not execution
        (9, 0xf940_0020, 0x2ffc),  // read crossing two GPU-newer RAM pages
        (10, 0xc8a2_7c20, 0x2000), // CAS: successful replacement on tracked RAM
        (11, 0xc8a0_7c22, 0x2000), // CAS: mismatch still checks write permission
        (12, super::atomic::rmw(3, 0, 3, 1, 0, 2), 0x2000), // LDADD on tracked RAM
        (13, super::atomic::rmw(3, 7, 3, 1, 31, 2), 0x2000), // unchanged LDUMIN still writes
        (14, super::casp::word(3, 1, 2, 4), 0x2000), // CASP W on tracked RAM
        (15, super::casp::word(3, 1, 0, 2), 0x2000), // CASP mismatch still writes
        (16, super::casp::word(3, 1, 2, 4) | (1 << 30), 0x2000), // CASP X tracked store
        (17, super::casp::word(3, 1, 0, 2) | (1 << 30), 0x2000), // CASP X mismatch still writes
    ] {
        let words = [0x9100_0421u32, word, 0xd420_0000]; // dirty X1
        let mut memory = ExecutionMemory::new();
        let device_calls = Arc::new(AtomicUsize::new(0));
        for page in 1..=3 {
            if page == 3 && case == 2 {
                continue;
            }
            let id = GuestPhysicalPageId::new(page);
            if page == 3 && matches!(case, 4 | 8) {
                assert!(memory.add_mmio_page(id, Device(device_calls.clone())));
            } else {
                assert!(memory.add_ram_page(id));
                if page == 1 {
                    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                    memory.initialize_ram(id, 0, &bytes).unwrap();
                }
            }
            let permissions = if page == 1 {
                MemoryPermissions::READ_EXECUTE
            } else if (page == 3 && matches!(case, 3 | 7))
                || (page == 2 && matches!(case, 11 | 13 | 15 | 17))
            {
                MemoryPermissions::READ
            } else {
                MemoryPermissions::READ_WRITE
            };
            assert!(memory.map_page(
                SPACE,
                GuestVirtualAddress::new(page * 4096),
                id,
                permissions
            ));
        }
        memory
            .bind_cpu_memory_backend(SPACE, ARENA as u64, DirectBackendPolicy::Required)
            .unwrap();
        let arena = memory.direct_address_space_view(SPACE).unwrap();
        if case == 9 {
            let range = memory
                .translate_canonical_range(
                    SPACE,
                    GuestVirtualAddress::new(0x2000),
                    8192,
                    MemoryPermissions::READ_WRITE,
                )
                .unwrap();
            let declaration = nixe_memory::DeviceAccessDeclaration::write(
                nixe_memory::NonCpuDeviceId::new(1),
                nixe_memory::DeviceVisibilityPoint::new(1),
                nixe_memory::DeviceVisibilityPoint::new(2),
            )
            .unwrap();
            let coordinator: Arc<dyn nixe_memory::VisibilityCoordinator> = Arc::new(Writeback);
            range
                .prepare_device_access(declaration, coordinator.clone())
                .unwrap();
            range
                .publish_device_write(declaration, coordinator)
                .unwrap();
        }
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
        let mut reader = process.register().unwrap();
        let Request::Owner(claim) = reader.claim(key()).unwrap() else {
            panic!()
        };
        let compilation = Compilation::capture(claim, &memory).unwrap();
        Compiler::for_arena(native_abi(), ARENA)
            .unwrap()
            .publish(compilation, &process, &cache, &memory)
            .unwrap();
        let mut state = A64State::default();
        state.set_pc(PC);
        state.general_register_storage_mut()[0] = 0x1234_5678_9abc_def0;
        state.general_register_storage_mut()[1] = address - 1;
        state.general_register_storage_mut()[4] = 0x9abc_def0;
        state.general_register_storage_mut()[5] = 0x1234_5678;
        if case == 16 {
            state.general_register_storage_mut()[4] = 0x1234_5678_9abc_def0;
            state.general_register_storage_mut()[5] = 0xfedc_ba98_7654_3210;
        }
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        let lease = memory.acquire_execution_lease();
        let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
        let entry = invocation.payload().preferred().unwrap().canonical.get();
        let (frame, lookup) = invocation.frame_and_faults();
        let mut dispatch_state = Dispatch {
            frame: std::ptr::from_ref(frame).cast(),
            lookup,
            memory: &memory,
            resolution: None,
            count: 0,
        };
        let mut call = CapturedEntry {
            frame,
            arena: arena.base as *mut u8,
            result: None,
        };
        let mut worker = WorkerFaultContext::register().unwrap();
        let outcome = unsafe {
            worker.invoke_captured(
                arena,
                [
                    call.frame.host_fp.saved_control,
                    call.frame.host_fp.saved_status,
                ],
                dispatch,
                std::ptr::from_mut(&mut dispatch_state).cast(),
                NativeInvocation {
                    gateway: captured_entry,
                    context: std::ptr::from_mut(&mut call).cast(),
                    entry,
                },
            )
        }
        .unwrap();
        assert_eq!(dispatch_state.count, 1, "case {case}");
        let resolution = dispatch_state.resolution.unwrap();
        if case <= 1 || matches!(case, 9 | 10 | 12 | 14 | 16) {
            assert_eq!(resolution, DirectFaultResolution::Retry);
            assert_eq!(outcome, InvocationOutcome::Returned);
            assert_eq!(
                call.result.unwrap().unwrap().reason,
                NativeExitReason::Architectural
            );
            assert_eq!(
                memory
                    .read(
                        SPACE,
                        GuestVirtualAddress::new(address),
                        MemoryAccess {
                            alignment: nixe_cpu::memory::MemoryAlignment::Unaligned,
                            ..MemoryAccess::normal(MemoryAccessSize::Doubleword)
                        }
                    )
                    .unwrap()
                    .value,
                MemoryValue::U64(if case == 9 {
                    0x5a5a_5a5a_5a5a_5a5a
                } else {
                    0x1234_5678_9abc_def0
                })
            );
            assert_eq!(
                memory.direct_protection_at(SPACE, GuestVirtualAddress::new(0x2000)),
                Some(if case == 9 {
                    DirectProtection::Read
                } else {
                    DirectProtection::ReadWrite
                })
            );
            if case == 1 || case == 9 {
                assert_eq!(
                    memory.direct_protection_at(SPACE, GuestVirtualAddress::new(0x3000)),
                    Some(if case == 9 {
                        DirectProtection::Read
                    } else {
                        DirectProtection::ReadWrite
                    })
                );
            }
        } else {
            assert_eq!(outcome, InvocationOutcome::Escaped);
            assert!(call.result.is_none());
            let captured = worker.escaped_fault().unwrap();
            let fault = dispatch_state.lookup.find(captured.native_pc()).unwrap();
            unsafe { crate::lcq::fault::reconstruct(call.frame, &captured, &fault) }.unwrap();
            if matches!(case, 4 | 8) {
                assert_eq!(resolution, DirectFaultResolution::Cold);
            } else {
                let DirectFaultResolution::Fault(fault) = resolution else {
                    panic!("case {case}: {resolution:?}");
                };
                assert_eq!(
                    fault.address.get(),
                    if case == 7 { 0x3000 } else { address }
                );
                assert_eq!(
                    fault.reason,
                    match case {
                        3 | 7 | 11 | 13 | 15 | 17 => DataAccessFaultReason::WritePermissionDenied,
                        5 => DataAccessFaultReason::Misaligned {
                            required_alignment: 8
                        },
                        _ => DataAccessFaultReason::Unmapped,
                    }
                );
            }
            // Classification must not dirty the first page when the second
            // page is invalid, or when an ordered access is misaligned.
            assert_eq!(
                memory.direct_protection_at(SPACE, GuestVirtualAddress::new(0x2000)),
                Some(DirectProtection::Read)
            );
        }
        drop(invocation);
        drop(lease);
        assert_eq!(device_calls.load(Ordering::Relaxed), 0);
        assert_eq!(state.general_register_storage_mut()[1], address);
        if case == 9 {
            assert_eq!(
                state.general_register_storage_mut()[0],
                0x5a5a_5a5a_5a5a_5a5a
            );
        } else if case > 1 && !matches!(case, 10 | 12 | 14 | 16) {
            assert_eq!(state.pc(), PC + 4);
        }
    }
}
