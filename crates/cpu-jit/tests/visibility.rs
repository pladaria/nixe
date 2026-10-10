use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState};
use nixe_cpu::memory::{
    CpuMemory, DataAccessFaultReason, ExecutionMemory, MemoryAccess, MemoryAccessSize,
    MemoryPermissions, MemoryValue, ProcessMemory, SyntheticMmio,
};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State};
use nixe_cpu_jit::{JitProcess, JitRunRequest, JitThread, ThreadState};
use nixe_memory::{
    AddressSpaceId, CanonicalRangeTranslator, CpuVisibilityRequest, DeviceAccessDeclaration,
    DeviceVisibilityPoint, DeviceVisibilityRequest, GuestPhysicalPageId, GuestVirtualAddress,
    NonCpuDeviceId, VisibilityCoordinator, VisibilityCoordinatorError,
};
use std::sync::{
    Arc, Weak,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::time::Duration;

const SPACE: AddressSpaceId = AddressSpaceId::new(7);
const CODE: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);
const DATA: GuestVirtualAddress = GuestVirtualAddress::new(0x3000);

struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}

struct Writeback {
    memory: Weak<ExecutionMemory>,
    start: GuestVirtualAddress,
    bytes: Box<[u8]>,
    readonly: bool,
    calls: AtomicUsize,
}

impl VisibilityCoordinator for Writeback {
    fn cache_cpu_page(
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
        self.calls.fetch_add(1, Ordering::Relaxed);
        // This is a real gate-taking mutation, not a lock probe. A native
        // callback retaining its shared lease would deadlock here.
        self.memory
            .upgrade()
            .unwrap()
            .set_permissions(
                SPACE,
                self.start,
                0x1000,
                if self.start == CODE {
                    MemoryPermissions::READ_EXECUTE
                } else if self.readonly {
                    MemoryPermissions::READ
                } else {
                    MemoryPermissions::READ_WRITE
                },
            )
            .map_err(|error| VisibilityCoordinatorError::new(format!("{error:?}")))?;
        Ok(self.bytes.clone())
    }
}

fn x(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}

fn fixture(code: &[u32], device: Option<CountingDevice>) -> Arc<ExecutionMemory> {
    let mut memory = ExecutionMemory::new();
    for (id, address, permissions) in [
        (1, CODE, MemoryPermissions::READ_EXECUTE),
        (3, DATA, MemoryPermissions::READ_WRITE),
    ] {
        assert!(memory.add_ram_page(GuestPhysicalPageId::new(id)));
        if id == 1 {
            let bytes = code
                .iter()
                .copied()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            memory
                .initialize_ram(GuestPhysicalPageId::new(id), 0, &bytes)
                .unwrap();
        }
        assert!(memory.map_page(SPACE, address, GuestPhysicalPageId::new(id), permissions));
    }
    if let Some(device) = device {
        assert!(memory.add_mmio_page(GuestPhysicalPageId::new(2), device));
    } else {
        assert!(memory.add_ram_page(GuestPhysicalPageId::new(2)));
    }
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0x2000),
        GuestPhysicalPageId::new(2),
        MemoryPermissions::READ_WRITE
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, nixe_memory::DirectBackendPolicy::Required)
        .unwrap();
    Arc::new(memory)
}

fn publish(
    memory: &Arc<ExecutionMemory>,
    start: GuestVirtualAddress,
    bytes: Box<[u8]>,
    readonly: bool,
) -> Arc<Writeback> {
    let range = memory
        .translate_canonical_range(SPACE, start, 0x1000, MemoryPermissions::READ)
        .unwrap();
    let coordinator = Arc::new(Writeback {
        memory: Arc::downgrade(memory),
        start,
        bytes,
        readonly,
        calls: AtomicUsize::new(0),
    });
    let write = DeviceAccessDeclaration::write(
        NonCpuDeviceId::new(1),
        DeviceVisibilityPoint::new(1),
        DeviceVisibilityPoint::new(2),
    )
    .unwrap();
    nixe_memory::CanonicalBackingRange::prepare_resident_device_accesses(
        [(&range, write)],
        coordinator.clone(),
    )
    .unwrap();
    nixe_memory::CanonicalBackingRange::publish_device_writes(
        [(&range, write)],
        coordinator.clone(),
    )
    .unwrap();
    coordinator
}

fn execute(
    memory: &Arc<ExecutionMemory>,
    state: &mut ThreadState,
) -> nixe_cpu::execution::ExecutionReport {
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE);
    let mut jit = JitThread::new(Arc::new(
        JitProcess::new(cpu, memory.clone(), 19_200_000).unwrap(),
    ))
    .unwrap();
    jit.run_slice(
        &mut nixe_cpu_direct_memory::NativeWorker::default(),
        JitRunRequest {
            state,
            instruction_budget: 100,
            timer: &Timer,
            events: &VcpuEventState::default(),
            capture_context: true,
        },
    )
    .unwrap()
}

fn bounded(task: impl FnOnce() + Send + 'static) {
    let (done, result) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        task();
        done.send(()).unwrap();
    });
    result
        .recv_timeout(Duration::from_secs(5))
        .expect("CPU visibility must not wait while retaining execution admission");
    worker.join().unwrap();
}

#[test]
fn gpu_readback_leaves_native_execution_and_revalidates_pair_writeback() {
    bounded(|| {
        // LDP X2,X3,[X1],#16; SVC. Only the second pair access needs readback.
        let memory = fixture(&[0xa8c1_0c22, 0xd400_0001], None);
        memory
            .write(
                SPACE,
                GuestVirtualAddress::new(0x2ff8),
                MemoryAccess::normal(MemoryAccessSize::Doubleword),
                MemoryValue::U64(17),
            )
            .unwrap();
        let coordinator = publish(&memory, DATA, vec![0x22; 0x1000].into_boxed_slice(), false);
        let mut saved = A64State::default();
        saved.set_pc(CODE.get());
        saved.write_x(x(1), 0x2ff8);
        let mut state: ThreadState = saved.into();
        assert!(matches!(
            execute(&memory, &mut state).stop,
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(1)), 0x3008);
        assert_eq!(state.read_x(x(2)), 17);
        assert_eq!(state.read_x(x(3)), 0x2222_2222_2222_2222);
        assert_eq!(coordinator.calls.load(Ordering::Relaxed), 1);
    });
}

#[test]
fn permissions_changed_during_readback_fault_before_the_store() {
    bounded(|| {
        let memory = fixture(&[0xf900_0020, 0xd400_0001], None); // STR X0,[X1]
        let coordinator = publish(&memory, DATA, vec![0x22; 0x1000].into_boxed_slice(), true);
        let mut saved = A64State::default();
        saved.set_pc(CODE.get());
        saved.write_x(x(0), 99);
        saved.write_x(x(1), DATA.get());
        let mut state: ThreadState = saved.into();
        let report = execute(&memory, &mut state);
        assert!(
            matches!(report.stop, CpuExit::DataFault { fault, .. } if fault.reason == DataAccessFaultReason::WritePermissionDenied)
        );
        assert_eq!(state.pc(), CODE.get());
        assert_eq!(
            memory
                .read(
                    SPACE,
                    DATA,
                    MemoryAccess::normal(MemoryAccessSize::Doubleword)
                )
                .unwrap()
                .value
                .bits(),
            0x2222_2222_2222_2222
        );
        assert_eq!(coordinator.calls.load(Ordering::Relaxed), 1);
    });
}

struct CountingDevice(Arc<AtomicUsize>);
impl SyntheticMmio for CountingDevice {
    fn read(&mut self, _: u64, access: MemoryAccess) -> Result<MemoryValue, Box<str>> {
        Ok(MemoryValue::from_bits(access.size, 0))
    }
    fn write(&mut self, _: u64, _: MemoryAccess, value: MemoryValue) -> Result<(), Box<str>> {
        assert_eq!(value.bits(), 17);
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[test]
fn pair_device_write_is_not_replayed_when_the_second_access_demands_gpu_data() {
    bounded(|| {
        let writes = Arc::new(AtomicUsize::new(0));
        let memory = fixture(
            &[0xa900_0c22, 0xd400_0001],
            Some(CountingDevice(writes.clone())),
        ); // STP X2,X3,[X1]
        let coordinator = publish(&memory, DATA, vec![0x22; 0x1000].into_boxed_slice(), false);
        let mut saved = A64State::default();
        saved.set_pc(CODE.get());
        saved.write_x(x(1), 0x2ff8);
        saved.write_x(x(2), 17);
        saved.write_x(x(3), 34);
        let mut state: ThreadState = saved.into();
        assert!(matches!(
            execute(&memory, &mut state).stop,
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(writes.load(Ordering::Relaxed), 1);
        assert_eq!(coordinator.calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            memory
                .read(
                    SPACE,
                    DATA,
                    MemoryAccess::normal(MemoryAccessSize::Doubleword)
                )
                .unwrap()
                .value
                .bits(),
            34
        );
    });
}

#[test]
fn gpu_owned_code_is_resolved_at_execution_without_a_translation_lease() {
    bounded(|| {
        let memory = fixture(&[0xd503_201f, 0xd400_0001], None);
        let mut bytes = vec![0; 0x1000];
        bytes[..8].copy_from_slice(
            &[0xd280_0540_u32, 0xd400_0001]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        ); // MOV X0,#42; SVC
        let coordinator = publish(&memory, CODE, bytes.into_boxed_slice(), false);
        let mut saved = A64State::default();
        saved.set_pc(CODE.get());
        let mut state: ThreadState = saved.into();
        assert!(matches!(
            execute(&memory, &mut state).stop,
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(0)), 42);
        assert_eq!(coordinator.calls.load(Ordering::Relaxed), 1);
    });
}

#[test]
fn data_cache_clean_resolves_gpu_visibility_outside_native_execution() {
    bounded(|| {
        // DC CVAC,X1; LDR X2,[X1]; SVC. The coordinator takes an exclusive
        // mapping lease, so attempting readback in the native callback deadlocks.
        let memory = fixture(&[0xd50b_7a21, 0xf940_0022, 0xd400_0001], None);
        let coordinator = publish(&memory, DATA, vec![0x22; 0x1000].into_boxed_slice(), false);
        let mut saved = A64State::default();
        saved.set_pc(CODE.get());
        saved.write_x(x(1), DATA.get());
        let mut state: ThreadState = saved.into();
        assert!(matches!(
            execute(&memory, &mut state).stop,
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(2)), 0x2222_2222_2222_2222);
        assert_eq!(coordinator.calls.load(Ordering::Relaxed), 1);
    });
}

#[test]
fn data_cache_clean_faults_at_the_unmapped_instruction_without_advancing() {
    bounded(|| {
        let memory = fixture(&[0xd50b_7a21, 0xd280_0540, 0xd400_0001], None);
        let mut saved = A64State::default();
        saved.set_pc(CODE.get());
        saved.write_x(x(0), 17);
        saved.write_x(x(1), 0x5000);
        let mut state: ThreadState = saved.into();
        let report = execute(&memory, &mut state);
        assert!(matches!(
            report.stop,
            CpuExit::DataFault { fault, .. } if fault.reason == DataAccessFaultReason::Unmapped
        ));
        assert_eq!(state.pc(), CODE.get());
        assert_eq!(state.read_x(x(0)), 17);
    });
}

#[test]
fn exclusives_retry_visibility_without_committing_a_failed_transaction() {
    bounded(|| {
        let memory = fixture(
            &[
                0xc85f_7c23,
                0x9100_0463,
                0xc804_7c23,
                0xf940_0025,
                0xd400_0001,
            ],
            None,
        );
        let coordinator = publish(&memory, DATA, vec![0x22; 0x1000].into_boxed_slice(), false);
        let mut saved = A64State::default();
        saved.set_pc(CODE.get());
        saved.write_x(x(1), DATA.get());
        let mut state: ThreadState = saved.into();
        assert!(matches!(
            execute(&memory, &mut state).stop,
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(4)), 0);
        assert_eq!(state.read_x(x(5)), 0x2222_2222_2222_2223);
        assert_eq!(coordinator.calls.load(Ordering::Relaxed), 1);
    });
}

#[test]
fn repeated_ordered_loads_from_readonly_ram_complete_in_one_native_invocation() {
    bounded(|| {
        // LDAR W2,[X1]; ADD X3,X3,X2; SUBS X0,X0,#1; B.NE loop; SVC.
        // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/LDAR--Load-Acquire-Register-
        // Dynarmic emits an x86 locked operation for LDAR. On read-only RAM it
        // must use its read callback on every iteration, including the same
        // native PC and address, without treating completed loads as retries.
        let memory = fixture(
            &[
                0x88df_fc22,
                0x8b02_0063,
                0xf100_0400,
                0x54ff_ffa1,
                0xd400_0001,
            ],
            None,
        );
        memory
            .overwrite_mapped_ram(SPACE, DATA, &17_u32.to_le_bytes())
            .unwrap();
        memory
            .set_permissions(SPACE, DATA, 0x1000, MemoryPermissions::READ)
            .unwrap();
        let mut saved = A64State::default();
        saved.set_pc(CODE.get());
        saved.write_x(x(0), 16);
        saved.write_x(x(1), DATA.get());
        let mut state: ThreadState = saved.into();
        assert!(matches!(
            execute(&memory, &mut state).stop,
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(0)), 0);
        assert_eq!(state.read_x(x(2)), 17);
        assert_eq!(state.read_x(x(3)), 16 * 17);
        assert_eq!(
            memory
                .read(SPACE, DATA, MemoryAccess::normal(MemoryAccessSize::Word))
                .unwrap()
                .value,
            MemoryValue::U32(17)
        );
    });
}

#[test]
fn unaligned_simd_load_waits_for_the_second_page_without_committing_stale_bytes() {
    bounded(|| {
        let memory = fixture(&[0x3dc0_0020, 0xd400_0001], None); // LDR Q0,[X1]
        memory
            .write(
                SPACE,
                GuestVirtualAddress::new(0x2ff8),
                MemoryAccess::normal(MemoryAccessSize::Doubleword),
                MemoryValue::U64(0x1111_1111_1111_1111),
            )
            .unwrap();
        let coordinator = publish(&memory, DATA, vec![0x22; 0x1000].into_boxed_slice(), false);
        let mut saved = A64State::default();
        saved.set_pc(CODE.get());
        saved.write_x(x(1), 0x2ff8);
        let mut state: ThreadState = saved.into();
        assert!(matches!(
            execute(&memory, &mut state).stop,
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(
            state.vector(0),
            Some(0x2222_2222_2222_2222_1111_1111_1111_1111)
        );
        assert_eq!(coordinator.calls.load(Ordering::Relaxed), 1);
    });
}

#[test]
fn store_tail_preserves_completed_ram_prefix_and_vector_element_grouping() {
    bounded(|| {
        struct OverwritePrefix(Weak<ExecutionMemory>);
        impl VisibilityCoordinator for OverwritePrefix {
            fn cache_cpu_page(
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
                // The native first store has completed. Simulate a later writer
                // while admission is released for the second store's demand.
                self.0
                    .upgrade()
                    .unwrap()
                    .write(
                        SPACE,
                        GuestVirtualAddress::new(0x2ff8),
                        MemoryAccess::normal(MemoryAccessSize::Doubleword),
                        MemoryValue::U64(99),
                    )
                    .unwrap();
                Ok(vec![0x22; 0x1000].into_boxed_slice())
            }
        }
        // STP X2,X3,[X1],#16 and ST1 {V0.16B,V1.16B},[X1],#32.
        // Dynarmic uses whole vectors for ST1; the interpreter uses elements.
        for (word, base, advance, vector) in [
            (0xa881_0c22, 0x2ff8, 16, false),
            (0x4c9f_a020, 0x2ff0, 32, true),
        ] {
            let memory = fixture(&[word, 0xd400_0001], None);
            let range = memory
                .translate_canonical_range(SPACE, DATA, 0x1000, MemoryPermissions::READ)
                .unwrap();
            let coordinator = Arc::new(OverwritePrefix(Arc::downgrade(&memory)));
            let write = DeviceAccessDeclaration::write(
                NonCpuDeviceId::new(1),
                DeviceVisibilityPoint::new(1),
                DeviceVisibilityPoint::new(2),
            )
            .unwrap();
            nixe_memory::CanonicalBackingRange::prepare_resident_device_accesses(
                [(&range, write)],
                coordinator.clone(),
            )
            .unwrap();
            nixe_memory::CanonicalBackingRange::publish_device_writes(
                [(&range, write)],
                coordinator,
            )
            .unwrap();
            let mut saved = A64State::default();
            saved.set_pc(CODE.get());
            saved.write_x(x(1), base);
            saved.write_x(x(2), 17);
            saved.write_x(x(3), 34);
            if vector {
                saved.set_vector(0, u128::from_le_bytes([0x11; 16]));
                saved.set_vector(1, u128::from_le_bytes([0x22; 16]));
            }
            let mut state: ThreadState = saved.into();
            assert!(matches!(
                execute(&memory, &mut state).stop,
                CpuExit::SupervisorCall { .. }
            ));
            assert_eq!(state.read_x(x(1)), base + advance);
            assert_eq!(
                memory
                    .read(
                        SPACE,
                        GuestVirtualAddress::new(0x2ff8),
                        MemoryAccess::normal(MemoryAccessSize::Doubleword)
                    )
                    .unwrap()
                    .value
                    .bits(),
                99
            );
            assert_eq!(
                memory
                    .read(
                        SPACE,
                        DATA,
                        MemoryAccess::normal(MemoryAccessSize::Doubleword)
                    )
                    .unwrap()
                    .value
                    .bits(),
                if vector { 0x2222_2222_2222_2222 } else { 34 }
            );
        }
    });
}

#[test]
fn exclusive_store_visibility_wait_does_not_fabricate_success() {
    bounded(|| {
        let memory = fixture(
            &[
                0xc85f_7c23,
                0xd503_203f,
                0xc804_7c23,
                0xf940_0025,
                0xd400_0001,
            ],
            None,
        ); // LDXR; YIELD; STXR; LDR; SVC
        memory
            .write(
                SPACE,
                DATA,
                MemoryAccess::normal(MemoryAccessSize::Doubleword),
                MemoryValue::U64(0x2222_2222_2222_2222),
            )
            .unwrap();
        let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE);
        let mut jit = JitThread::new(Arc::new(
            JitProcess::new(cpu, memory.clone(), 19_200_000).unwrap(),
        ))
        .unwrap();
        let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
        let events = VcpuEventState::default();
        let mut saved = A64State::default();
        saved.set_pc(CODE.get());
        saved.write_x(x(1), DATA.get());
        let mut state: ThreadState = saved.into();
        let first = jit
            .run_slice(
                &mut worker,
                JitRunRequest {
                    state: &mut state,
                    instruction_budget: 100,
                    timer: &Timer,
                    events: &events,
                    capture_context: true,
                },
            )
            .unwrap();
        assert!(matches!(
            first.stop,
            CpuExit::Scheduled {
                request: nixe_cpu::execution::SchedulerRequest::Yield,
                ..
            }
        ));
        state.write_x(x(3), 99);
        let coordinator = publish(&memory, DATA, vec![0x22; 0x1000].into_boxed_slice(), false);
        let second = jit
            .run_slice(
                &mut worker,
                JitRunRequest {
                    state: &mut state,
                    instruction_budget: 100,
                    timer: &Timer,
                    events: &events,
                    capture_context: true,
                },
            )
            .unwrap();
        assert!(matches!(second.stop, CpuExit::SupervisorCall { .. }));
        // An interrupted exclusive can fail spuriously, but its status must
        // agree with whether the replacement actually became visible.
        let status = state.read_x(x(4));
        assert!(status <= 1);
        assert_eq!(
            state.read_x(x(5)),
            if status == 0 {
                99
            } else {
                0x2222_2222_2222_2222
            }
        );
        assert_eq!(coordinator.calls.load(Ordering::Relaxed), 1);
    });
}
