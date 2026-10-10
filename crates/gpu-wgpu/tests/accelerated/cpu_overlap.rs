use super::*;
use nixe_cpu::execution::{
    ArchitecturalTimer, ControlRequest, CpuExit, TimerSnapshot, VcpuEventState,
};
use nixe_cpu::memory::{CpuMemory, ExecutionMemory, MemoryAccess, MemoryAccessSize, ProcessMemory};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State};
use nixe_cpu_jit::{JitProcess, JitRunRequest, JitThread, ThreadState};
use nixe_memory::{AddressSpaceId, CanonicalRangeTranslator, GuestVirtualAddress};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::time::Duration;

const SPACE: AddressSpaceId = AddressSpaceId::new(81);
const GENERATIONS: u32 = 32;
const TIMEOUT: Duration = Duration::from_secs(10);

struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}

enum Job {
    Store(u32),
    Read(u64),
    Stop,
}

struct DemandedReadback {
    owner: Weak<RuntimeOwner>,
    entered: mpsc::Sender<()>,
}
impl BackendVisibilityRequester for DemandedReadback {
    fn make_cpu_visible(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        self.entered.send(()).unwrap();
        RuntimeRequester(self.owner.clone()).make_cpu_visible(request)
    }
}

fn register(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}

#[test]
fn native_cpu_overlaps_submission_reuse_readback_and_shutdown() {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(8100),
        NonCpuDeviceId::new(8100),
        Default::default(),
    ) else {
        return;
    };
    let (completed, completion) = mpsc::channel();
    let test = std::thread::spawn(move || {
        let owner = Arc::new(RuntimeOwner {
            runtime: Mutex::new(initialized.into_runtime()),
        });
        let (demanded, demand) = mpsc::channel();
        owner
            .runtime()
            .bind_visibility_requester(Arc::new(DemandedReadback {
                owner: Arc::downgrade(&owner),
                entered: demanded,
            }))
            .unwrap();

        let mut memory = ExecutionMemory::new();
        // STR W0,[X1]; SVC; LDR W2,[X1]; SVC; ADD W0,W0,#1; STR W0,[X1]; B -8.
        // Arm instruction semantics are independently checked by the JIT differential suite.
        let code = [
            0xb900_0020_u32,
            0xd400_0001,
            0xb940_0022,
            0xd400_0001,
            0x1100_0400,
            0xb900_0020,
            0x17ff_fffe,
        ];
        for page in 1..=GENERATIONS + 3 {
            let physical = GuestPhysicalPageId::new(u64::from(page));
            assert!(memory.add_ram_page(physical));
            if page == 1 {
                memory
                    .initialize_ram(
                        physical,
                        0,
                        &code
                            .into_iter()
                            .flat_map(u32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
            }
            assert!(memory.map_page(
                SPACE,
                GuestVirtualAddress::new(u64::from(page) * 4096),
                physical,
                if page == 1 {
                    MemoryPermissions::READ_EXECUTE
                } else {
                    MemoryPermissions::READ_WRITE
                }
            ));
            if page > 3 {
                assert!(memory.map_page(
                    SPACE,
                    GuestVirtualAddress::new(0x80000 + u64::from(page) * 4096),
                    physical,
                    MemoryPermissions::READ_WRITE
                ));
            }
        }
        memory
            .bind_cpu_memory_backend(SPACE, 0x100000, nixe_memory::DirectBackendPolicy::Required)
            .unwrap();
        let memory = Arc::new(memory);
        let (jobs, job) = mpsc::channel();
        let (results, result) = mpsc::channel();
        let (controls, control) = mpsc::channel();
        let slices = Arc::new(AtomicUsize::new(0));
        let cpu_memory = memory.clone();
        let cpu_slices = slices.clone();
        let cpu = std::thread::spawn(move || {
            let context = ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE);
            let mut jit = JitThread::new(Arc::new(
                JitProcess::new(context, cpu_memory, 19_200_000).unwrap(),
            ))
            .unwrap();
            controls.send(jit.control()).unwrap();
            let mut native = nixe_cpu_direct_memory::NativeWorker::default();
            let events = VcpuEventState::default();
            let mut state: ThreadState = A64State::default().into();
            loop {
                let current = match job.try_recv() {
                    Ok(Job::Stop) | Err(mpsc::TryRecvError::Disconnected) => break,
                    Ok(current) => Some(current),
                    Err(mpsc::TryRecvError::Empty) => None,
                };
                let (pc, address) = match current {
                    Some(Job::Store(value)) => {
                        state.write_w(register(0), value);
                        (0x1000, 0x2000)
                    }
                    Some(Job::Read(address)) => (0x1008, address),
                    None => (0x1010, 0x3000),
                    Some(Job::Stop) => unreachable!(),
                };
                state.set_pc(pc);
                state.write_x(register(1), address);
                loop {
                    let report = jit
                        .run_slice(
                            &mut native,
                            JitRunRequest {
                                state: &mut state,
                                instruction_budget: 256,
                                timer: &Timer,
                                events: &events,
                                capture_context: false,
                            },
                        )
                        .unwrap();
                    if matches!(report.stop, CpuExit::SupervisorCall { .. }) {
                        assert!(current.is_some());
                        results.send(state.read_w(register(2))).unwrap();
                        break;
                    }
                    assert!(matches!(
                        report.stop,
                        CpuExit::BudgetExhausted | CpuExit::Safepoint
                    ));
                    if current.is_none() {
                        cpu_slices.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                }
            }
        });
        let control = control.recv_timeout(TIMEOUT).unwrap();
        memory.set_transition_notifier(Some(Arc::new(move || {
            control.request(ControlRequest::Preempt)
        })));
        let make_buffer = |id: BufferId, allocation: GpuAllocationId, address| {
            let description = GpuAllocationDescription::new(64, 4).unwrap();
            let range = memory
                .translate_canonical_range(
                    SPACE,
                    GuestVirtualAddress::new(address),
                    64,
                    MemoryPermissions::READ_WRITE,
                )
                .unwrap();
            vec![
                BackendResourceCreateInfo::Allocation {
                    id: allocation,
                    description,
                },
                BackendResourceCreateInfo::Buffer {
                    id,
                    description: BufferDescription::new(64).unwrap(),
                    view: Some(
                        BufferView::new(
                            id,
                            BufferDescription::new(64).unwrap(),
                            0,
                            BackingView::new(allocation, description, 0, range).unwrap(),
                        )
                        .unwrap(),
                    ),
                },
            ]
        };
        let source = BufferId::new(8100);
        let destination = BufferId::new(8101);
        let mut creations = make_buffer(source, GpuAllocationId::new(8100), 0x2000);
        let mut destinations = Vec::new();
        #[cfg(feature = "performance-counters")]
        let readbacks_before = readback_counts();
        for generation in 0..GENERATIONS {
            let address = u64::from(generation + 4) * 4096;
            jobs.send(Job::Store(0xa100_0000 + generation)).unwrap();
            result.recv_timeout(TIMEOUT).unwrap();
            let destination_creation = make_buffer(
                destination,
                GpuAllocationId::new(8200 + u64::from(generation)),
                address,
            );
            // Reuse the logical buffer slot, retaining distinct physical generations.
            creations.extend(destination_creation);
            let submission = OperationSubmission::new(
                FrontendSubmissionId::new(u64::from(generation) + 1),
                vec![],
                vec![GpuOperation::new(
                    GpuCommand::Copy(CopyOperation::BufferToBuffer {
                        source: BufferRegion {
                            buffer: source,
                            range: BufferRange::new(0, 4).unwrap(),
                        },
                        destination: BufferRegion {
                            buffer: destination,
                            range: BufferRange::new(0, 4).unwrap(),
                        },
                    }),
                    [],
                    [],
                    CapabilityRequirements::none(),
                )],
            )
            .unwrap();
            owner
                .runtime()
                .submit(
                    &creations,
                    &[ResourceDependency::Buffer(destination)],
                    &submission,
                )
                .unwrap();
            owner.runtime().wait_for_completion().unwrap().unwrap();
            creations.clear();
            destinations.push(address);
        }
        assert!(
            slices.load(Ordering::Relaxed) > 0,
            "guest scratch stores must overlap GPU work"
        );
        #[cfg(feature = "performance-counters")]
        assert_eq!(
            readback_counts(),
            readbacks_before,
            "submission/reuse must stay resident"
        );
        for (generation, address) in destinations.into_iter().enumerate().rev() {
            let gpu = owner.runtime();
            jobs.send(Job::Read(0x80000 + address)).unwrap();
            demand
                .recv_timeout(TIMEOUT)
                .expect("native alias load must demand its retired GPU generation");
            // CPU is blocked on this owner's mutex. A real mapping mutation must still
            // finish: native demand may not retain its execution lease/page/mapping locks.
            memory
                .set_permissions(
                    SPACE,
                    GuestVirtualAddress::new(0x80000 + address),
                    4096,
                    MemoryPermissions::READ,
                )
                .unwrap();
            drop(gpu);
            assert_eq!(
                result.recv_timeout(TIMEOUT).unwrap(),
                0xa100_0000 + generation as u32
            );
        }
        #[cfg(feature = "performance-counters")]
        {
            let after = readback_counts();
            assert_eq!(after.0 - readbacks_before.0, u64::from(GENERATIONS));
            assert_eq!(after.1 - readbacks_before.1, u64::from(GENERATIONS) * 4);
            assert_eq!(after.2, readbacks_before.2);
        }
        // Drain accepted work during shutdown while native scratch stores still run.
        let creations = make_buffer(destination, GpuAllocationId::new(8200), 0x4000);
        let clear = ClearOperation::buffer(
            BufferRegion {
                buffer: destination,
                range: BufferRange::new(0, 4).unwrap(),
            },
            0x1234,
        )
        .unwrap();
        owner
            .runtime()
            .submit(
                &creations[1..],
                &[],
                &OperationSubmission::new(
                    FrontendSubmissionId::new(100),
                    vec![],
                    vec![GpuOperation::new(
                        GpuCommand::Clear(clear),
                        [],
                        [],
                        CapabilityRequirements::none(),
                    )],
                )
                .unwrap(),
            )
            .unwrap();
        owner.runtime().teardown().unwrap();
        jobs.send(Job::Stop).unwrap();
        cpu.join().unwrap();
        memory.set_transition_notifier(None);
        let _ = memory
            .read(
                SPACE,
                GuestVirtualAddress::new(0x3000),
                MemoryAccess::normal(MemoryAccessSize::Word),
            )
            .unwrap();
        completed.send(()).unwrap();
    });
    completion
        .recv_timeout(Duration::from_secs(45))
        .expect("CPU/GPU overlap and shutdown must finish within the bound");
    test.join().unwrap();
}

#[cfg(feature = "performance-counters")]
fn readback_counts() -> (u64, u64, u64) {
    let counters = nixe_gpu::metrics::snapshot();
    let get = |name| counters.iter().find(|(key, _)| *key == name).unwrap().1;
    (
        get("DeviceReadbackCopies"),
        get("BufferReadbackBytes"),
        get("ImageReadbackBytes"),
    )
}
