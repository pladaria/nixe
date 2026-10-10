//! Fixed guest workloads for architecture comparisons, using encodings already
//! covered by tests/differential.rs. Reports wall time separately from JIT ticks.
use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState};
use nixe_cpu::memory::{ExecutionMemory, MemoryPermissions};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_cpu_jit::{JitProcess, JitRunRequest, JitThread, ThreadState};
use nixe_memory::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress};
use std::{sync::Arc, time::Instant};
const SPACE: AddressSpaceId = AddressSpaceId::new(7);
struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}
fn x(i: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(i).unwrap())
}
fn main() {
    let workload = std::env::args()
        .nth(1)
        .expect("svc|mixed|switch|alias|invalidate|compile|invalidate-batch [iterations]");
    assert!(
        [
            "svc",
            "mixed",
            "switch",
            "alias",
            "invalidate",
            "compile",
            "invalidate-batch"
        ]
        .contains(&workload.as_str())
    );
    let iterations: u64 = std::env::args()
        .nth(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(100_000);
    assert!(iterations > 0);
    let compilation_code: Vec<u32> = std::iter::repeat_n(0x91000400, 511)
        .chain(std::iter::once(0xd4000001))
        .collect();
    let code: &[u32] = match workload.as_str() {
        "compile" => &compilation_code,
        "mixed" | "switch" | "alias" => &[
            0xf9000020, 0xf9400022, 0xc85f7c23, 0x91000463, 0xc8047c23, 0xf9400025, 0x1e6e1000,
            0x1e602801, 0x4f02e442, 0xd53bd046, 0xd4000841,
        ],
        "invalidate" | "invalidate-batch" => &[0xd2800540, 0xd4000001],
        _ => &[0xd4000001],
    };
    let mut memory = ExecutionMemory::new();
    for id in [1, 2] {
        assert!(memory.add_ram_page(GuestPhysicalPageId::new(id)));
    }
    memory
        .initialize_ram(
            GuestPhysicalPageId::new(1),
            0,
            &code
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for (address, id, permissions) in [
        (0x1000, 1, MemoryPermissions::READ_EXECUTE),
        (0x2000, 2, MemoryPermissions::READ_WRITE),
        (0x3000, 2, MemoryPermissions::READ_WRITE),
    ] {
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(address),
            GuestPhysicalPageId::new(id),
            permissions
        ));
    }
    if workload == "invalidate-batch" {
        // Two compiled aliases and fourteen uncompiled aliases of one page.
        for address in (0x4000..0x13000).step_by(4096) {
            assert!(memory.map_page(
                SPACE,
                GuestVirtualAddress::new(address),
                GuestPhysicalPageId::new(1),
                MemoryPermissions::READ_EXECUTE,
            ));
        }
    }
    memory
        .bind_cpu_memory_backend(SPACE, 0x20000, nixe_memory::DirectBackendPolicy::Required)
        .unwrap();
    let memory = Arc::new(memory);
    let process = Arc::new(
        JitProcess::new(
            ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE),
            memory.clone(),
            19_200_000,
        )
        .unwrap(),
    );
    let mut cores = [
        JitThread::new(process.clone()).unwrap(),
        JitThread::new(process.clone()).unwrap(),
    ];
    let mut states = [ThreadState::default(), ThreadState::default()];
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    let events = VcpuEventState::default();
    if workload == "invalidate-batch" {
        for address in [0x1000, 0x4000] {
            states[0].set_pc(address);
            cores[0]
                .run_slice(
                    &mut worker,
                    JitRunRequest {
                        state: &mut states[0],
                        instruction_budget: 1000,
                        timer: &Timer,
                        events: &events,
                        capture_context: false,
                    },
                )
                .unwrap();
        }
    }
    // Two passes: deterministic warm-up, then a fixed amount of guest work.
    for (phase, count) in [("warmup", 1000), ("sample", iterations)] {
        let start = Instant::now();
        let mut ticks = 0;
        for i in 0..count {
            let thread = if workload == "switch" {
                (i % 2) as usize
            } else {
                0
            };
            // Each synthetic guest thread migrates between two host-owned JIT cores.
            let core = if workload == "switch" {
                ((i / 2) % 2) as usize
            } else {
                0
            };
            if workload == "switch" {
                states[1 - thread].materialize();
            }
            let state = &mut states[thread];
            state.set_pc(if workload == "invalidate-batch" && i % 2 == 1 {
                0x4000
            } else {
                0x1000
            });
            state.write_x(x(0), 41 + thread as u64);
            state.write_x(
                x(1),
                if workload == "alias" && i % 2 == 0 {
                    0x3000
                } else {
                    0x2000
                },
            );
            state.set_tpidr_el0(0xabcdef + thread as u64);
            if matches!(
                workload.as_str(),
                "invalidate" | "invalidate-batch" | "compile"
            ) {
                let instruction = if i % 2 == 0 {
                    0xd2800540_u32
                } else {
                    0xd2800560
                };
                let instruction = if workload == "compile" {
                    0x91000400
                } else {
                    instruction
                };
                for _ in 0..if workload == "invalidate-batch" {
                    64
                } else {
                    1
                } {
                    memory
                        .overwrite_mapped_ram(
                            SPACE,
                            GuestVirtualAddress::new(0x1000),
                            &instruction.to_le_bytes(),
                        )
                        .unwrap();
                }
            }
            let result = cores[core]
                .run_slice(
                    &mut worker,
                    JitRunRequest {
                        state,
                        instruction_budget: 1000,
                        timer: &Timer,
                        events: &events,
                        capture_context: false,
                    },
                )
                .unwrap();
            assert!(matches!(result.stop, CpuExit::SupervisorCall { .. }));
            ticks += result.progress;
            if matches!(workload.as_str(), "mixed" | "switch" | "alias") {
                assert_eq!(state.read_x(x(5)), 42 + thread as u64);
                assert_eq!(state.read_x(x(4)), 0);
                assert_eq!(state.read_x(x(6)), 0xabcdef + thread as u64);
                assert_eq!(state.vector(1), Some(u128::from(2_f64.to_bits())));
                assert_eq!(state.vector(2), Some(u128::from_le_bytes([0x42; 16])));
            } else if matches!(workload.as_str(), "invalidate" | "invalidate-batch") {
                assert_eq!(state.read_x(x(0)), 42 + i % 2);
            } else if workload == "compile" {
                assert_eq!(state.read_x(x(0)), 41 + 511);
            }
        }
        println!(
            "{{\"workload\":\"{workload}\",\"phase\":\"{phase}\",\"iterations\":{count},\"seconds\":{},\"guest_ticks\":{ticks}}}",
            start.elapsed().as_secs_f64()
        );
    }
    #[cfg(feature = "performance-counters")]
    for (name, value) in nixe_cpu_jit::metrics::snapshot()
        .into_iter()
        .chain(nixe_memory::metrics::snapshot())
    {
        println!("{name}={value}");
    }
}
