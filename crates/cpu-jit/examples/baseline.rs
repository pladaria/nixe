//! Fixed guest workloads for native execution comparisons.
//! Reports wall time for a fixed amount of guest work.
use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_cpu_jit::{JitProcess, JitRunRequest, ThreadState};
use nixe_memory::{
    AddressSpace, AddressWidth, AllocationRange, GuestVirtualAddress, Machine, MemoryPermissions,
    PAGE_SIZE,
};
use std::time::Instant;
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
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let account = machine.account(2 * PAGE_SIZE).unwrap();
    let executable = account.allocate(PAGE_SIZE).unwrap();
    let data = account.allocate(PAGE_SIZE).unwrap();
    let encoded: Vec<_> = code.iter().flat_map(|word| word.to_le_bytes()).collect();
    machine
        .write_resolved(
            &[AllocationRange::new(executable.pin(), 0, encoded.len() as u64).unwrap()],
            &encoded,
        )
        .unwrap();
    let code_word = AllocationRange::new(executable.pin(), 0, 4).unwrap();
    let memory = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    memory
        .map(
            GuestVirtualAddress::new(0x1000),
            executable.clone(),
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_EXECUTE,
        )
        .unwrap();
    for address in [0x2000, 0x3000] {
        memory
            .map(
                GuestVirtualAddress::new(address),
                data.clone(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
    }
    if workload == "invalidate-batch" {
        // Two compiled aliases and fourteen uncompiled aliases of one page.
        for address in (0x4000..0x13000).step_by(PAGE_SIZE as usize) {
            memory
                .map(
                    GuestVirtualAddress::new(address),
                    executable.clone(),
                    0,
                    PAGE_SIZE,
                    MemoryPermissions::READ_EXECUTE,
                )
                .unwrap();
        }
    }
    let process = JitProcess::new(
        ProcessCpuContext::for_platform(TargetPlatform::Switch1, memory.identity()),
        memory,
        19_200_000,
    )
    .unwrap();
    let mut cores = [
        process.create_thread().unwrap(),
        process.create_thread().unwrap(),
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
                        timer: &Timer,
                        events: &events,
                    },
                )
                .unwrap();
        }
    }
    // Two passes: deterministic warm-up, then a fixed amount of guest work.
    for (phase, count) in [("warmup", 1000), ("sample", iterations)] {
        let start = Instant::now();
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
                    machine
                        .write_resolved(
                            std::slice::from_ref(&code_word),
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
                        timer: &Timer,
                        events: &events,
                    },
                )
                .unwrap();
            assert!(matches!(result.stop, CpuExit::SupervisorCall { .. }));
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
            "{{\"workload\":\"{workload}\",\"phase\":\"{phase}\",\"iterations\":{count},\"seconds\":{}}}",
            start.elapsed().as_secs_f64()
        );
    }
    #[cfg(feature = "performance-counters")]
    for (name, value) in nixe_cpu_jit::metrics::snapshot() {
        println!("{name}={value}");
    }
}
