//! Diagnostic fastmem fault attribution. This is not a game timing benchmark.
//! Guest encodings are also exercised by tests/differential.rs.
use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState};
use nixe_cpu::memory::{
    CpuMemory, DataAccessFaultReason, ExecutionMemory, MemoryAccess, MemoryAccessSize,
    MemoryAlignment, ProcessMemory,
};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_cpu_jit::{JitProcess, JitRunRequest, JitThread, ThreadState};
use nixe_memory::*;
use std::{collections::BTreeMap, sync::Arc, time::Instant};

const SPACE: AddressSpaceId = AddressSpaceId::new(7);
const ACCESSES: usize = 64;
struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}
struct Device;
impl VisibilityCoordinator for Device {
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
        let mut bytes = vec![0; 4096];
        bytes[..8].copy_from_slice(&7_u64.to_le_bytes());
        Ok(bytes.into_boxed_slice())
    }
}
fn x(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}
fn counters() -> BTreeMap<&'static str, u64> {
    nixe_cpu_jit::metrics::snapshot()
        .into_iter()
        .chain(nixe_memory::metrics::snapshot())
        .filter(|(name, _)| {
            matches!(
                *name,
                "JitNativeEntries"
                    | "JitGuestTicks"
                    | "JitFaultDispatches"
                    | "JitMemoryCallbacks"
                    | "GateSharedAcquisitions"
                    | "GateExclusiveAcquisitions"
                    | "DirectProtectionCalls"
                    | "DirectProtectionBytes"
                    | "PageOwnershipUpdates"
            )
        })
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workload = std::env::args().nth(1).ok_or("expected ram|alias|read-only|clean-read|tracking-write|gpu-read|gpu-write|cross-page|denied-write|cross-page-denied|unmapped|out-of-range|exclusive [iterations]")?;
    let iterations = std::env::args()
        .nth(2)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(10_000);
    assert!(iterations > 0);
    let (instructions, writes, invalid) = match workload.as_str() {
        "ram" | "alias" => (vec![0xf9400022_u32, 0xf9000020], true, false),
        "tracking-write" | "gpu-write" => (vec![0xf9000020], true, false),
        "exclusive" => (vec![0xc85f7c22, 0xc8037c20], true, false),
        "read-only" | "clean-read" | "gpu-read" | "cross-page" => (vec![0xf9400022], false, false),
        "unmapped" | "out-of-range" => (vec![0xf9400022], false, true),
        "denied-write" | "cross-page-denied" => (vec![0xf9000020], true, true),
        _ => return Err("unknown workload".into()),
    };
    let code = instructions
        .iter()
        .copied()
        .cycle()
        .take(instructions.len() * ACCESSES)
        .chain(std::iter::once(0xd4000001))
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let mut memory = ExecutionMemory::new();
    for id in [1, 2] {
        assert!(memory.add_ram_page(GuestPhysicalPageId::new(id)));
    }
    memory.initialize_ram(GuestPhysicalPageId::new(1), 0, &code)?;
    memory.initialize_ram(GuestPhysicalPageId::new(2), 0, &7_u64.to_le_bytes())?;
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
    memory.bind_cpu_memory_backend(SPACE, 0x20000, DirectBackendPolicy::Required)?;
    let range = memory.translate_canonical_range(
        SPACE,
        GuestVirtualAddress::new(0x2000),
        4096,
        MemoryPermissions::READ_WRITE,
    )?;
    if matches!(
        workload.as_str(),
        "read-only" | "denied-write" | "cross-page-denied"
    ) {
        memory
            .set_permissions(
                SPACE,
                GuestVirtualAddress::new(if workload == "cross-page-denied" {
                    0x3000
                } else {
                    0x2000
                }),
                if workload == "read-only" { 8192 } else { 4096 },
                MemoryPermissions::READ,
            )
            .expect("read-only aliases must support protection changes");
    }
    let memory = Arc::new(memory);
    let process = Arc::new(JitProcess::new(
        ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE),
        memory.clone(),
        19_200_000,
    )?);
    let mut core = JitThread::new(process.clone())?;
    let mut state = ThreadState::default();
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    let events = VcpuEventState::default();
    let device: Arc<dyn VisibilityCoordinator> = Arc::new(Device);
    let mut point = 0;
    for (phase, count) in [("warmup", 100), ("sample", iterations)] {
        let before = counters();
        let start = Instant::now();
        for index in 0..count {
            point += 1;
            if matches!(workload.as_str(), "clean-read" | "tracking-write") {
                CanonicalBackingRange::prepare_resident_device_accesses(
                    [(
                        &range,
                        DeviceAccessDeclaration::read(
                            NonCpuDeviceId::new(1),
                            DeviceVisibilityPoint::new(point),
                        ),
                    )],
                    device.clone(),
                )?;
            } else if matches!(workload.as_str(), "gpu-read" | "gpu-write") {
                let declaration = DeviceAccessDeclaration::write(
                    NonCpuDeviceId::new(1),
                    DeviceVisibilityPoint::new(point),
                    DeviceVisibilityPoint::new(point),
                )?;
                CanonicalBackingRange::prepare_resident_device_accesses(
                    [(&range, declaration)],
                    device.clone(),
                )?;
                CanonicalBackingRange::publish_device_writes(
                    [(&range, declaration)],
                    device.clone(),
                )?;
            }
            let address = match workload.as_str() {
                "alias" if index % 2 == 0 => 0x3000,
                "cross-page" | "cross-page-denied" => 0x2ffc,
                "unmapped" => 0x5000,
                "out-of-range" => 0x20000,
                _ => 0x2000,
            };
            state.set_pc(0x1000);
            state.write_x(x(0), 42);
            state.write_x(x(1), address);
            state.write_x(x(2), 0xdeadbeef);
            let result = core.run_slice(
                &mut worker,
                JitRunRequest {
                    state: &mut state,
                    instruction_budget: 1000,
                    timer: &Timer,
                    events: &events,
                    capture_context: false,
                },
            )?;
            if invalid {
                let fault_address = if workload == "cross-page-denied" {
                    0x3000
                } else {
                    address
                };
                assert!(
                    matches!(result.stop, CpuExit::DataFault { ref source, ref fault }
                    if source.pc.get() == 0x1000 && fault.address.get() == fault_address
                        && fault.reason == if writes { DataAccessFaultReason::WritePermissionDenied } else { DataAccessFaultReason::Unmapped })
                );
                assert_eq!(state.read_x(x(2)), 0xdeadbeef);
                if writes {
                    assert_eq!(
                        memory
                            .read(
                                SPACE,
                                GuestVirtualAddress::new(address),
                                MemoryAccess {
                                    alignment: MemoryAlignment::Unaligned,
                                    ..MemoryAccess::normal(MemoryAccessSize::Doubleword)
                                }
                            )
                            .expect("denied store must leave both readable aliases intact")
                            .value
                            .bits(),
                        if workload == "cross-page-denied" {
                            7 << 32
                        } else {
                            7
                        }
                    );
                }
            } else {
                assert!(matches!(result.stop, CpuExit::SupervisorCall { .. }));
                if writes {
                    assert_eq!(
                        memory
                            .read(
                                SPACE,
                                GuestVirtualAddress::new(address),
                                MemoryAccess::normal(MemoryAccessSize::Doubleword)
                            )
                            .expect("successful guest store must be readable")
                            .value
                            .bits(),
                        42
                    );
                    if workload == "exclusive" {
                        assert_eq!(state.read_x(x(3)), 0);
                    }
                } else {
                    assert_eq!(
                        state.read_x(x(2)),
                        if workload == "cross-page" { 7 << 32 } else { 7 }
                    );
                }
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        let changes = counters()
            .into_iter()
            .map(|(name, value)| format!("\"{name}\":{}", value - before[name]))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{{\"workload\":\"{workload}\",\"phase\":\"{phase}\",\"iterations\":{count},\"memory_instructions_per_successful_entry\":{},\"seconds\":{elapsed},\"counters\":{{{changes}}}}}",
            instructions.len() * ACCESSES
        );
    }
    drop(state);
    drop(core);
    assert!(process.try_shutdown()?);
    Ok(())
}
