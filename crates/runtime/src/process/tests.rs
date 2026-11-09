use super::*;

use std::fs;

use crate::{Launcher, LauncherInput};

pub(crate) fn test_kernel() -> std::sync::Arc<nixe_scheduler::SchedulerState> {
    use nixe_scheduler::{
        MachineSchedulerProfile, PriorityRange, SchedulerState, VirtualCpuDescriptor, VirtualCpuId,
    };
    SchedulerState::new(
        MachineSchedulerProfile::new(
            (0..8)
                .map(|id| VirtualCpuDescriptor::new(VirtualCpuId::new(id), 0))
                .collect(),
            PriorityRange::new(0, 63).unwrap(),
        )
        .unwrap(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    )
}

pub(crate) fn reference_process_builder(
    kernel: std::sync::Arc<nixe_scheduler::SchedulerState>,
) -> ProcessBuilder {
    let machine = nixe_memory::Machine::switch_ram().unwrap();
    let system = machine.account(nixe_memory::DEFAULT_RAM_BYTES).unwrap();
    ProcessBuilder::new(machine, system, kernel)
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn replace_entry_instructions(process: &RunnableProcess, encodings: &[u32]) {
    let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
    write_code(
        process,
        entry,
        &encodings
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>(),
    );
}

fn write_code(process: &RunnableProcess, address: GuestVirtualAddress, bytes: &[u8]) {
    let ranges = process
        .memory
        .space()
        .capture(address, bytes.len() as u64, MemoryPermissions::EXECUTE)
        .unwrap();
    process
        .memory
        .account()
        .machine()
        .write_resolved(&ranges, bytes)
        .unwrap();
}

fn synthetic_nro() -> Vec<u8> {
    let mut bytes = vec![0; 0x2800];
    bytes[..4].copy_from_slice(&0x1400_0020_u32.to_le_bytes()); // B entry + 0x80
    bytes[0x80..0x84].copy_from_slice(&0xd420_0000_u32.to_le_bytes()); // BRK #0
    bytes[0x10..0x14].copy_from_slice(b"NRO0");
    put_u32(&mut bytes, 0x18, 0x2800);
    put_u32(&mut bytes, 0x20, 0);
    put_u32(&mut bytes, 0x24, 0x1000);
    put_u32(&mut bytes, 0x28, 0x1000);
    put_u32(&mut bytes, 0x2c, 0x1000);
    put_u32(&mut bytes, 0x30, 0x2000);
    put_u32(&mut bytes, 0x34, 0x800);
    put_u32(&mut bytes, 0x38, 0x800);
    bytes[0x40..0x60].fill(0x5a);
    bytes
}

fn plan() -> (tempfile::TempDir, LaunchPlan) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("synthetic.nro");
    fs::write(&path, synthetic_nro()).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    (directory, plan)
}

pub(crate) fn synthetic_process(
    kernel: &std::sync::Arc<nixe_scheduler::SchedulerState>,
    process_id: u64,
) -> RunnableProcess {
    let (_directory, plan) = plan();
    reference_process_builder(kernel.clone())
        .with_config(ProcessBuildConfig {
            process_id,

            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap()
}

pub(crate) fn synthetic_instruction_process(
    kernel: &std::sync::Arc<nixe_scheduler::SchedulerState>,
    process_id: u64,
    encodings: &[u32],
) -> RunnableProcess {
    let process = synthetic_process(kernel, process_id);
    replace_entry_instructions(&process, encodings);
    process
}

pub fn ready<T>(future: impl std::future::Future<Output = T>) -> T {
    match std::pin::pin!(future).poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    {
        std::task::Poll::Ready(value) => value,
        std::task::Poll::Pending => {
            panic!("fixture setup unexpectedly requires asynchronous progress")
        }
    }
}

pub(crate) fn poll_session(
    session: &crate::SessionObject,
    owner: crate::SessionRequestOwner,
) -> Result<Option<crate::SessionMessage>, crate::SessionError> {
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, Waker},
    };
    match Pin::new(&mut session.response(owner)).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Pending => Ok(None),
        Poll::Ready(result) => result.map(Some),
    }
}

pub(crate) fn watch_future<F: std::future::Future + Unpin>(
    mut future: F,
    callback: std::sync::Arc<dyn Fn() + Send + Sync>,
) -> F {
    use std::{
        pin::Pin,
        task::{Context, Wake, Waker},
    };
    struct Notify(std::sync::Arc<dyn Fn() + Send + Sync>);
    impl Wake for Notify {
        fn wake(self: std::sync::Arc<Self>) {
            (self.0)();
        }
    }
    let waker = Waker::from(std::sync::Arc::new(Notify(callback)));
    assert!(
        Pin::new(&mut future)
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    future
}
