//! One admitted native invocation. No borrowed code or captured machine image
//! escapes this boundary; typed memory completion runs after it returns.

use super::fault::{self, cold::Completion};
use crate::{
    abi::{BlockKey, ExclusiveStoreOperation, NativeFrame, PollOutcome, PublishedEntry},
    lifetime::{
        self, FaultLookup, Reader,
        unit::{EdgeKind, GuestExit, Instruction},
    },
    native::{self, NativeReturn, NativeReturnError},
};
use nixe_cpu::{
    exclusive::ExclusiveMonitorState,
    memory::{DataAccessFault, DirectFaultResolution, ExecutionMemory},
};
use nixe_cpu_direct_memory::{
    CapturedFault, FaultDisposition, FaultRuntimeError, InvocationOutcome, NativeInvocation,
    WorkerFaultContext,
};
use nixe_memory::DirectAddressSpaceView;

mod memory;

pub(crate) enum Exit {
    Native {
        // Linked execution consumes this outcome in Task 4; the unlinked loop
        // already receives the reconciled PollBudget separately.
        #[allow(dead_code)]
        returned: NativeReturn,
        guest: GuestExit,
        /// The exiting instruction, not the destination PC's current bytes.
        instruction: Instruction,
    },
    Memory {
        instruction: Instruction,
        #[allow(dead_code)] // Task 4 native polling; LCQ uses reconciled budget.
        poll: PollOutcome,
        outcome: MemoryExit,
    },
}

pub(crate) enum MemoryExit {
    // Allocate only on a cold memory escape, not on every native return.
    Cold(Box<Completion>),
    /// A canonical PRE exit using a reservation from another invocation or VA.
    ExclusiveStore(ExclusiveStoreOperation),
    Fault(DataAccessFault),
    /// An attributed access whose published host mapping contradicts policy.
    /// This is terminal, not a guest fault or a request to retry the instruction.
    Fatal(Box<str>),
}

#[derive(Debug)]
pub(crate) enum Error {
    Lifetime(lifetime::Error),
    Runtime(FaultRuntimeError),
    Native(NativeReturnError),
    Exclusive(DataAccessFault),
    Internal(&'static str),
}
impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lifetime(error) => write!(formatter, "admission: {error}"),
            Self::Runtime(error) => write!(formatter, "fault runtime: {error}"),
            Self::Native(error) => write!(formatter, "native return: {error:?}"),
            Self::Exclusive(error) => write!(formatter, "exclusive completion: {error:?}"),
            Self::Internal(detail) => formatter.write_str(detail),
        }
    }
}

/// Acquire mapping protection before admission and keep it through capture,
/// repair/retry, reconstruction and exclusive-monitor handoff. A cache miss
/// returns None; Closing/Closed retain their lifetime error so the caller can
/// wait only AFTER both protections have been released. Errors are terminal
/// for this invocation, except admission failures before any native execution.
///
/// # Safety
/// The reader and memory belong to the same process. Published entries are
/// unlinked LCQ fragments with contiguous instruction images; they use
/// this host's checked NativeFrame ABI and this memory's arena size. No other
/// FP owner is active on this OS thread. The canonical state matches `key`,
/// and the frame has no pending exclusive load from an earlier invocation.
/// The caller must not resume guest execution after an internal/runtime error.
pub(crate) unsafe fn run(
    reader: &mut Reader,
    frame: &mut NativeFrame<'_>,
    memory: &ExecutionMemory,
    worker: &mut WorkerFaultContext,
    monitor: &mut ExclusiveMonitorState,
    key: BlockKey,
) -> Result<Option<Exit>, Error> {
    // Declaration order matters: Invocation must quiesce before the mapping
    // lease is released, including on every early error return.
    let _lease = memory.acquire_execution_lease();
    let arena = memory
        .direct_address_space_view(key.address_space)
        .ok_or(Error::Internal("LCQ invocation has no bound direct arena"))?;
    let Some(mut invocation) = (unsafe { reader.admit(frame, key) }).map_err(Error::Lifetime)?
    else {
        return Ok(None);
    };
    let entry = invocation.payload().preferred().unwrap();
    let (frame, lookup) = invocation.frame_and_faults();
    let mut dispatch = Dispatch {
        frame: std::ptr::from_mut(frame).cast(),
        lookup,
        memory,
        arena,
        resolution: None,
    };
    // Raw pointers only: the callback borrows the frame while the gateway is
    // suspended. No Rust &mut frame is retained in the invocation context.
    let mut call = Entry {
        frame: dispatch.frame,
        arena: arena.base as *mut u8,
        result: None,
    };
    let returned = unsafe {
        worker.invoke_captured(
            arena,
            [frame.host_fp.saved_control, frame.host_fp.saved_status],
            dispatch_fault,
            std::ptr::from_mut(&mut dispatch).cast(),
            NativeInvocation {
                gateway: enter,
                context: std::ptr::from_mut(&mut call).cast(),
                entry: entry.canonical.get(),
            },
        )
    }
    .map_err(Error::Runtime)?;
    let exit = match returned {
        InvocationOutcome::Returned => {
            let returned = call
                .result
                .ok_or(Error::Internal("LCQ gateway returned without an exit"))?
                .map_err(Error::Native)?;
            let (guest, instruction) = canonical_exit(&dispatch.lookup, entry, frame)?;
            if let EdgeKind::ExclusiveStore(operation) = guest.kind {
                Exit::Memory {
                    instruction,
                    poll: returned.poll,
                    outcome: MemoryExit::ExclusiveStore(operation),
                }
            } else {
                Exit::Native {
                    returned,
                    guest,
                    instruction,
                }
            }
        }
        InvocationOutcome::Escaped => {
            let captured = worker.escaped_fault().map_err(Error::Runtime)?;
            let fault = dispatch
                .lookup
                .find(captured.native_pc())
                .ok_or(Error::Internal("LCQ escaped PC has no live fault record"))?;
            let (completed, instruction) = fault
                .unit
                .instructions
                .iter()
                .enumerate()
                .find(|(_, instruction)| instruction.key == fault.record.instruction)
                .ok_or(Error::Internal(
                    "LCQ fault has no captured guest instruction",
                ))?;
            let instruction = *instruction;
            let reconstructed =
                unsafe { fault::reconstruct(frame, &captured, &fault) }.map_err(Error::Internal)?;
            let resolution = dispatch
                .resolution
                .take()
                .ok_or(Error::Internal("LCQ escape has no memory resolution"))?
                .map_err(Error::Internal)?;
            let outcome = match resolution {
                DirectFaultResolution::Cold => MemoryExit::Cold(Box::new(
                    unsafe { Completion::prepare(frame, &fault, reconstructed.completed_read) }
                        .map_err(Error::Internal)?,
                )),
                DirectFaultResolution::Fault(fault) => MemoryExit::Fault(fault),
                DirectFaultResolution::Fatal(detail) => MemoryExit::Fatal(detail),
                DirectFaultResolution::Retry => {
                    return Err(Error::Internal("LCQ retry unexpectedly escaped"));
                }
            };
            let poll = frame
                .budget
                // No canonical epilogue ran on escape. Charge only the prefix;
                // repair/retry never comes here and cold completion owns the
                // still-uncommitted instruction, including partial pair accesses.
                .reconcile(
                    reconstructed
                        .poll_remaining
                        .checked_sub(completed as i64)
                        .ok_or(Error::Internal("LCQ fault work accounting overflow"))?,
                    false,
                )
                .map_err(|error| Error::Native(NativeReturnError::Budget(error)))?;
            Exit::Memory {
                instruction,
                poll,
                outcome,
            }
        }
    };
    frame
        .finish_exclusive_load(memory, key.address_space, monitor)
        .map_err(Error::Exclusive)?;
    // Exit owns everything it needs. Dropping Invocation finishes FP before
    // announcing quiescence; only then can the lease admit mapping mutations.
    Ok(Some(exit))
}

/// Consume exit identity before quiescence. The entry is the already-admitted
/// value, never a fresh dispatch lookup that could select a replacement unit.
/// Inter-unit native linking will need the actual exiting unit's identity;
/// an unlinked LCQ invocation must exit from the unit it entered.
fn canonical_exit(
    lookup: &FaultLookup<'_>,
    entry: PublishedEntry,
    frame: &NativeFrame<'_>,
) -> Result<(GuestExit, Instruction), Error> {
    let unit = lookup
        .unit(entry.canonical.get())
        .ok_or(Error::Internal("LCQ canonical entry has no live code unit"))?;
    if unit.id != entry.unit
        || unit.version != entry.version
        || frame.exit_source_version != entry.version.get()
    {
        return Err(Error::Internal(
            "LCQ canonical exit has a different source version",
        ));
    }
    let guest = unit
        .states
        .get(frame.exit_state_map as usize)
        .and_then(|record| record.exit)
        .ok_or(Error::Internal(
            "LCQ canonical exit has no guest exit record",
        ))?;
    let first = unit
        .instructions
        .first()
        .ok_or(Error::Internal("LCQ exit unit has no instruction image"))?;
    let instruction = guest
        .pc
        .get()
        .checked_sub(first.key.block_key().pc.get())
        .filter(|offset| offset % 4 == 0)
        .and_then(|offset| usize::try_from(offset / 4).ok())
        .and_then(|index| unit.instructions.get(index))
        .filter(|instruction| instruction.key.block_key().pc == guest.pc)
        .copied()
        .ok_or(Error::Internal(
            "LCQ canonical exit is absent from its instruction image",
        ))?;
    Ok((guest, instruction))
}

struct Dispatch<'a> {
    frame: *mut libc::c_void,
    lookup: FaultLookup<'a>,
    memory: &'a ExecutionMemory,
    arena: DirectAddressSpaceView,
    resolution: Option<Result<DirectFaultResolution, &'static str>>,
}

unsafe extern "C" fn dispatch_fault(
    opaque: *mut libc::c_void,
    captured: *mut CapturedFault,
) -> FaultDisposition {
    // Runs on the normal dispatcher stack, never in the signal handler.
    // An unwind cannot cross the native landing/retry assembly boundary.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dispatch = unsafe { &mut *opaque.cast::<Dispatch<'_>>() };
        let captured = unsafe { &*captured };
        let Some(fault) = dispatch.lookup.find(captured.native_pc()) else {
            return FaultDisposition::FatalUnattributed;
        };
        let resolution = unsafe {
            fault::access::resolve(
                &*dispatch.frame.cast::<NativeFrame<'_>>(),
                captured,
                &fault,
                dispatch.arena,
                dispatch.memory,
            )
        }
        .map(|(_, resolution)| resolution);
        if matches!(resolution, Ok(DirectFaultResolution::Retry)) {
            // Keep the captured image, lazy flags, FP and exclusive-load record
            // untouched. Shared capture rejects repeated unchanged repairs.
            FaultDisposition::Retry
        } else {
            dispatch.resolution = Some(resolution);
            FaultDisposition::Escape
        }
    }))
    .unwrap_or(FaultDisposition::FatalPanic)
}

struct Entry {
    frame: *mut libc::c_void,
    arena: *mut u8,
    result: Option<Result<NativeReturn, NativeReturnError>>,
}

unsafe extern "C" fn enter(opaque: *mut libc::c_void, entry: usize) {
    // Escape skips this Rust frame. It must own no values requiring Drop.
    let call = unsafe { &mut *opaque.cast::<Entry>() };
    call.result = Some(unsafe {
        native::enter_protected(
            &mut *call.frame.cast::<NativeFrame<'_>>(),
            call.arena,
            entry as *const u8,
        )
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        abi::{FpSpecialization, HostAbi, PollBudget},
        executable::Cache,
        lcq::{Compilation, compiler::Compiler},
        lifetime::{Lifetime, compile::Request},
    };
    use nixe_cpu::{
        memory::{MemoryPermissions, SyntheticMemory},
        platform::TargetPlatform,
        profile::ProcessCpuContext,
        state::a64::A64State,
    };
    use nixe_memory::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress};
    use std::sync::Arc;

    fn reader_with_breakpoint() -> (Reader, BlockKey) {
        let space = AddressSpaceId::new(1);
        let pc = GuestVirtualAddress::new(0x1000);
        let page = GuestPhysicalPageId::new(1);
        let key = BlockKey::new(
            ProcessCpuContext::new(TargetPlatform::Switch1, space),
            pc,
            FpSpecialization::Dynamic,
        )
        .unwrap();
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(page));
        assert!(memory.initialize_ram(page, 0, &0xd420_0000u32.to_le_bytes()));
        assert!(memory.map_page(space, pc, page, MemoryPermissions::READ_EXECUTE));
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
        let mut reader = process.register().unwrap();
        let Request::Owner(claim) = reader.claim(key).unwrap() else {
            panic!()
        };
        let abi = if cfg!(target_arch = "x86_64") {
            HostAbi::X86_64
        } else {
            HostAbi::Aarch64
        };
        Compiler::new(abi)
            .unwrap()
            .publish(
                Compilation::capture(claim, &memory).unwrap(),
                &process,
                &cache,
                &memory,
            )
            .unwrap();
        (reader, key)
    }

    #[test]
    fn canonical_exit_rejects_stale_versions_missing_units_and_invalid_map_indices() {
        let (mut reader, key) = reader_with_breakpoint();
        let mut state = A64State::default();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        let mut invocation = unsafe { reader.admit(&mut frame, key) }.unwrap().unwrap();
        let entry = invocation.payload().preferred().unwrap();
        let (frame, lookup) = invocation.frame_and_faults();
        frame.exit_source_version = entry.version.get();
        frame.exit_state_map = 0;
        assert_eq!(
            canonical_exit(&lookup, entry, frame).unwrap().1.bits,
            0xd420_0000
        );
        frame.exit_source_version += 1;
        assert!(matches!(
            canonical_exit(&lookup, entry, frame),
            Err(Error::Internal(
                "LCQ canonical exit has a different source version"
            ))
        ));
        frame.exit_source_version = entry.version.get();
        frame.exit_state_map = u32::MAX;
        assert!(matches!(
            canonical_exit(&lookup, entry, frame),
            Err(Error::Internal(
                "LCQ canonical exit has no guest exit record"
            ))
        ));
        frame.exit_state_map = 0;
        let missing = PublishedEntry {
            canonical: std::num::NonZeroUsize::new(1).unwrap(),
            ..entry
        };
        assert!(matches!(
            canonical_exit(&lookup, missing, frame),
            Err(Error::Internal("LCQ canonical entry has no live code unit"))
        ));
    }

    #[test]
    fn unattributed_fault_subprocess_entry() {
        if std::env::var_os("NIXE_LCQ_UNATTRIBUTED_FAULT").is_none() {
            return;
        }
        unsafe extern "C" fn unregistered_load(address: *mut libc::c_void, _: usize) {
            unsafe {
                address.cast::<u8>().read_volatile();
            }
        }
        let (mut reader, key) = reader_with_breakpoint();
        let arena = nixe_memory::DirectArena::new(0x4000).unwrap();
        let view = arena.view();
        let memory = ExecutionMemory::new();
        let mut state = A64State::default();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        let mut invocation = unsafe { reader.admit(&mut frame, key) }.unwrap().unwrap();
        let (frame, lookup) = invocation.frame_and_faults();
        let mut dispatcher = Dispatch {
            frame: std::ptr::from_mut(frame).cast(),
            lookup,
            memory: &memory,
            arena: view,
            resolution: None,
        };
        let mut worker = WorkerFaultContext::register().unwrap();
        // The arena address is valid for capture, but this Rust load has no
        // published native-PC metadata. Attribution must fail before reading
        // the frame or asking the deliberately unbound memory for resolution.
        let _ = unsafe {
            worker.invoke_captured(
                view,
                [frame.host_fp.saved_control, frame.host_fp.saved_status],
                dispatch_fault,
                std::ptr::from_mut(&mut dispatcher).cast(),
                NativeInvocation {
                    gateway: unregistered_load,
                    context: (view.base + 4096) as *mut libc::c_void,
                    entry: 0,
                },
            )
        };
        panic!("unattributed LCQ fault unexpectedly returned");
    }

    #[test]
    fn lcq_dispatcher_rejects_an_arena_access_without_native_pc_metadata() {
        use std::os::unix::process::ExitStatusExt;
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "lcq::invocation::tests::unattributed_fault_subprocess_entry",
                "--nocapture",
            ])
            .env("NIXE_LCQ_UNATTRIBUTED_FAULT", "1")
            .output()
            .unwrap();
        assert_eq!(output.status.signal(), Some(libc::SIGSEGV));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("reason=unattributed-native-pc"), "{stderr}");
        assert!(stderr.contains("native_pc=0x"), "{stderr}");
        assert!(stderr.contains("address=0x"), "{stderr}");
    }
}
