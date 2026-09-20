//! One admitted native invocation. No borrowed code or captured machine image
//! escapes this boundary; typed memory completion runs after it returns.

use super::fault::{self, cold::Completion};
use crate::{
    abi::{BlockKey, ExclusiveStoreOperation, NativeFrame},
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
        // The canonical loop consumes control exits after owned completion;
        // it receives the reconciled PollBudget separately.
        returned: NativeReturn,
        guest: GuestExit,
        /// The exiting instruction, not the destination PC's current bytes.
        instruction: Instruction,
        completion_sample: Option<lifetime::unit::CompletionSample>,
    },
    Memory {
        instruction: Instruction,
        outcome: MemoryExit,
        completion_sample: Option<lifetime::unit::CompletionSample>,
    },
}

impl Exit {
    pub(crate) fn completion_sample(&self) -> Option<lifetime::unit::CompletionSample> {
        match self {
            Self::Native {
                completion_sample, ..
            }
            | Self::Memory {
                completion_sample, ..
            } => *completion_sample,
        }
    }
}

pub(crate) enum MemoryExit {
    /// Failed native proof; the original CIVAC has not executed yet.
    CacheCleanInvalidate {
        address: nixe_memory::GuestVirtualAddress,
    },
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
/// LCQ fragments with contiguous instruction images; they use
/// this host's checked NativeFrame ABI and this memory's arena size. No other
/// FP owner is active on this OS thread. The canonical state matches `key`,
/// and the frame has no pending exclusive load from an earlier invocation.
/// Production static edges use the Closed linker as their targets become
/// resident; indirect hits use the owning vCPU's PIC. Every target in the chain
/// remains rooted through this same epoch. Misses resume canonical ingress.
/// The caller must not resume guest execution after an internal/runtime error.
pub(crate) unsafe fn run(
    samples: &mut crate::sampling::Samples,
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
        dispatch_error: None,
        samples,
        observation_status: 0,
    };
    frame.dispatch_resolver = Some(dispatch_link);
    frame.sample_observer = Some(observe_sample);
    frame.dispatch_context = std::ptr::from_mut(&mut dispatch).cast();
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
    };
    // The frame may outlive this stack-owned dispatcher, including fault and
    // runtime error paths. No borrowed callback pointer survives the call.
    frame.dispatch_resolver = None;
    frame.sample_observer = None;
    frame.dispatch_context = std::ptr::null_mut();
    let returned = returned.map_err(Error::Runtime)?;
    // An observer failure retained its hardware contribution under the caller
    // environment. Its native failure adapter has now written mapped software
    // FPSR; only now can we merge without that writeback erasing sticky flags.
    if dispatch.observation_status != 0 {
        unsafe { *frame.canonical.fpsr |= dispatch.observation_status };
    }
    if let Some(error) = dispatch.dispatch_error.take() {
        return Err(error);
    }
    let exit = match returned {
        InvocationOutcome::Returned => {
            let returned = call
                .result
                .ok_or(Error::Internal("LCQ gateway returned without an exit"))?
                .map_err(Error::Native)?;
            let (guest, instruction) = canonical_exit(&dispatch.lookup, frame)?;
            let terminal = matches!(
                guest.kind,
                EdgeKind::Static
                    | EdgeKind::Taken
                    | EdgeKind::NotTaken
                    | EdgeKind::Call
                    | EdgeKind::Indirect
                    | EdgeKind::Return
                    | EdgeKind::FragmentLimit
            );
            if returned.poll.sample {
                let unit = dispatch
                    .lookup
                    .unit(frame.exit_native_pc)
                    .ok_or(Error::Internal("LCQ sampled exit has no live code unit"))?;
                let edge = terminal.then(|| crate::sampling::ObservedEdge {
                    destination: nixe_memory::GuestVirtualAddress::new(unsafe {
                        *frame.canonical.pc
                    }),
                    kind: guest.kind,
                });
                dispatch
                    .lookup
                    .sample_lcq(unit, dispatch.samples, edge)
                    .map_err(Error::Lifetime)?;
            }
            // Only a successful one-instruction completion can cross this
            // deadline. Do not look up identity on every canonical exit.
            let completion_sample = if !terminal
                && frame.budget.sample_remaining == 1
                && returned.reason != crate::abi::NativeExitReason::Control
            {
                let unit = dispatch
                    .lookup
                    .unit(frame.exit_native_pc)
                    .ok_or(Error::Internal("LCQ completion sample has no live source"))?;
                dispatch
                    .lookup
                    .completion_sample(unit)
                    .map_err(Error::Lifetime)?
            } else {
                None
            };
            if let EdgeKind::ExclusiveStore(operation) = guest.kind {
                Exit::Memory {
                    instruction,
                    outcome: MemoryExit::ExclusiveStore(operation),
                    completion_sample,
                }
            } else {
                Exit::Native {
                    returned,
                    guest,
                    instruction,
                    completion_sample,
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
            let (access, resolution) = dispatch
                .resolution
                .take()
                .ok_or(Error::Internal("LCQ escape has no memory resolution"))?
                .map_err(Error::Internal)?;
            let outcome = match resolution {
                DirectFaultResolution::Cold
                    if fault.record.access == lifetime::unit::Access::CacheProbe =>
                {
                    MemoryExit::CacheCleanInvalidate {
                        address: access.address,
                    }
                }
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
            if poll.sample {
                dispatch
                    .lookup
                    .sample_lcq(fault.unit, dispatch.samples, None)
                    .map_err(Error::Lifetime)?;
            }
            let completion_sample = if frame.budget.sample_remaining == 1
                && matches!(
                    &outcome,
                    MemoryExit::Cold(_)
                        | MemoryExit::CacheCleanInvalidate { .. }
                        | MemoryExit::ExclusiveStore(_)
                ) {
                dispatch
                    .lookup
                    .completion_sample(fault.unit)
                    .map_err(Error::Lifetime)?
            } else {
                None
            };
            Exit::Memory {
                instruction,
                outcome,
                completion_sample,
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

/// Resolve the actual exiting unit while the invocation still protects its
/// metadata. The guest destination and initial entry need not belong to it.
/// Never perform a fresh dispatch lookup which could select a replacement.
fn canonical_exit(
    lookup: &FaultLookup<'_>,
    frame: &NativeFrame<'_>,
) -> Result<(GuestExit, Instruction), Error> {
    let unit = lookup
        .unit(frame.exit_native_pc)
        .ok_or(Error::Internal("LCQ canonical exit has no live code unit"))?;
    if frame.exit_source_version != unit.version.get() {
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
    resolution: Option<Result<(fault::access::Access, DirectFaultResolution), &'static str>>,
    dispatch_error: Option<Error>,
    samples: &'a mut crate::sampling::Samples,
    observation_status: u32,
}

unsafe extern "C" fn observe_sample(
    opaque: *mut libc::c_void,
    frame: *mut libc::c_void,
    native_pc: usize,
    version: u64,
    state_map: u32,
) -> u32 {
    let frame = unsafe { &mut *frame.cast::<NativeFrame<'_>>() };
    // Only the bounded FP leaf may precede caller restoration. The emitted
    // adapter already saved caller-clobbered values and the actual destination.
    let pause = unsafe { frame.host_fp.pause_observation() };
    let dispatch = unsafe { &mut *opaque.cast::<Dispatch<'_>>() };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let unit = dispatch
            .lookup
            .unit(native_pc)
            .ok_or(Error::Internal("sample poll has no protected source"))?;
        if unit.version.get() != version {
            return Err(Error::Internal("sample poll source version mismatch"));
        }
        let map = unit
            .states
            .get(state_map as usize)
            .ok_or(Error::Internal("sample poll has no source state map"))?;
        let guest = map
            .exit
            .ok_or(Error::Internal("sample poll has no guest exit"))?;
        if map
            .transfer
            .as_ref()
            .and_then(|transfer| transfer.poll_offset)
            .is_none()
        {
            return Err(Error::Internal("sample poll is not a charged terminal"));
        }
        let destination = unsafe {
            frame
                .spill
                .as_ptr()
                .byte_add(crate::native::observation::DESTINATION as usize)
                .cast::<u64>()
                .read()
        };
        dispatch
            .lookup
            .sample_lcq(
                unit,
                dispatch.samples,
                Some(crate::sampling::ObservedEdge {
                    destination: nixe_memory::GuestVirtualAddress::new(destination),
                    kind: guest.kind,
                }),
            )
            .map_err(Error::Lifetime)
    }))
    .unwrap_or(Err(Error::Internal("panic in sample poll observer")));
    match result {
        Ok(()) => {
            // All lookup guards and observer temporaries are gone. No general
            // Rust work follows restoration of the interrupted guest FP image.
            unsafe { pause.resume() };
            1
        }
        Err(error) => {
            dispatch.observation_status = pause.abort();
            dispatch.dispatch_error = Some(error);
            0
        }
    }
}

unsafe extern "C" fn dispatch_link(
    opaque: *mut libc::c_void,
    frame: *mut libc::c_void,
    remaining: i64,
) -> usize {
    let frame = unsafe { &mut *frame.cast::<NativeFrame<'_>>() };
    // Canonical source writeback precedes this helper. Collect host status only
    // afterwards, so mapped software FPSR cannot overwrite it. No general Rust
    // (locking, unwinding or lookup) may run with the guest FP environment.
    unsafe { frame.suspend_fp() };
    let dispatch = unsafe { &mut *opaque.cast::<Dispatch<'_>>() };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        use crate::abi::NativeExitReason;
        use std::sync::atomic::Ordering;
        if frame
            .poll_requests
            .iter()
            .any(|word| unsafe { &**word }.load(Ordering::Acquire) != 0)
        {
            frame.exit_reason = NativeExitReason::Control as u32;
            return Ok(0);
        }
        let spent = frame
            .budget
            .armed_span
            .checked_sub(remaining)
            .ok_or(Error::Internal("link fallback work accounting overflow"))?;
        let left = frame
            .budget
            .slice_remaining
            .checked_sub(spent)
            .ok_or(Error::Internal("link fallback slice accounting overflow"))?;
        if left <= 0 {
            return Ok(0);
        }
        let unit = dispatch
            .lookup
            .unit(frame.exit_native_pc)
            .ok_or(Error::Internal("link fallback has no live source"))?;
        if frame.exit_source_version != unit.version.get() {
            return Err(Error::Internal("link fallback source version mismatch"));
        }
        let map = unit
            .states
            .get(frame.exit_state_map as usize)
            .ok_or(Error::Internal("link fallback has no source state map"))?;
        let transfer = map
            .transfer
            .as_ref()
            .ok_or(Error::Internal("link fallback has no terminal transfer"))?;
        if frame.exit_reason != NativeExitReason::Dispatch as u32 {
            return Err(Error::Internal("link fallback destination mismatch"));
        }
        let resolved = if let Some(target) = transfer.static_target {
            if target.pc.get() != frame.exit_pc {
                return Err(Error::Internal("link fallback destination mismatch"));
            }
            dispatch.lookup.static_entry(target)
        } else {
            if !map.exit.is_some_and(|exit| {
                matches!(
                    exit.kind,
                    EdgeKind::Indirect | EdgeKind::Call | EdgeKind::Return
                )
            }) {
                return Err(Error::Internal("link fallback is not an indirect terminal"));
            }
            let Some(target) = unit.instructions[0]
                .key
                .block_key()
                .at(nixe_memory::GuestVirtualAddress::new(frame.exit_pc))
            else {
                // Preserve the guest destination for ordinary canonical fault
                // handling; a misaligned address can never become a PIC key.
                return Ok(0);
            };
            let source = unit
                .registered_handle()
                .ok_or(Error::Internal("link fallback has no source registration"))?;
            // Canonical writeback and FP suspension precede this exclusive
            // borrow. No native execution/retry resumes until it is dropped.
            unsafe { dispatch.lookup.suspend_native() }.resolve_bridge(
                source,
                frame.exit_state_map,
                target,
            )
        };
        match resolved {
            Ok(entry) => Ok(entry.map_or(0, |entry| entry.canonical.get())),
            Err(
                lifetime::Error::Closed
                | lifetime::Error::Shutdown
                | lifetime::Error::StaleUnit
                | lifetime::Error::StalePublication,
            ) => {
                frame.exit_reason = NativeExitReason::Control as u32;
                Ok(0)
            }
            Err(error) => Err(Error::Lifetime(error)),
        }
    }))
    .unwrap_or(Err(Error::Internal("panic in link fallback resolver")));
    match result {
        Ok(address) => {
            if address != 0 && unsafe { frame.resume_fp() }.is_err() {
                dispatch.dispatch_error = Some(Error::Internal("link fallback cannot resume FP"));
                return 0;
            }
            // No Rust work with destructors follows successful FP resumption.
            address
        }
        Err(error) => {
            dispatch.dispatch_error = Some(error);
            0
        }
    }
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
        };
        if matches!(resolution, Ok((_, DirectFaultResolution::Retry))) {
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

    fn reader_with_breakpoints() -> (Reader, [BlockKey; 2]) {
        reader_with_words(&[0xd420_0000, 0xd420_0020])
    }

    fn reader_with_words(words: &[u32]) -> (Reader, [BlockKey; 2]) {
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
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        assert!(memory.initialize_ram(page, 0, &bytes));
        assert!(memory.map_page(space, pc, page, MemoryPermissions::READ_EXECUTE));
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
        let mut reader = process.register().unwrap();
        let abi = if cfg!(target_arch = "x86_64") {
            HostAbi::X86_64
        } else {
            HostAbi::Aarch64
        };
        let keys = [key, key.at(GuestVirtualAddress::new(0x1004)).unwrap()];
        let mut compiler = Compiler::new(abi).unwrap();
        for key in keys {
            let Request::Owner(claim) = reader.claim(key).unwrap() else {
                panic!()
            };
            compiler
                .publish(
                    Compilation::capture(claim, &memory).unwrap(),
                    &process,
                    &cache,
                    &memory,
                )
                .unwrap();
            process.try_service_links().unwrap();
        }
        (reader, keys)
    }

    #[test]
    fn sample_observer_failure_restores_source_and_defers_fp_merge_until_writeback() {
        unsafe extern "C" fn wrong_version(
            opaque: *mut libc::c_void,
            frame: *mut libc::c_void,
            pc: usize,
            version: u64,
            map: u32,
        ) -> u32 {
            // Deliberately corrupt only the immutable identity passed to the
            // real observer, not its native code or protected state maps.
            unsafe { observe_sample(opaque, frame, pc, version.wrapping_add(1), map) }
        }
        let _restore = crate::fp_env::tests::RestoreHost::new();
        let caller = crate::fp_env::tests::distinct_caller();
        // FADD D0,D1,D2; ADDS X0,X0,#1; B start. A failed observer must
        // canonicalize this source, never start the next loop iteration.
        let (mut reader, [key, _]) = reader_with_words(&[0x1e622820, 0xb1000400, 0x17fffffe]);
        let mut state = A64State::default();
        state.set_pc(key.pc.get());
        state.set_fpsr(1 << 27);
        state.set_vector(1, u128::from(1.0f64.to_bits()));
        state.set_vector(2, u128::from(2.0f64.powi(-53).to_bits()));
        let memory = ExecutionMemory::new();
        let arena = nixe_memory::DirectArena::new(0x4000).unwrap();
        let mut samples = crate::sampling::Samples::new();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(2, 20).unwrap());
        let mut invocation = unsafe { reader.admit(&mut frame, key) }.unwrap().unwrap();
        let entry = invocation.payload().preferred().unwrap();
        let (frame, lookup) = invocation.frame_and_faults();
        let epoch = frame.execution_epoch;
        let mut dispatch = Dispatch {
            frame: std::ptr::from_mut(frame).cast(),
            lookup,
            memory: &memory,
            arena: arena.view(),
            resolution: None,
            dispatch_error: None,
            samples: &mut samples,
            observation_status: 0,
        };
        frame.sample_observer = Some(wrong_version);
        frame.dispatch_context = std::ptr::from_mut(&mut dispatch).cast();
        let returned = unsafe {
            native::enter_protected(
                frame,
                arena.view().base as *mut u8,
                entry.canonical.get() as *const u8,
            )
        }
        .unwrap();
        frame.sample_observer = None;
        frame.dispatch_context = std::ptr::null_mut();
        assert!(matches!(
            dispatch.dispatch_error,
            Some(Error::Internal("sample poll source version mismatch"))
        ));
        assert_eq!(returned.reason, crate::abi::NativeExitReason::Control);
        assert!(!returned.poll.sample && !returned.poll.exhausted);
        assert_eq!(frame.execution_epoch, epoch);
        assert_ne!(epoch, 0);
        assert_eq!(frame.budget.slice_remaining, 17);
        assert_eq!(frame.budget.sample_remaining, 4095);
        assert_eq!(frame.exit_source_version, entry.version.get());
        assert_eq!(dispatch.observation_status, 1 << 4);
        assert_eq!(frame.host_fp.active, 0);
        assert_eq!(unsafe { *frame.canonical.x }, 1);
        assert_eq!(unsafe { *frame.canonical.pc }, key.pc.get());
        assert_eq!(unsafe { *frame.canonical.fpsr }, 1 << 27);
        // Same post-writeback merge as run(), under the caller environment.
        unsafe {
            *frame.canonical.fpsr |= dispatch.observation_status;
        }
        assert_eq!(unsafe { *frame.canonical.fpsr }, (1 << 27) | (1 << 4));
        assert!(dispatch.samples.seed_snapshot(key).is_none());
        let mut host = crate::abi::HostFpState::default();
        unsafe {
            host.begin();
            host.finish();
        }
        assert_eq!((host.saved_control, host.saved_status), caller);
    }

    #[test]
    fn canonical_exit_rejects_stale_versions_missing_units_and_invalid_map_indices() {
        let (mut reader, [key, _]) = reader_with_breakpoints();
        let mut state = A64State::default();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        let mut invocation = unsafe { reader.admit(&mut frame, key) }.unwrap().unwrap();
        let entry = invocation.payload().preferred().unwrap();
        let (frame, lookup) = invocation.frame_and_faults();
        frame.exit_native_pc = entry.canonical.get();
        frame.exit_source_version = entry.version.get();
        frame.exit_state_map = 0;
        assert_eq!(canonical_exit(&lookup, frame).unwrap().1.bits, 0xd420_0000);
        frame.exit_source_version += 1;
        assert!(matches!(
            canonical_exit(&lookup, frame),
            Err(Error::Internal(
                "LCQ canonical exit has a different source version"
            ))
        ));
        frame.exit_source_version = entry.version.get();
        frame.exit_state_map = u32::MAX;
        assert!(matches!(
            canonical_exit(&lookup, frame),
            Err(Error::Internal(
                "LCQ canonical exit has no guest exit record"
            ))
        ));
        frame.exit_state_map = 0;
        frame.exit_native_pc = 1;
        assert!(matches!(
            canonical_exit(&lookup, frame),
            Err(Error::Internal("LCQ canonical exit has no live code unit"))
        ));
    }

    #[test]
    fn canonical_exit_resolves_the_executed_unit_not_the_admitted_entry() {
        let (mut reader, [first, second]) = reader_with_breakpoints();
        let mut state = A64State::default();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        // No retirement or mutation runs in this fixture. The admission below
        // protects both published units when the captured second address is used.
        let second_entry = {
            let invocation = unsafe { reader.admit(&mut frame, second) }
                .unwrap()
                .unwrap();
            invocation.payload().preferred().unwrap()
        };
        let mut invocation = unsafe { reader.admit(&mut frame, first) }.unwrap().unwrap();
        let first_entry = invocation.payload().preferred().unwrap();
        assert_ne!(first_entry.version, second_entry.version);
        let (frame, lookup) = invocation.frame_and_faults();
        let epoch = frame.execution_epoch;
        // Exercise final-unit attribution independently of edge publication:
        // enter the second real LCQ unit under the first unit's admission.
        unsafe {
            native::enter_protected(
                frame,
                std::ptr::null_mut(),
                second_entry.canonical.get() as *const u8,
            )
        }
        .unwrap();
        assert_eq!(frame.execution_epoch, epoch);
        assert_ne!(epoch, 0);
        assert_eq!(frame.exit_source_version, second_entry.version.get());
        assert_eq!(
            lookup.unit(frame.exit_native_pc).unwrap().id,
            second_entry.unit
        );
        let (guest, instruction) = canonical_exit(&lookup, frame).unwrap();
        assert_eq!(guest.pc, second.pc);
        assert_eq!(instruction.bits, 0xd420_0020);
        // A mapped address alone must not allow an unrelated version/map pair.
        frame.exit_source_version = first_entry.version.get();
        assert!(matches!(
            canonical_exit(&lookup, frame),
            Err(Error::Internal(
                "LCQ canonical exit has a different source version"
            ))
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
        let (mut reader, [key, _]) = reader_with_breakpoints();
        let arena = nixe_memory::DirectArena::new(0x4000).unwrap();
        let view = arena.view();
        let memory = ExecutionMemory::new();
        let mut state = A64State::default();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        let mut invocation = unsafe { reader.admit(&mut frame, key) }.unwrap().unwrap();
        let (frame, lookup) = invocation.frame_and_faults();
        let mut samples = crate::sampling::Samples::new();
        let mut dispatcher = Dispatch {
            frame: std::ptr::from_mut(frame).cast(),
            lookup,
            memory: &memory,
            arena: view,
            resolution: None,
            dispatch_error: None,
            samples: &mut samples,
            observation_status: 0,
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
