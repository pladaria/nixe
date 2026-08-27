mod compiler;
mod lookup;
mod region;
mod slow;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use nixe_cpu::decode::{self, DecodeResult};
use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::exclusive::ExclusiveMonitorState;
use nixe_cpu::execution::{
    ArchitecturalTimer, ControlRequest, CpuControl, CpuExit, CpuFault, CpuFaultKind,
    ExecutionReport, MemoryBinding, RunRequest, SchedulerRequest, VcpuEventState,
};
use nixe_cpu::location::{InstructionEncoding, LocationDescriptor};
use nixe_cpu::memory::{CodePageDependency, CpuMemory, DataAccessFault};
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State, Nzcv};
use nixe_memory::{
    GuestPhysicalPageId, GuestVirtualAddress, MemoryInvalidation, MemoryInvalidationCursor,
    MemoryInvalidationError, MemoryInvalidationKind, MemoryInvalidationSource,
};

use self::compiler::{DirectCompiler, NativeGateway};
use self::lookup::{NativeLookupSlot, RegionLookup};
use self::region::{RegionKey, RegionLimits, discover_region};

const DEFAULT_MAX_NATIVE_CODE_BYTES: usize = 64 * 1024 * 1024;

const EXIT_NONE: u32 = 0;
const EXIT_DISPATCH: u32 = 1;
const EXIT_BUDGET: u32 = 2;
const EXIT_CONTROL: u32 = 3;
const EXIT_ARCHITECTURAL: u32 = 4;
const EXIT_UNSUPPORTED: u32 = 5;
const EXIT_DATA_FAULT: u32 = 6;
const EXIT_SCHEDULED: u32 = 7;
const EXIT_INTERNAL: u32 = 8;
const EXIT_RECONCILE: u32 = 9;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectJitErrorKind {
    InvalidGuestCode,
    Unsupported,
    Capacity,
    Shutdown,
    Internal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectJitError {
    kind: DirectJitErrorKind,
    detail: Box<str>,
}

impl DirectJitError {
    fn invalid(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: DirectJitErrorKind::InvalidGuestCode,
            detail: detail.into(),
        }
    }

    fn unsupported(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: DirectJitErrorKind::Unsupported,
            detail: detail.into(),
        }
    }

    fn capacity(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: DirectJitErrorKind::Capacity,
            detail: detail.into(),
        }
    }

    fn internal(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: DirectJitErrorKind::Internal,
            detail: detail.into(),
        }
    }

    fn shutdown() -> Self {
        Self {
            kind: DirectJitErrorKind::Shutdown,
            detail: "direct JIT process is shut down".into(),
        }
    }
}

impl fmt::Display for DirectJitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for DirectJitError {}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DirectExit {
    Dispatch {
        pc: GuestVirtualAddress,
        instructions: u64,
    },
    Budget {
        pc: GuestVirtualAddress,
        instructions: u64,
    },
    Control {
        pc: GuestVirtualAddress,
        instructions: u64,
    },
    Architectural {
        pc: GuestVirtualAddress,
        detail: u32,
        instructions: u64,
    },
    Unsupported {
        pc: GuestVirtualAddress,
        instructions: u64,
    },
    DataFault {
        pc: GuestVirtualAddress,
        fault: DataAccessFault,
        instructions: u64,
    },
    Scheduled {
        pc: GuestVirtualAddress,
        request: SchedulerRequest,
        instructions: u64,
    },
    LoaderReturn {
        pc: GuestVirtualAddress,
        result_code: u64,
        instructions: u64,
    },
    Internal {
        pc: GuestVirtualAddress,
        instructions: u64,
    },
    Reconcile,
}

#[repr(C)]
struct NativeContext {
    state: *mut A64State,
    x: *mut u64,
    vector: *mut u128,
    sp: *mut u64,
    pc: *mut u64,
    nzcv: *mut Nzcv,
    fpcr: *mut u32,
    fpsr: *mut u32,
    tpidr_el0: *mut u64,
    tpidrro_el0: *mut u64,
    memory: *const dyn CpuMemory,
    exclusive: *mut ExclusiveMonitorState,
    timer: *const dyn ArchitecturalTimer,
    events: *const VcpuEventState,
    address_space: u64,
    fastmem_base: usize,
    fastmem_entries: usize,
    fastmem_size: usize,
    instruction_budget: u64,
    loader_return: u64,
    control_pending: usize,
    retired: u64,
    exit_pc: u64,
    exit_kind: u32,
    exit_detail: u32,
    slow_status: u32,
    slow_result_low: u64,
    slow_result_high: u64,
    slow_memory_calls: *const AtomicU64,
    native_lookup: *const NativeLookupSlot,
    invalidation_signal: *const AtomicU64,
    invalidation_cursor: u64,
    process_pending: *const AtomicU32,
    data_fault: Option<DataAccessFault>,
}

impl NativeContext {
    #[allow(clippy::too_many_arguments)]
    fn new(
        state: &mut A64State,
        memory: &dyn CpuMemory,
        exclusive: &mut ExclusiveMonitorState,
        timer: &dyn ArchitecturalTimer,
        events: &VcpuEventState,
        address_space: nixe_memory::AddressSpaceId,
        instruction_budget: u64,
        loader_return: Option<GuestVirtualAddress>,
        control: &CpuControl,
        slow_memory_calls: &AtomicU64,
        native_lookup: *const NativeLookupSlot,
        process_pending: &AtomicU32,
    ) -> Self {
        let fastmem = memory.fastmem_view(address_space);
        let state_pointer = std::ptr::from_mut(&mut *state);
        let x = state.general_register_storage_mut().as_mut_ptr();
        let vector = state.vector_register_storage_mut().as_mut_ptr();
        let sp = std::ptr::from_mut(state.stack_pointer_storage_mut());
        let pc = std::ptr::from_mut(state.program_counter_storage_mut());
        let nzcv = std::ptr::from_mut(state.nzcv_storage_mut());
        let fpcr = std::ptr::from_mut(state.fpcr_storage_mut());
        let fpsr = std::ptr::from_mut(state.fpsr_storage_mut());
        let tpidr_el0 = std::ptr::from_mut(state.tpidr_el0_storage_mut());
        let tpidrro_el0 = std::ptr::from_mut(state.tpidrro_el0_storage_mut());
        let invalidation_signal = memory.invalidation_signal();
        // Native code and every slow callback complete before `run` returns.
        // Erasing these borrow lifetimes keeps the generated ABI pointer-only;
        // no pointer escapes the stack-owned native context.
        let memory = unsafe { std::mem::transmute::<&dyn CpuMemory, *const dyn CpuMemory>(memory) };
        let timer = unsafe {
            std::mem::transmute::<&dyn ArchitecturalTimer, *const dyn ArchitecturalTimer>(timer)
        };
        Self {
            state: state_pointer,
            x,
            vector,
            sp,
            pc,
            nzcv,
            fpcr,
            fpsr,
            tpidr_el0,
            tpidrro_el0,
            memory,
            exclusive,
            timer,
            events,
            address_space: address_space.get(),
            fastmem_base: fastmem.map_or(0, |view| view.base),
            fastmem_entries: fastmem.map_or(0, |view| view.entries),
            fastmem_size: fastmem.map_or(0, |view| view.address_space_size),
            instruction_budget,
            loader_return: loader_return.map_or(u64::MAX, GuestVirtualAddress::get),
            control_pending: control.pending_word_address(),
            retired: 0,
            exit_pc: 0,
            exit_kind: EXIT_NONE,
            exit_detail: 0,
            slow_status: 0,
            slow_result_low: 0,
            slow_result_high: 0,
            slow_memory_calls,
            native_lookup,
            invalidation_signal,
            invalidation_cursor: MemoryInvalidationCursor::INITIAL.get(),
            process_pending,
            data_fault: None,
        }
    }

    fn exit(&self) -> Result<DirectExit, DirectJitError> {
        let pc = GuestVirtualAddress::new(self.exit_pc);
        let instructions = self.retired;
        match self.exit_kind {
            EXIT_DISPATCH => Ok(DirectExit::Dispatch { pc, instructions }),
            EXIT_BUDGET => Ok(DirectExit::Budget { pc, instructions }),
            EXIT_CONTROL => Ok(DirectExit::Control { pc, instructions }),
            EXIT_ARCHITECTURAL => Ok(DirectExit::Architectural {
                pc,
                detail: self.exit_detail,
                instructions,
            }),
            EXIT_UNSUPPORTED => Ok(DirectExit::Unsupported { pc, instructions }),
            EXIT_DATA_FAULT => Ok(DirectExit::DataFault {
                pc,
                fault: self.data_fault.clone().ok_or_else(|| {
                    DirectJitError::internal("direct JIT data-fault exit has no fault")
                })?,
                instructions,
            }),
            EXIT_SCHEDULED => {
                let request = match self.exit_detail {
                    0 => SchedulerRequest::Yield,
                    1 => SchedulerRequest::WaitForEvent,
                    2 => SchedulerRequest::WaitForInterrupt,
                    3 => SchedulerRequest::SendEvent,
                    detail => {
                        return Err(DirectJitError::internal(format!(
                            "direct JIT returned unknown scheduler request {detail}"
                        )));
                    }
                };
                Ok(DirectExit::Scheduled {
                    pc,
                    request,
                    instructions,
                })
            }
            EXIT_INTERNAL => Ok(DirectExit::Internal { pc, instructions }),
            EXIT_RECONCILE => Ok(DirectExit::Reconcile),
            kind => Err(DirectJitError::internal(format!(
                "direct JIT returned unknown native exit {kind}"
            ))),
        }
    }
}

struct StaticLink {
    target: RegionKey,
    slot: Arc<AtomicUsize>,
}

struct PublishedRegion {
    key: RegionKey,
    entry: usize,
    #[cfg(test)]
    native_bytes: usize,
    #[cfg(test)]
    clif_instructions: usize,
    guest_blocks: usize,
    #[cfg(test)]
    register_loads: usize,
    #[cfg(test)]
    register_stores: usize,
    dependencies: Box<[CodePageDependency]>,
    links: Box<[StaticLink]>,
}

struct ProcessState {
    compiler: DirectCompiler,
    lookup: RegionLookup,
    incoming: HashMap<RegionKey, Vec<Arc<AtomicUsize>>>,
    physical_dependencies: HashMap<GuestPhysicalPageId, HashSet<RegionKey>>,
    retired: Vec<Arc<PublishedRegion>>,
    invalidation_cursor: MemoryInvalidationCursor,
    invalidations: Vec<MemoryInvalidation>,
    native_bytes: usize,
    compiled_regions: usize,
    compiled_guest_blocks: usize,
}

pub struct JitProcess {
    cpu: ProcessCpuContext,
    limits: RegionLimits,
    max_native_code_bytes: usize,
    pending: AtomicU32,
    state: Mutex<ProcessState>,
}

impl JitProcess {
    pub fn new(cpu: ProcessCpuContext) -> Result<Self, DirectJitError> {
        Ok(Self {
            cpu,
            limits: RegionLimits::default(),
            max_native_code_bytes: DEFAULT_MAX_NATIVE_CODE_BYTES,
            pending: AtomicU32::new(0),
            state: Mutex::new(ProcessState {
                compiler: DirectCompiler::new()?,
                lookup: RegionLookup::new(),
                incoming: HashMap::new(),
                physical_dependencies: HashMap::new(),
                retired: Vec::new(),
                invalidation_cursor: MemoryInvalidationCursor::INITIAL,
                invalidations: Vec::new(),
                native_bytes: 0,
                compiled_regions: 0,
                compiled_guest_blocks: 0,
            }),
        })
    }

    fn entry_for(
        &self,
        memory: &(impl CpuMemory + ?Sized),
        location: LocationDescriptor,
    ) -> Result<(NativeGateway, usize, u64), DirectJitError> {
        let key = RegionKey::new(self.cpu, location);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        reconcile_invalidations(&mut state, memory, key.address_space)?;
        if self.pending.load(Ordering::Acquire) != 0 {
            return Err(DirectJitError::shutdown());
        }
        if let Some(region) = state.lookup.get(key) {
            return Ok((
                state.compiler.gateway(),
                region.entry,
                state.invalidation_cursor.get(),
            ));
        }

        let (region, link_targets, slots, compiled) = loop {
            let region = discover_region(self.cpu, memory, location, self.limits)?;
            let invalidated = reconcile_invalidations(&mut state, memory, key.address_space)?;
            if invalidated.affects(&region) {
                continue;
            }
            let link_targets: Vec<_> = region
                .external_exits
                .iter()
                .filter_map(|exit| exit.target.map(|target| key.at(target)))
                .collect();
            let slots: Vec<_> = link_targets
                .iter()
                .map(|_| Arc::new(AtomicUsize::new(0)))
                .collect();
            let slot_addresses: Vec<_> =
                slots.iter().map(|slot| Arc::as_ptr(slot).addr()).collect();
            let compiled = state.compiler.compile(&region, &slot_addresses)?;
            state.native_bytes = state.compiler.native_bytes();
            if state.native_bytes > self.max_native_code_bytes {
                return Err(DirectJitError::capacity(format!(
                    "direct JIT code arena exhausted: used={} limit={} regions={} guest_blocks={}",
                    state.native_bytes,
                    self.max_native_code_bytes,
                    state.compiled_regions,
                    state.compiled_guest_blocks,
                )));
            }
            let invalidated = reconcile_invalidations(&mut state, memory, key.address_space)?;
            if self.pending.load(Ordering::Acquire) != 0 {
                return Err(DirectJitError::shutdown());
            }
            if !invalidated.affects(&region) {
                break (region, link_targets, slots, compiled);
            }
        };
        let entry = compiled.entry;

        let links: Vec<_> = link_targets
            .into_iter()
            .zip(slots)
            .map(|(target, slot)| StaticLink { target, slot })
            .collect();
        let published = Arc::new(PublishedRegion {
            key,
            entry,
            #[cfg(test)]
            native_bytes: compiled.native_bytes,
            #[cfg(test)]
            clif_instructions: compiled.clif_instructions,
            guest_blocks: region.blocks.len(),
            #[cfg(test)]
            register_loads: compiled.register_loads,
            #[cfg(test)]
            register_stores: compiled.register_stores,
            dependencies: region.dependencies,
            links: links.into_boxed_slice(),
        });

        for link in &published.links {
            if let Some(target) = state.lookup.get(link.target) {
                link.slot.store(target.entry, Ordering::Release);
            }
            state
                .incoming
                .entry(link.target)
                .or_default()
                .push(Arc::clone(&link.slot));
        }
        for dependency in &published.dependencies {
            state
                .physical_dependencies
                .entry(dependency.page)
                .or_default()
                .insert(key);
        }
        if let Some(incoming) = state.incoming.get(&key) {
            for slot in incoming {
                slot.store(entry, Ordering::Release);
            }
        }
        state.compiled_regions += 1;
        state.compiled_guest_blocks += published.guest_blocks;
        state.lookup.insert(Arc::clone(&published));
        Ok((
            state.compiler.gateway(),
            published.entry,
            state.invalidation_cursor.get(),
        ))
    }

    fn reconcile(&self, memory: &(impl CpuMemory + ?Sized)) -> Result<(), DirectJitError> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        reconcile_invalidations(&mut state, memory, self.cpu.address_space_id())?;
        Ok(())
    }

    fn native_lookup(&self) -> *const NativeLookupSlot {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .lookup
            .native_base()
    }

    pub fn shutdown(&self) {
        self.pending.store(1, Ordering::Release);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let keys: Vec<_> = state.lookup.keys().collect();
        for key in keys {
            retire_region(&mut state, key);
        }
    }

    pub fn synchronize_address_space(&self, memory: &dyn CpuMemory) -> Result<(), DirectJitError> {
        self.reconcile(memory)
    }
}

#[derive(Default)]
struct InvalidationSummary {
    all: bool,
    physical_pages: HashSet<GuestPhysicalPageId>,
}

impl InvalidationSummary {
    fn affects(&self, region: &region::NativeRegion) -> bool {
        self.all
            || region
                .dependencies
                .iter()
                .any(|dependency| self.physical_pages.contains(&dependency.page))
    }
}

fn reconcile_invalidations(
    state: &mut ProcessState,
    memory: &(impl MemoryInvalidationSource + ?Sized),
    address_space: nixe_memory::AddressSpaceId,
) -> Result<InvalidationSummary, DirectJitError> {
    let mut records = std::mem::take(&mut state.invalidations);
    records.clear();
    let cursor = match memory.read_invalidations_since(state.invalidation_cursor, &mut records) {
        Ok(cursor) => cursor,
        Err(MemoryInvalidationError::HistoryLost { latest, .. }) => {
            state.invalidation_cursor = latest;
            state.invalidations = records;
            let keys: Vec<_> = state.lookup.keys().collect();
            for key in keys {
                retire_region(state, key);
            }
            return Ok(InvalidationSummary {
                all: true,
                physical_pages: HashSet::new(),
            });
        }
        Err(error) => {
            state.invalidations = records;
            return Err(DirectJitError::internal(format!(
                "direct JIT could not consume memory invalidations: {error}"
            )));
        }
    };
    state.invalidation_cursor = cursor;

    let mut summary = InvalidationSummary::default();
    for record in &records {
        match record.kind {
            MemoryInvalidationKind::Mapping {
                address_space: changed,
                ..
            }
            | MemoryInvalidationKind::InstructionCache {
                address_space: changed,
            } if changed == address_space => summary.all = true,
            MemoryInvalidationKind::ExecutableContent { first, second } => {
                summary.physical_pages.insert(first);
                summary.physical_pages.extend(second);
            }
            MemoryInvalidationKind::Mapping { .. }
            | MemoryInvalidationKind::InstructionCache { .. } => {}
        }
    }

    let keys: HashSet<_> = if summary.all {
        state.lookup.keys().collect()
    } else {
        summary
            .physical_pages
            .iter()
            .filter_map(|page| state.physical_dependencies.get(page))
            .flatten()
            .copied()
            .collect()
    };
    for key in keys {
        retire_region(state, key);
    }
    state.invalidations = records;
    Ok(summary)
}

fn retire_region(state: &mut ProcessState, key: RegionKey) {
    if let Some(incoming) = state.incoming.get(&key) {
        for slot in incoming {
            slot.store(0, Ordering::Release);
        }
    }
    let Some(region) = state.lookup.remove(key) else {
        return;
    };
    for dependency in &region.dependencies {
        let remove_page = state
            .physical_dependencies
            .get_mut(&dependency.page)
            .is_some_and(|keys| {
                keys.remove(&key);
                keys.is_empty()
            });
        if remove_page {
            state.physical_dependencies.remove(&dependency.page);
        }
    }
    for link in &region.links {
        link.slot.store(0, Ordering::Release);
        let remove_target = state.incoming.get_mut(&link.target).is_some_and(|slots| {
            slots.retain(|slot| !Arc::ptr_eq(slot, &link.slot));
            slots.is_empty()
        });
        if remove_target {
            state.incoming.remove(&link.target);
        }
    }
    state.retired.push(region);
}

pub struct JitThread {
    control: CpuControl,
    exclusive: Mutex<ExclusiveMonitorState>,
    #[cfg(test)]
    events: VcpuEventState,
    slow_memory_calls: AtomicU64,
    #[cfg(test)]
    rust_dispatches: AtomicU64,
}

impl Default for JitThread {
    fn default() -> Self {
        Self::new()
    }
}

struct NativeRunRequest<'a> {
    memory: &'a dyn CpuMemory,
    state: &'a mut A64State,
    instruction_budget: u64,
    loader_return: Option<GuestVirtualAddress>,
    timer: &'a dyn ArchitecturalTimer,
    events: &'a VcpuEventState,
}

impl JitThread {
    pub fn new() -> Self {
        Self {
            control: CpuControl::default(),
            exclusive: Mutex::new(ExclusiveMonitorState::default()),
            #[cfg(test)]
            events: VcpuEventState::default(),
            slow_memory_calls: AtomicU64::new(0),
            #[cfg(test)]
            rust_dispatches: AtomicU64::new(0),
        }
    }

    #[cfg(test)]
    fn request_preempt(&self) {
        self.control.request(ControlRequest::Preempt);
    }

    #[must_use]
    pub fn control(&self) -> CpuControl {
        self.control.clone()
    }

    pub fn clear_local_exclusive_reservation(&mut self) {
        *self
            .exclusive
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner) = Default::default();
    }

    pub fn synchronize_address_space(
        &mut self,
        process: &JitProcess,
        binding: MemoryBinding<'_>,
    ) -> Result<(), CpuFault> {
        process
            .synchronize_address_space(binding.memory)
            .map_err(|error| jit_fault(error, 0, &A64State::default()))?;
        self.control
            .acknowledge_invalidation(binding.invalidation_cursor.get());
        Ok(())
    }

    pub fn prepare_shutdown(
        &mut self,
        process: &JitProcess,
        binding: MemoryBinding<'_>,
    ) -> Result<(), CpuFault> {
        self.synchronize_address_space(process, binding)?;
        self.clear_local_exclusive_reservation();
        Ok(())
    }

    pub fn run_slice(
        &mut self,
        process: &JitProcess,
        request: RunRequest<'_>,
    ) -> Result<ExecutionReport, CpuFault> {
        let mut executed = 0;
        let mut remaining = request.instruction_budget;
        loop {
            let pending_interrupts = request.events.take_pending_interrupts();
            if pending_interrupts != 0 {
                return Ok(report(
                    executed,
                    CpuExit::PendingEvent {
                        mask: pending_interrupts,
                    },
                    request.state,
                ));
            }
            let exit = self
                .run_with_runtime(
                    process,
                    NativeRunRequest {
                        memory: request.memory,
                        state: request.state,
                        instruction_budget: remaining,
                        loader_return: request.loader_return,
                        timer: request.timer,
                        events: &request.events,
                    },
                )
                .map_err(|error| jit_fault(error, executed, request.state))?;
            let retired = direct_exit_instructions(&exit);
            executed = executed.saturating_add(retired);
            remaining = remaining.saturating_sub(retired);
            let source = |pc| LocationDescriptor::new(pc, process.cpu.profile_id());
            let stop = match exit {
                DirectExit::Dispatch { .. } => continue,
                DirectExit::Budget { .. } => CpuExit::BudgetExhausted,
                DirectExit::Control { .. } => {
                    let Some(control) = self.control.take_pending() else {
                        return Err(internal_fault(
                            "native control exit has no pending request",
                            executed,
                            request.state,
                        ));
                    };
                    self.control.acknowledge(control);
                    if control.contains(ControlRequest::Preempt) {
                        CpuExit::Safepoint
                    } else {
                        continue;
                    }
                }
                DirectExit::Architectural { pc, detail, .. } => {
                    let class = detail >> 24;
                    let immediate = detail & 0x00ff_ffff;
                    match class {
                        1 => CpuExit::SupervisorCall {
                            source: source(pc),
                            immediate,
                        },
                        2 => CpuExit::ArchitecturalException {
                            source: source(pc),
                            kind: ExceptionKind::Breakpoint,
                            syndrome: Some(u64::from(immediate)),
                        },
                        6 => CpuExit::ArchitecturalException {
                            source: source(pc),
                            kind: ExceptionKind::FloatingPoint,
                            syndrome: Some(u64::from(immediate)),
                        },
                        _ => {
                            return Err(internal_fault(
                                format!("native exit has unknown architectural class {class}"),
                                executed,
                                request.state,
                            ));
                        }
                    }
                }
                DirectExit::Unsupported { pc, .. } => {
                    unsupported_exit(process, request.memory, pc, executed, request.state)?
                }
                DirectExit::DataFault { pc, fault, .. } => CpuExit::DataFault {
                    source: source(pc),
                    fault,
                },
                DirectExit::Scheduled { pc, request, .. } => CpuExit::Scheduled {
                    source: source(pc),
                    request,
                },
                DirectExit::LoaderReturn {
                    pc, result_code, ..
                } => CpuExit::LoaderReturn {
                    source: source(pc),
                    result_code,
                },
                DirectExit::Internal { .. } => {
                    return Err(internal_fault(
                        "native execution returned an internal exit",
                        executed,
                        request.state,
                    ));
                }
                DirectExit::Reconcile => unreachable!("reconciliation is handled internally"),
            };
            return Ok(report(executed, stop, request.state));
        }
    }

    #[cfg(test)]
    fn run(
        &self,
        process: &JitProcess,
        memory: &dyn CpuMemory,
        state: &mut A64State,
        instruction_budget: u64,
    ) -> Result<DirectExit, DirectJitError> {
        self.run_with_runtime(
            process,
            NativeRunRequest {
                memory,
                state,
                instruction_budget,
                loader_return: None,
                timer: &ZeroTimer,
                events: &self.events,
            },
        )
    }

    fn run_with_runtime(
        &self,
        process: &JitProcess,
        request: NativeRunRequest<'_>,
    ) -> Result<DirectExit, DirectJitError> {
        let NativeRunRequest {
            memory,
            state,
            instruction_budget,
            loader_return,
            timer,
            events,
        } = request;
        let mut exclusive = self
            .exclusive
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let native_lookup = process.native_lookup();
        let mut context = NativeContext::new(
            state,
            memory,
            &mut exclusive,
            timer,
            events,
            process.cpu.address_space_id(),
            instruction_budget,
            loader_return,
            &self.control,
            &self.slow_memory_calls,
            native_lookup,
            &process.pending,
        );
        loop {
            let pc = GuestVirtualAddress::new(unsafe { *context.pc });
            if pc.get() == context.loader_return {
                return Ok(DirectExit::LoaderReturn {
                    pc,
                    result_code: state.read_x(A64Register::General(
                        A64GeneralRegister::new(0).expect("valid result register"),
                    )),
                    instructions: context.retired,
                });
            }
            let location = LocationDescriptor::new(pc, process.cpu.profile_id());
            let (gateway, entry, invalidation_cursor) = process.entry_for(memory, location)?;
            context.invalidation_cursor = invalidation_cursor;
            unsafe { gateway(&mut context, entry) };
            let exit = context.exit()?;
            if matches!(exit, DirectExit::Reconcile) {
                process.reconcile(memory)?;
                continue;
            }
            if matches!(exit, DirectExit::Dispatch { .. }) {
                #[cfg(test)]
                self.rust_dispatches.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            return Ok(exit);
        }
    }
}

fn direct_exit_instructions(exit: &DirectExit) -> u64 {
    match exit {
        DirectExit::Dispatch { instructions, .. }
        | DirectExit::Budget { instructions, .. }
        | DirectExit::Control { instructions, .. }
        | DirectExit::Architectural { instructions, .. }
        | DirectExit::Unsupported { instructions, .. }
        | DirectExit::DataFault { instructions, .. }
        | DirectExit::Scheduled { instructions, .. }
        | DirectExit::Internal { instructions, .. }
        | DirectExit::LoaderReturn { instructions, .. } => *instructions,
        DirectExit::Reconcile => 0,
    }
}

fn report(instructions_executed: u64, stop: CpuExit, state: &A64State) -> ExecutionReport {
    ExecutionReport {
        instructions_executed,
        stop,
        context: state.register_context(),
    }
}

fn jit_fault(error: DirectJitError, instructions_executed: u64, state: &A64State) -> CpuFault {
    let kind = match error.kind {
        DirectJitErrorKind::InvalidGuestCode => CpuFaultKind::InvalidRequest,
        DirectJitErrorKind::Unsupported
        | DirectJitErrorKind::Capacity
        | DirectJitErrorKind::Shutdown => CpuFaultKind::Unavailable,
        DirectJitErrorKind::Internal => CpuFaultKind::Internal,
    };
    CpuFault {
        backend: "jit",
        kind,
        instructions_executed,
        message: error.detail,
        context: Box::new(state.register_context()),
    }
}

fn internal_fault(
    message: impl Into<Box<str>>,
    instructions_executed: u64,
    state: &A64State,
) -> CpuFault {
    CpuFault {
        backend: "jit",
        kind: CpuFaultKind::Internal,
        instructions_executed,
        message: message.into(),
        context: Box::new(state.register_context()),
    }
}

fn unsupported_exit(
    process: &JitProcess,
    memory: &dyn CpuMemory,
    pc: GuestVirtualAddress,
    instructions_executed: u64,
    state: &A64State,
) -> Result<CpuExit, CpuFault> {
    let source = LocationDescriptor::new(pc, process.cpu.profile_id());
    let fetched = memory
        .fetch32(process.cpu.address_space_id(), pc)
        .map_err(|fault| internal_fault(fault.to_string(), instructions_executed, state))?;
    let encoding = InstructionEncoding::from_u32(fetched.bits);
    match decode::decode(process.cpu.decoder(), source, encoding) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
            Ok(CpuExit::UnsupportedSemantics {
                source,
                encoding,
                disassembly: decode::disassemble(&decoded.instruction).to_string().into(),
                coverage_id: decoded.instruction.coverage_id(),
            })
        }
        DecodeResult::Unallocated { reason, .. } => Err(internal_fault(
            format!("native unsupported exit decoded as unallocated: {reason}"),
            instructions_executed,
            state,
        )),
        DecodeResult::Reserved { name, reason, .. } => Err(internal_fault(
            format!("native unsupported exit decoded as reserved {name}: {reason}"),
            instructions_executed,
            state,
        )),
    }
}

#[cfg(test)]
struct ZeroTimer;

#[cfg(test)]
impl ArchitecturalTimer for ZeroTimer {
    fn snapshot(&self) -> nixe_cpu::execution::TimerSnapshot {
        nixe_cpu::execution::TimerSnapshot {
            counter: 0,
            frequency: 0,
        }
    }
}

#[cfg(test)]
mod tests;
