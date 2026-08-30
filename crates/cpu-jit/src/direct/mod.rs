mod compiler;
mod lookup;
mod region;
mod slow;

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
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
use nixe_cpu::memory::{CodePageDependency, CodePageSpan, CpuMemory, DataAccessFault};
use nixe_cpu::memory::{
    DataAccessKind, DirectFaultResolution, MemoryAccess, MemoryAccessClass, MemoryAccessSize,
    MemoryAlignment, MemoryOrdering, MemoryValue,
};
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State, Nzcv};
use nixe_cpu_direct_memory::{
    CapturedFault, FaultDisposition, InvocationOutcome, NativeFaultCompletion, NativeFaultRegion,
    NativeFaultRegistry, NativeFaultSite, NativeInvocation, NativeMemoryAccessKind,
    WorkerFaultContext,
};
use nixe_memory::{
    CpuMemoryBackend, DirectAddressSpaceView, GuestPhysicalPageId, GuestVirtualAddress,
    MemoryInvalidation, MemoryInvalidationCursor, MemoryInvalidationError, MemoryInvalidationKind,
    MemoryInvalidationSource,
};

use self::compiler::{DirectCompiler, NativeGateway};
use self::lookup::{NativeLookupSlot, RegionLookup};
use self::region::{RegionKey, RegionLimits, discover_region};

const DEFAULT_MAX_NATIVE_CODE_BYTES: usize = 1024 * 1024 * 1024;
const MAX_NATIVE_FAULT_REGIONS: usize = 262_144;

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
    direct_base: usize,
    direct_size: usize,
    direct_store_controls: usize,
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
    native_lookup: *const NativeLookupSlot,
    invalidation_signal: *const AtomicU64,
    invalidation_cursor: u64,
    process_pending: *const AtomicU32,
    data_fault: Option<DataAccessFault>,
    direct_fault_error: Option<Box<str>>,
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
        native_lookup: *const NativeLookupSlot,
        process_pending: &AtomicU32,
    ) -> Self {
        let direct = memory.direct_address_space_view(address_space);
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
            direct_base: direct.map_or(0, |view| view.base),
            direct_size: direct.map_or(0, |view| view.address_space_size),
            direct_store_controls: direct.map_or(0, |view| view.store_controls),
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
            native_lookup,
            invalidation_signal,
            invalidation_cursor: MemoryInvalidationCursor::INITIAL.get(),
            process_pending,
            data_fault: None,
            direct_fault_error: None,
        }
    }

    fn exit(&mut self) -> Result<DirectExit, DirectJitError> {
        if let Some(detail) = self.direct_fault_error.take() {
            return Err(DirectJitError::internal(detail));
        }
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
    entry_keys: Box<[RegionKey]>,
    entry: usize,
    #[cfg(test)]
    native_bytes: usize,
    #[cfg(test)]
    clif_instructions: usize,
    #[cfg(test)]
    register_loads: usize,
    #[cfg(test)]
    register_stores: usize,
    dependencies: Box<[CodePageDependency]>,
    mapping_dependencies: Box<[CodePageSpan]>,
    links: Box<[StaticLink]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BoundMemoryBackend {
    address_space: nixe_memory::AddressSpaceId,
    end_exclusive: GuestVirtualAddress,
    backend: CpuMemoryBackend,
    direct: Option<DirectAddressSpaceView>,
}

struct ProcessState {
    compiler: DirectCompiler,
    lookup: RegionLookup,
    regions: HashMap<RegionKey, Arc<PublishedRegion>>,
    incoming: HashMap<RegionKey, Vec<Arc<AtomicUsize>>>,
    physical_dependencies: HashMap<GuestPhysicalPageId, HashSet<RegionKey>>,
    retired: Vec<Arc<PublishedRegion>>,
    invalidation_cursor: MemoryInvalidationCursor,
    invalidations: Vec<MemoryInvalidation>,
    native_bytes: usize,
    memory_backend: Option<BoundMemoryBackend>,
}

impl ProcessState {
    #[cfg(test)]
    fn region_for(&self, key: RegionKey) -> Option<&Arc<PublishedRegion>> {
        let owner = self.lookup.get(key)?.owner;
        self.regions.get(&owner)
    }
}

pub struct JitProcess {
    cpu: ProcessCpuContext,
    limits: RegionLimits,
    max_native_code_bytes: usize,
    pending: AtomicU32,
    fault_registry: Arc<NativeFaultRegistry>,
    state: Mutex<ProcessState>,
}

impl JitProcess {
    pub fn new(cpu: ProcessCpuContext) -> Result<Self, DirectJitError> {
        let fault_registry = Arc::new(
            NativeFaultRegistry::with_capacity(MAX_NATIVE_FAULT_REGIONS)
                .map_err(|error| DirectJitError::unsupported(error.to_string()))?,
        );
        Ok(Self {
            cpu,
            limits: RegionLimits::default(),
            max_native_code_bytes: DEFAULT_MAX_NATIVE_CODE_BYTES,
            pending: AtomicU32::new(0),
            fault_registry,
            state: Mutex::new(ProcessState {
                compiler: DirectCompiler::new()?,
                lookup: RegionLookup::new(),
                regions: HashMap::new(),
                incoming: HashMap::new(),
                physical_dependencies: HashMap::new(),
                retired: Vec::new(),
                invalidation_cursor: MemoryInvalidationCursor::INITIAL,
                invalidations: Vec::new(),
                native_bytes: 0,
                memory_backend: None,
            }),
        })
    }

    pub fn bind_memory(&self, binding: MemoryBinding<'_>) -> Result<(), DirectJitError> {
        if binding.address_space != self.cpu.address_space_id() {
            return Err(DirectJitError::invalid(
                "JIT memory binding uses a different process address space",
            ));
        }
        let backend = binding.memory.cpu_memory_backend(binding.address_space);
        let direct = match backend {
            CpuMemoryBackend::Checked => None,
            CpuMemoryBackend::LinuxDirect => {
                nixe_cpu_direct_memory::install()
                    .map_err(|error| DirectJitError::unsupported(error.to_string()))?;
                let view = binding
                    .memory
                    .direct_address_space_view(binding.address_space)
                    .ok_or_else(|| {
                        DirectJitError::internal(
                            "LinuxDirect memory binding has no direct address-space view",
                        )
                    })?;
                if view.address_space_size as u64 != binding.end_exclusive.get() {
                    return Err(DirectJitError::invalid(
                        "direct arena size differs from the process address space",
                    ));
                }
                Some(view)
            }
        };
        let requested = BoundMemoryBackend {
            address_space: binding.address_space,
            end_exclusive: binding.end_exclusive,
            backend,
            direct,
        };
        {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(bound) = state.memory_backend {
                if bound != requested {
                    return Err(DirectJitError::invalid(
                        "JIT process memory backend is immutable after binding",
                    ));
                }
            } else {
                state.compiler.bind_memory_backend(backend)?;
                state.memory_backend = Some(requested);
            }
        }
        Ok(())
    }

    fn bound_memory_backend(&self) -> Result<BoundMemoryBackend, DirectJitError> {
        Ok(self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .memory_backend
            .unwrap_or(BoundMemoryBackend {
                address_space: self.cpu.address_space_id(),
                end_exclusive: GuestVirtualAddress::new(0),
                backend: CpuMemoryBackend::Checked,
                direct: None,
            }))
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
        if let Some(entry) = state.lookup.get(key) {
            return Ok((
                state.compiler.gateway(),
                entry.entry,
                state.invalidation_cursor.get(),
            ));
        }
        let (region, link_targets, slots, mut compiled) = loop {
            let region = discover_region(self.cpu, memory, location, self.limits, |pc| {
                state.lookup.get(key.at(pc)).is_some()
            })?;
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
                    "direct JIT code arena exhausted: used={} limit={}",
                    state.native_bytes, self.max_native_code_bytes,
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
        let entry_keys: Vec<_> = region
            .blocks
            .iter()
            .map(|block| key.at(block.start.pc))
            .collect();
        let links: Vec<_> = link_targets
            .into_iter()
            .zip(slots)
            .map(|(target, slot)| StaticLink { target, slot })
            .collect();
        if !compiled.fault_sites.is_empty() {
            let native_end = entry.checked_add(compiled.native_bytes).ok_or_else(|| {
                DirectJitError::internal("compiled native region address overflows")
            })?;
            let sites = std::mem::take(&mut compiled.fault_sites)
                .into_vec()
                .into_iter()
                .map(|site| {
                    let native_start =
                        entry
                            .checked_add(site.native_start as usize)
                            .ok_or_else(|| {
                                DirectJitError::internal("native fault-site start overflows")
                            })?;
                    let native_end =
                        entry.checked_add(site.native_end as usize).ok_or_else(|| {
                            DirectJitError::internal("native fault-site end overflows")
                        })?;
                    Ok(NativeFaultSite {
                        native_start,
                        native_end,
                        access: site.access,
                        completion: site.completion,
                        guest_address: Some(site.guest_address),
                        retired_delta: site.retired_delta,
                    })
                })
                .collect::<Result<Vec<_>, DirectJitError>>()?;
            let metadata = Arc::new(NativeFaultRegion {
                native_start: entry,
                native_end,
                sites: Arc::from(sites),
            });
            self.fault_registry
                .publish(Arc::clone(&metadata))
                .map_err(|error| DirectJitError::internal(error.to_string()))?;
        }
        let published = Arc::new(PublishedRegion {
            key,
            entry_keys: entry_keys.into_boxed_slice(),
            entry,
            #[cfg(test)]
            native_bytes: compiled.native_bytes,
            #[cfg(test)]
            clif_instructions: compiled.clif_instructions,
            #[cfg(test)]
            register_loads: compiled.register_loads,
            #[cfg(test)]
            register_stores: compiled.register_stores,
            dependencies: region.dependencies,
            mapping_dependencies: region.mapping_dependencies,
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
        let previous = state.regions.insert(key, Arc::clone(&published));
        assert!(
            previous.is_none(),
            "a native region owner is published once"
        );
        for &entry_key in &published.entry_keys {
            state.lookup.insert(entry_key, key, entry);
        }
        for &entry_key in &published.entry_keys {
            if let Some(incoming) = state.incoming.get(&entry_key) {
                for slot in incoming {
                    slot.store(entry, Ordering::Release);
                }
            }
        }
        Ok((
            state.compiler.gateway(),
            published.entry,
            state.invalidation_cursor.get(),
        ))
    }

    /// Reconciles invalidations and returns an already-published entry without
    /// compiling. Callers use this after acquiring the short native-execution
    /// lease; a missing entry means that a transition invalidated the region
    /// between compilation and lease acquisition, so compilation is retried
    /// outside the lease.
    fn published_entry_for(
        &self,
        memory: &(impl CpuMemory + ?Sized),
        location: LocationDescriptor,
    ) -> Result<Option<(NativeGateway, usize, u64)>, DirectJitError> {
        let key = RegionKey::new(self.cpu, location);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        reconcile_invalidations(&mut state, memory, key.address_space)?;
        if self.pending.load(Ordering::Acquire) != 0 {
            return Err(DirectJitError::shutdown());
        }
        Ok(state.lookup.get(key).map(|entry| {
            (
                state.compiler.gateway(),
                entry.entry,
                state.invalidation_cursor.get(),
            )
        }))
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
        let keys: Vec<_> = state.regions.keys().copied().collect();
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
    mapping_ranges: Vec<(GuestVirtualAddress, u64)>,
}

impl InvalidationSummary {
    fn affects(&self, region: &region::NativeRegion) -> bool {
        self.all
            || self.mapping_ranges.iter().any(|&(start, size)| {
                region
                    .mapping_dependencies
                    .iter()
                    .any(|span| mapping_range_overlaps(*span, start, size))
            })
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
            let keys: Vec<_> = state.regions.keys().copied().collect();
            for key in keys {
                retire_region(state, key);
            }
            return Ok(InvalidationSummary {
                all: true,
                ..InvalidationSummary::default()
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
                start,
                size,
            } if changed == address_space => summary.mapping_ranges.push((start, size)),
            MemoryInvalidationKind::InstructionCache {
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

    let mut keys: HashSet<_> = if summary.all {
        state.regions.keys().copied().collect()
    } else {
        summary
            .physical_pages
            .iter()
            .filter_map(|page| state.physical_dependencies.get(page))
            .flatten()
            .copied()
            .collect()
    };
    if !summary.all && !summary.mapping_ranges.is_empty() {
        keys.extend(state.regions.values().filter_map(|region| {
            summary
                .mapping_ranges
                .iter()
                .any(|&(start, size)| {
                    region
                        .mapping_dependencies
                        .iter()
                        .any(|span| mapping_range_overlaps(*span, start, size))
                })
                .then_some(region.key)
        }));
    }
    for key in keys {
        retire_region(state, key);
    }
    state.invalidations = records;
    Ok(summary)
}

fn mapping_range_overlaps(span: CodePageSpan, start: GuestVirtualAddress, size: u64) -> bool {
    if size == 0 {
        return false;
    }
    let changed_start = u128::from(start.get());
    let changed_end = (changed_start + u128::from(size)).min(1_u128 << 64);
    let span_start = u128::from(span.start.get());
    let span_end = span
        .end_exclusive
        .map_or(1_u128 << 64, |end| u128::from(end.get()));
    changed_start < span_end && span_start < changed_end
}

fn retire_region(state: &mut ProcessState, key: RegionKey) {
    let owner = state.lookup.get(key).map_or(key, |entry| entry.owner);
    let Some(region) = state.regions.remove(&owner) else {
        return;
    };
    for &entry_key in &region.entry_keys {
        if let Some(incoming) = state.incoming.get(&entry_key) {
            for slot in incoming {
                slot.store(0, Ordering::Release);
            }
        }
        state.lookup.remove(entry_key);
    }
    for dependency in &region.dependencies {
        let remove_page = state
            .physical_dependencies
            .get_mut(&dependency.page)
            .is_some_and(|keys| {
                keys.remove(&owner);
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
    fault_context: Mutex<Option<WorkerFaultContext>>,
    #[cfg(test)]
    events: VcpuEventState,
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
            fault_context: Mutex::new(None),
            #[cfg(test)]
            events: VcpuEventState::default(),
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
            .bind_memory(binding)
            .map_err(|error| jit_fault(error, 0, &A64State::default()))?;
        process
            .synchronize_address_space(binding.memory)
            .map_err(|error| jit_fault(error, 0, &A64State::default()))?;
        let backend = process
            .bound_memory_backend()
            .map_err(|error| jit_fault(error, 0, &A64State::default()))?;
        if backend.backend == CpuMemoryBackend::LinuxDirect {
            let context = self
                .fault_context
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner);
            if context.is_none() {
                *context = Some(WorkerFaultContext::register().map_err(|error| {
                    jit_fault(
                        DirectJitError::unsupported(error.to_string()),
                        0,
                        &A64State::default(),
                    )
                })?);
            }
        }
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
        self.fault_context
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        Ok(())
    }

    pub fn run_slice(
        &mut self,
        process: &JitProcess,
        mut request: RunRequest<'_>,
    ) -> Result<ExecutionReport, CpuFault> {
        let backend = process
            .bound_memory_backend()
            .map_err(|error| jit_fault(error, 0, request.state))?;
        if backend.backend == CpuMemoryBackend::LinuxDirect
            && !request
                .memory_lease
                .as_ref()
                .is_some_and(|lease| lease.authorizes(request.memory))
        {
            return Err(jit_fault(
                DirectJitError::invalid(
                    "LinuxDirect JIT execution requires its live mapping lease",
                ),
                0,
                request.state,
            ));
        }
        // The caller lease proves the public binding, but retaining it while
        // discovering and compiling a region can stall unrelated GPU or
        // mapping transitions for the complete compile. Native execution
        // acquires its own fresh lease after the compiled entry has been
        // revalidated against pending invalidations.
        if backend.backend == CpuMemoryBackend::LinuxDirect {
            drop(request.memory_lease.take());
        }
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
        let backend = process.bound_memory_backend()?;
        let actual_backend = memory.cpu_memory_backend(process.cpu.address_space_id());
        let actual_direct = memory.direct_address_space_view(process.cpu.address_space_id());
        if actual_backend != backend.backend || actual_direct != backend.direct {
            return Err(DirectJitError::invalid(
                "JIT execution memory differs from its immutable process binding",
            ));
        }
        let mut fault_context = self
            .fault_context
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if backend.backend == CpuMemoryBackend::LinuxDirect && fault_context.is_none() {
            return Err(DirectJitError::internal(
                "direct JIT worker entered native code without a fault context",
            ));
        }
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
            let compiled_entry = process.entry_for(memory, location)?;
            let native_lease = if backend.backend == CpuMemoryBackend::LinuxDirect {
                let lease = memory.acquire_execution_lease().ok_or_else(|| {
                    DirectJitError::invalid(
                        "LinuxDirect memory cannot acquire its native execution lease",
                    )
                })?;
                if !lease.authorizes(memory) {
                    return Err(DirectJitError::invalid(
                        "LinuxDirect memory acquired a lease from another owner",
                    ));
                }
                Some(lease)
            } else {
                None
            };
            let (gateway, entry, invalidation_cursor) =
                if backend.backend == CpuMemoryBackend::LinuxDirect {
                    let Some(entry) = process.published_entry_for(memory, location)? else {
                        drop(native_lease);
                        continue;
                    };
                    entry
                } else {
                    compiled_entry
                };
            context.invalidation_cursor = invalidation_cursor;
            if let Some(arena) = backend.direct {
                let worker = fault_context
                    .as_mut()
                    .expect("direct backend retains a registered fault context");
                let gateway = unsafe {
                    std::mem::transmute::<NativeGateway, nixe_cpu_direct_memory::NativeGateway>(
                        gateway,
                    )
                };
                let outcome = unsafe {
                    worker.invoke(
                        arena,
                        &process.fault_registry,
                        dispatch_direct_fault,
                        std::ptr::from_mut(&mut context).cast(),
                        NativeInvocation {
                            gateway,
                            context: std::ptr::from_mut(&mut context).cast(),
                            entry,
                        },
                    )
                }
                .map_err(|error| DirectJitError::internal(error.to_string()))?;
                if outcome == InvocationOutcome::Retry {
                    return Err(DirectJitError::internal(
                        "direct JIT gateway unexpectedly requested caller retry",
                    ));
                }
            } else {
                unsafe { gateway(&mut context, entry) };
            }
            drop(native_lease);
            let exit = context.exit()?;
            if matches!(exit, DirectExit::Reconcile) {
                process.reconcile(memory)?;
                continue;
            }
            if matches!(exit, DirectExit::Dispatch { .. }) {
                continue;
            }
            return Ok(exit);
        }
    }
}

unsafe extern "C" fn dispatch_direct_fault(
    opaque: *mut c_void,
    fault: *mut CapturedFault,
) -> FaultDisposition {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        dispatch_direct_fault_inner(&mut *opaque.cast::<NativeContext>(), &*fault)
    }))
    .unwrap_or(FaultDisposition::Fatal)
}

unsafe fn dispatch_direct_fault_inner(
    context: &mut NativeContext,
    fault: &CapturedFault,
) -> FaultDisposition {
    let site = fault.site();
    if site.access.address_space.get() != context.address_space {
        return FaultDisposition::Fatal;
    }
    let Some(guest_address_register) = site.guest_address else {
        return FaultDisposition::Fatal;
    };
    let Ok(guest_address) = fault.read_host_integer(guest_address_register) else {
        return FaultDisposition::Fatal;
    };
    let Some(size) = direct_access_size(site.access.size) else {
        return FaultDisposition::Fatal;
    };
    let kind = match site.access.kind {
        NativeMemoryAccessKind::Read => DataAccessKind::Read,
        NativeMemoryAccessKind::Write => DataAccessKind::Write,
    };
    let memory = unsafe { &*context.memory };
    let address = GuestVirtualAddress::new(guest_address);
    let access = MemoryAccess::new(
        size,
        MemoryAlignment::Unaligned,
        MemoryOrdering::Relaxed,
        MemoryAccessClass::Normal,
    );
    match memory.resolve_direct_fault(site.access.address_space, address, size, kind) {
        DirectFaultResolution::Retry => {
            #[cfg(target_arch = "x86_64")]
            {
                FaultDisposition::Retry
            }
            #[cfg(target_arch = "aarch64")]
            {
                // AArch64 glibc does not restore all volatile host state from
                // setcontext. Re-enter the faulting guest checkpoint instead;
                // this path is cold and leaves generated memory accesses bare.
                let pre_retired = direct_fault_retired(context, fault);
                unsafe { &mut *context.state }.set_pc(site.access.guest_pc.get());
                context.retired = pre_retired;
                context.exit_pc = site.access.guest_pc.get();
                context.exit_kind = EXIT_DISPATCH;
                context.exit_detail = 0;
                context.data_fault = None;
                context.direct_fault_error = None;
                FaultDisposition::Escape
            }
        }
        DirectFaultResolution::Checked => {
            let pre_retired = direct_fault_retired(context, fault);
            match kind {
                DataAccessKind::Read => {
                    match site.completion {
                        NativeFaultCompletion::IntegerPairLoad { .. } => {
                            if !complete_direct_pair_read(
                                context,
                                memory,
                                site,
                                address,
                                size,
                                access,
                                pre_retired,
                            ) {
                                return FaultDisposition::Fatal;
                            }
                            return FaultDisposition::Escape;
                        }
                        NativeFaultCompletion::VectorPairLoad { .. } => {
                            if !complete_direct_vector_pair_read(
                                context,
                                memory,
                                site,
                                address,
                                size,
                                access,
                                pre_retired,
                            ) {
                                return FaultDisposition::Fatal;
                            }
                            return FaultDisposition::Escape;
                        }
                        _ => {}
                    }
                    match memory.read(site.access.address_space, address, access) {
                        Ok(result) => {
                            if !complete_direct_read(
                                context,
                                site.completion,
                                size,
                                result.value.bits(),
                            ) {
                                return FaultDisposition::Fatal;
                            }
                            let next = site.access.guest_pc.wrapping_offset(4);
                            unsafe { &mut *context.state }.set_pc(next.get());
                            context.retired = pre_retired.saturating_add(1);
                            context.exit_pc = next.get();
                            context.exit_kind = EXIT_DISPATCH;
                            context.exit_detail = 0;
                            context.data_fault = None;
                        }
                        Err(data_fault) => {
                            unsafe { &mut *context.state }.set_pc(site.access.guest_pc.get());
                            context.retired = pre_retired.saturating_add(1);
                            context.exit_pc = site.access.guest_pc.get();
                            context.exit_kind = EXIT_DATA_FAULT;
                            context.exit_detail = 0;
                            context.data_fault = Some(data_fault);
                        }
                    }
                }
                DataAccessKind::Write => {
                    let Some(value) = direct_store_value(context, site.completion, size) else {
                        return FaultDisposition::Fatal;
                    };
                    match memory.complete_direct_write_fault(
                        site.access.address_space,
                        address,
                        access,
                        value,
                    ) {
                        Ok(_) => {
                            let next = site.access.guest_pc.wrapping_offset(4);
                            unsafe { &mut *context.state }.set_pc(next.get());
                            context.retired = pre_retired.saturating_add(1);
                            context.exit_pc = next.get();
                            context.exit_kind = EXIT_DISPATCH;
                            context.exit_detail = 0;
                            context.data_fault = None;
                        }
                        Err(data_fault) => {
                            unsafe { &mut *context.state }.set_pc(site.access.guest_pc.get());
                            context.retired = pre_retired.saturating_add(1);
                            context.exit_pc = site.access.guest_pc.get();
                            context.exit_kind = EXIT_DATA_FAULT;
                            context.exit_detail = 0;
                            context.data_fault = Some(data_fault);
                        }
                    }
                }
            }
            FaultDisposition::Escape
        }
        DirectFaultResolution::Fatal(detail) => {
            let pre_retired = direct_fault_retired(context, fault);
            unsafe { &mut *context.state }.set_pc(site.access.guest_pc.get());
            context.retired = pre_retired;
            context.exit_pc = site.access.guest_pc.get();
            context.exit_kind = EXIT_INTERNAL;
            context.direct_fault_error = Some(detail);
            FaultDisposition::Escape
        }
    }
}

fn direct_fault_retired(context: &NativeContext, fault: &CapturedFault) -> u64 {
    context
        .retired
        .saturating_add(u64::from(fault.site().retired_delta))
}

fn complete_direct_read(
    context: &mut NativeContext,
    completion: NativeFaultCompletion,
    size: MemoryAccessSize,
    bits: u128,
) -> bool {
    match completion {
        NativeFaultCompletion::IntegerLoad {
            register,
            signed,
            destination_bits,
        } => {
            let Some(value) = integer_load_value(size, bits, signed, destination_bits) else {
                return false;
            };
            write_integer_result(unsafe { &mut *context.state }, register, value)
        }
        NativeFaultCompletion::VectorLoad { register } => {
            unsafe { &mut *context.state }.set_vector(register, bits)
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn complete_direct_pair_read(
    context: &mut NativeContext,
    memory: &dyn CpuMemory,
    site: &NativeFaultSite,
    fault_address: GuestVirtualAddress,
    size: MemoryAccessSize,
    access: MemoryAccess,
    pre_retired: u64,
) -> bool {
    let NativeFaultCompletion::IntegerPairLoad {
        first_register,
        second_register,
        signed,
        destination_bits,
        access_index,
        writeback_register,
        writeback_offset,
        writeback,
    } = site.completion
    else {
        return false;
    };
    if access_index > 1 || first_register > 31 || second_register > 31 || writeback_register > 31 {
        return false;
    }
    let displacement = u64::from(access_index) * size.bytes() as u64;
    let Some(first_address) = fault_address.get().checked_sub(displacement) else {
        return false;
    };
    let Some(second_address) = first_address.checked_add(size.bytes() as u64) else {
        return false;
    };
    let first_address = GuestVirtualAddress::new(first_address);
    let second_address = GuestVirtualAddress::new(second_address);
    let first = match memory.read(site.access.address_space, first_address, access) {
        Ok(result) => result.value.bits(),
        Err(data_fault) => {
            finish_checked_direct_fault(context, site.access.guest_pc, pre_retired, data_fault);
            return true;
        }
    };
    let second = match memory.read(site.access.address_space, second_address, access) {
        Ok(result) => result.value.bits(),
        Err(data_fault) => {
            finish_checked_direct_fault(context, site.access.guest_pc, pre_retired, data_fault);
            return true;
        }
    };
    let Some(first) = integer_load_value(size, first, signed, destination_bits) else {
        return false;
    };
    let Some(second) = integer_load_value(size, second, signed, destination_bits) else {
        return false;
    };
    let state = unsafe { &mut *context.state };
    let writeback = if writeback {
        let register = if writeback_register == 31 {
            A64Register::StackPointer
        } else {
            let Some(register) = A64GeneralRegister::new(writeback_register) else {
                return false;
            };
            A64Register::General(register)
        };
        Some((
            register,
            state
                .read_x(register)
                .wrapping_add_signed(i64::from(writeback_offset)),
        ))
    } else {
        None
    };
    if !write_integer_result(state, first_register, first)
        || !write_integer_result(state, second_register, second)
    {
        return false;
    }
    if let Some((register, value)) = writeback {
        state.write_x(register, value);
    }
    finish_checked_direct_success(context, site.access.guest_pc, pre_retired);
    true
}

#[allow(clippy::too_many_arguments)]
fn complete_direct_vector_pair_read(
    context: &mut NativeContext,
    memory: &dyn CpuMemory,
    site: &NativeFaultSite,
    fault_address: GuestVirtualAddress,
    size: MemoryAccessSize,
    access: MemoryAccess,
    pre_retired: u64,
) -> bool {
    let NativeFaultCompletion::VectorPairLoad {
        first_register,
        second_register,
        access_index,
        writeback_register,
        writeback_offset,
        writeback,
    } = site.completion
    else {
        return false;
    };
    if access_index > 1 || first_register > 31 || second_register > 31 || writeback_register > 31 {
        return false;
    }
    let displacement = u64::from(access_index) * size.bytes() as u64;
    let Some(first_address) = fault_address.get().checked_sub(displacement) else {
        return false;
    };
    let Some(second_address) = first_address.checked_add(size.bytes() as u64) else {
        return false;
    };
    let first = match memory.read(
        site.access.address_space,
        GuestVirtualAddress::new(first_address),
        access,
    ) {
        Ok(result) => result.value.bits(),
        Err(data_fault) => {
            finish_checked_direct_fault(context, site.access.guest_pc, pre_retired, data_fault);
            return true;
        }
    };
    let second = match memory.read(
        site.access.address_space,
        GuestVirtualAddress::new(second_address),
        access,
    ) {
        Ok(result) => result.value.bits(),
        Err(data_fault) => {
            finish_checked_direct_fault(context, site.access.guest_pc, pre_retired, data_fault);
            return true;
        }
    };
    let state = unsafe { &mut *context.state };
    let writeback = if writeback {
        let register = if writeback_register == 31 {
            A64Register::StackPointer
        } else {
            let Some(register) = A64GeneralRegister::new(writeback_register) else {
                return false;
            };
            A64Register::General(register)
        };
        Some((
            register,
            state
                .read_x(register)
                .wrapping_add_signed(i64::from(writeback_offset)),
        ))
    } else {
        None
    };
    if !state.set_vector(first_register, first) || !state.set_vector(second_register, second) {
        return false;
    }
    if let Some((register, value)) = writeback {
        state.write_x(register, value);
    }
    finish_checked_direct_success(context, site.access.guest_pc, pre_retired);
    true
}

fn integer_load_value(
    size: MemoryAccessSize,
    bits: u128,
    signed: bool,
    destination_bits: u8,
) -> Option<u64> {
    if !matches!(destination_bits, 32 | 64) || size == MemoryAccessSize::Quadword {
        return None;
    }
    let source_bits = (size.bytes() * 8) as u32;
    let value = if signed {
        let shift = 64 - source_bits;
        (((bits as u64) << shift) as i64 >> shift) as u64
    } else {
        bits as u64
    };
    Some(if destination_bits == 32 {
        u64::from(value as u32)
    } else {
        value
    })
}

fn write_integer_result(state: &mut A64State, register: u8, value: u64) -> bool {
    if register == 31 {
        return true;
    }
    let Some(register) = A64GeneralRegister::new(register) else {
        return false;
    };
    state.write_x(A64Register::General(register), value);
    true
}

fn finish_checked_direct_success(
    context: &mut NativeContext,
    source: GuestVirtualAddress,
    pre_retired: u64,
) {
    let next = source.wrapping_offset(4);
    unsafe { &mut *context.state }.set_pc(next.get());
    context.retired = pre_retired.saturating_add(1);
    context.exit_pc = next.get();
    context.exit_kind = EXIT_DISPATCH;
    context.exit_detail = 0;
    context.data_fault = None;
}

fn finish_checked_direct_fault(
    context: &mut NativeContext,
    source: GuestVirtualAddress,
    pre_retired: u64,
    data_fault: DataAccessFault,
) {
    unsafe { &mut *context.state }.set_pc(source.get());
    context.retired = pre_retired.saturating_add(1);
    context.exit_pc = source.get();
    context.exit_kind = EXIT_DATA_FAULT;
    context.exit_detail = 0;
    context.data_fault = Some(data_fault);
}

fn direct_store_value(
    context: &NativeContext,
    completion: NativeFaultCompletion,
    size: MemoryAccessSize,
) -> Option<MemoryValue> {
    let bits = match completion {
        NativeFaultCompletion::IntegerStore { register } => {
            if register == 31 {
                0
            } else {
                let register = A64GeneralRegister::new(register)?;
                u128::from(unsafe { &*context.state }.read_x(A64Register::General(register)))
            }
        }
        NativeFaultCompletion::VectorStore { register } => {
            unsafe { &*context.state }.vector(register)?
        }
        _ => return None,
    };
    Some(MemoryValue::from_bits(size, bits))
}

const fn direct_access_size(bytes: u8) -> Option<MemoryAccessSize> {
    match bytes {
        1 => Some(MemoryAccessSize::Byte),
        2 => Some(MemoryAccessSize::Halfword),
        4 => Some(MemoryAccessSize::Word),
        8 => Some(MemoryAccessSize::Doubleword),
        16 => Some(MemoryAccessSize::Quadword),
        _ => None,
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
