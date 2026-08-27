mod compiler;
mod lookup;
mod region;
mod slow;

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use nixe_cpu::exclusive::ExclusiveMonitorState;
use nixe_cpu::execution::{
    ArchitecturalTimer, ControlRequest, CpuControl, SchedulerRequest, TimerSnapshot, VcpuEventState,
};
use nixe_cpu::location::{ExecutionState, LocationDescriptor};
use nixe_cpu::memory::{CodePageDependency, CpuMemory, DataAccessFault, InstructionMemory};
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64State, Nzcv};
use nixe_memory::GuestVirtualAddress;

use self::compiler::{CompiledRegion, DirectCompiler, NativeGateway};
use self::lookup::RegionLookup;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectJitErrorKind {
    InvalidGuestCode,
    Unsupported,
    Capacity,
    Internal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DirectJitError {
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
    Internal {
        pc: GuestVirtualAddress,
        instructions: u64,
    },
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
    control_pending: usize,
    retired: u64,
    exit_pc: u64,
    exit_kind: u32,
    exit_detail: u32,
    slow_status: u32,
    slow_result_low: u64,
    slow_result_high: u64,
    slow_memory_calls: *const AtomicU64,
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
        control: &CpuControl,
        slow_memory_calls: &AtomicU64,
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
            control_pending: control.pending_word_address(),
            retired: 0,
            exit_pc: 0,
            exit_kind: EXIT_NONE,
            exit_detail: 0,
            slow_status: 0,
            slow_result_low: 0,
            slow_result_high: 0,
            slow_memory_calls,
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
    native_bytes: usize,
    clif_instructions: usize,
    guest_blocks: usize,
    register_loads: usize,
    register_stores: usize,
    dependencies: Box<[CodePageDependency]>,
    links: Box<[StaticLink]>,
}

struct ProcessState {
    compiler: DirectCompiler,
    lookup: RegionLookup,
    incoming: HashMap<RegionKey, Vec<Arc<AtomicUsize>>>,
    native_bytes: usize,
    compiled_regions: usize,
    compiled_guest_blocks: usize,
}

struct JitProcess {
    cpu: ProcessCpuContext,
    limits: RegionLimits,
    max_native_code_bytes: usize,
    state: Mutex<ProcessState>,
}

impl JitProcess {
    fn new(cpu: ProcessCpuContext) -> Result<Self, DirectJitError> {
        Ok(Self {
            cpu,
            limits: RegionLimits::default(),
            max_native_code_bytes: DEFAULT_MAX_NATIVE_CODE_BYTES,
            state: Mutex::new(ProcessState {
                compiler: DirectCompiler::new()?,
                lookup: RegionLookup::new(),
                incoming: HashMap::new(),
                native_bytes: 0,
                compiled_regions: 0,
                compiled_guest_blocks: 0,
            }),
        })
    }

    fn entry_for(
        &self,
        memory: &(impl InstructionMemory + ?Sized),
        location: LocationDescriptor,
    ) -> Result<(NativeGateway, usize), DirectJitError> {
        let key = RegionKey::new(self.cpu, location);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(region) = state.lookup.get(key) {
            return Ok((state.compiler.gateway(), region.entry));
        }

        let region = discover_region(self.cpu, memory, location, self.limits)?;
        let link_targets: Vec<_> = region
            .external_exits
            .iter()
            .filter_map(|exit| exit.target.map(|target| key.at(target)))
            .collect();
        let slots: Vec<_> = link_targets
            .iter()
            .map(|_| Arc::new(AtomicUsize::new(0)))
            .collect();
        let slot_addresses: Vec<_> = slots.iter().map(|slot| Arc::as_ptr(slot).addr()).collect();
        let CompiledRegion {
            entry,
            native_bytes,
            clif_instructions,
            register_loads,
            register_stores,
        } = state.compiler.compile(&region, &slot_addresses)?;
        let attempted_bytes = state.native_bytes.saturating_add(native_bytes);
        if attempted_bytes > self.max_native_code_bytes {
            return Err(DirectJitError::capacity(format!(
                "direct JIT code arena exhausted: used={} attempted={} limit={} regions={} guest_blocks={}",
                state.native_bytes,
                native_bytes,
                self.max_native_code_bytes,
                state.compiled_regions,
                state.compiled_guest_blocks,
            )));
        }

        let links: Vec<_> = link_targets
            .into_iter()
            .zip(slots)
            .map(|(target, slot)| StaticLink { target, slot })
            .collect();
        let published = Arc::new(PublishedRegion {
            key,
            entry,
            native_bytes,
            clif_instructions,
            guest_blocks: region.blocks.len(),
            register_loads,
            register_stores,
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
        if let Some(incoming) = state.incoming.get(&key) {
            for slot in incoming {
                slot.store(entry, Ordering::Release);
            }
        }
        state.native_bytes = attempted_bytes;
        state.compiled_regions += 1;
        state.compiled_guest_blocks += published.guest_blocks;
        state.lookup.insert(Arc::clone(&published));
        Ok((state.compiler.gateway(), published.entry))
    }
}

struct JitThread {
    control: CpuControl,
    exclusive: Mutex<ExclusiveMonitorState>,
    events: VcpuEventState,
    slow_memory_calls: AtomicU64,
}

impl JitThread {
    fn new() -> Self {
        Self {
            control: CpuControl::default(),
            exclusive: Mutex::new(ExclusiveMonitorState::default()),
            events: VcpuEventState::default(),
            slow_memory_calls: AtomicU64::new(0),
        }
    }

    fn request_preempt(&self) {
        self.control.request(ControlRequest::Preempt);
    }

    fn run(
        &self,
        process: &JitProcess,
        memory: &dyn CpuMemory,
        state: &mut A64State,
        instruction_budget: u64,
    ) -> Result<DirectExit, DirectJitError> {
        self.run_with_runtime(
            process,
            memory,
            state,
            instruction_budget,
            &ZeroTimer,
            &self.events,
        )
    }

    fn run_with_runtime(
        &self,
        process: &JitProcess,
        memory: &dyn CpuMemory,
        state: &mut A64State,
        instruction_budget: u64,
        timer: &dyn ArchitecturalTimer,
        events: &VcpuEventState,
    ) -> Result<DirectExit, DirectJitError> {
        let mut exclusive = self
            .exclusive
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut context = NativeContext::new(
            state,
            memory,
            &mut exclusive,
            timer,
            events,
            process.cpu.address_space_id(),
            instruction_budget,
            &self.control,
            &self.slow_memory_calls,
        );
        loop {
            let location = LocationDescriptor::new(
                GuestVirtualAddress::new(unsafe { *context.pc }),
                ExecutionState::A64,
                process.cpu.profile().id(),
            );
            let (gateway, entry) = process.entry_for(memory, location)?;
            unsafe { gateway(&mut context, entry) };
            let exit = context.exit()?;
            if matches!(exit, DirectExit::Dispatch { .. })
                && context.retired < context.instruction_budget
                && unsafe { std::ptr::read_volatile(context.control_pending as *const u32) } == 0
            {
                continue;
            }
            return Ok(exit);
        }
    }
}

struct ZeroTimer;

impl ArchitecturalTimer for ZeroTimer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 0,
        }
    }
}

#[cfg(test)]
mod tests;
