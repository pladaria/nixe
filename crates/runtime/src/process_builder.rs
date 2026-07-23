//! Construction of a runnable CPU process from an immutable launch plan.

use std::error::Error;
use std::fmt::{Display, Formatter};

use nixe_cpu::address::{AddressSpaceId, GuestVirtualAddress};
use nixe_cpu::ir::block::IrBlock;
use nixe_cpu::ir::print::{IrPrintOptions, print_block};
use nixe_cpu::location::{ExecutionState, LocationDescriptor};
use nixe_cpu::memory::{
    MemoryMappingPurpose, MemoryPermissions, SYNTHETIC_PAGE_SIZE, SyntheticMemory, SyntheticRamPage,
};
use nixe_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::{ThreadCpuState, a32::A32GeneralRegister, a64::A64Register};
use nixe_cpu::translate::{
    BlockTranslationConfig, BlockTranslationReport, translate_block, translate_block_report,
};
use nixe_loader_executable::{
    AddressSpaceType, ExternalSymbol, PreparationConfig, PreparedModule, SymbolResolution,
};

use crate::exception_dispatch::ExceptionProcessMetadata;
use crate::{
    ExceptionDispatchContext, ExceptionDispatchOutcome, ExceptionDispatcher,
    ExceptionHandlingResult, ExceptionProcessContext, ExceptionResume, ExceptionRouteError,
    ExceptionTerminationReason, ExceptionTerminationScope, ExceptionThreadContext, ExecutionReport,
    ExecutionStop, LaunchKind, LaunchModuleImage, LaunchPlan, ProcessExecutionError,
    ProcessExecutionStatus, ProcessExit, ProcessExitCause, ProcessTeardownReport, ThreadExit,
    install_prepared_module,
};

const DEFAULT_IMAGE_BASE: u64 = 0x7100_0000;
const DEFAULT_HOME_BREW_STACK_SIZE: u64 = 0x10_0000;
const MODULE_GUARD_SIZE: u64 = 0x1_0000;
const RESOURCE_GUARD_SIZE: u64 = 0x1_0000;
const TLS_SIZE: u64 = SYNTHETIC_PAGE_SIZE as u64;
const HOME_BREW_CONFIG_ENTRY_SIZE: usize = 24;
const HOME_BREW_MAIN_THREAD_HANDLE_KEY: u32 = 1;
const HOME_BREW_EXIT_PROCESS_INSTRUCTION: u32 = 0xd400_00e1;
const DEFAULT_PHYSICAL_MEMORY_LIMIT: u64 = 0x4000_0000;
const HORIZON_REGION_ALIGNMENT: u64 = 0x20_0000;

/// Runtime interpretation of the address-space selector validated by NPDM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessAddressSpace {
    Bit32,
    Bit32NoReserved,
    Bit64Old,
    Bit64,
}

/// Horizon kernel generation governing process virtual-region availability.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProcessMemoryLayoutProfile {
    Horizon1,
    #[default]
    Horizon2Plus,
}

impl ProcessAddressSpace {
    const fn from_npdm(value: AddressSpaceType) -> Self {
        match value {
            AddressSpaceType::AddressSpace32Bit => Self::Bit32,
            AddressSpaceType::AddressSpace32BitNoReserved => Self::Bit32NoReserved,
            AddressSpaceType::AddressSpace64BitOld => Self::Bit64Old,
            AddressSpaceType::AddressSpace64Bit => Self::Bit64,
        }
    }

    pub const fn exclusive_limit(self) -> u64 {
        match self {
            Self::Bit32 | Self::Bit32NoReserved => 1_u64 << 32,
            Self::Bit64Old => 1_u64 << 36,
            Self::Bit64 => 1_u64 << 39,
        }
    }
}

/// One reserved guest-virtual region reported through platform process APIs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessVirtualRegion {
    base: GuestVirtualAddress,
    size: u64,
}

impl ProcessVirtualRegion {
    #[must_use]
    pub const fn new(base: GuestVirtualAddress, size: u64) -> Self {
        Self { base, size }
    }

    #[must_use]
    pub const fn base(self) -> GuestVirtualAddress {
        self.base
    }

    #[must_use]
    pub const fn size(self) -> u64 {
        self.size
    }

    const fn end(self) -> u64 {
        self.base.get() + self.size
    }
}

/// Runtime-owned Horizon process virtual-memory layout.
///
/// Region dimensions and the 39-bit placement policy follow the public
/// Atmosphere `svc_memory_map.hpp` and `KPageTableBase::InitializeForProcess`
/// definitions. The ASLR window may contain the concrete heap, alias, and
/// stack reservations; it is not a disjoint allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessMemoryLayout {
    aslr: ProcessVirtualRegion,
    heap: ProcessVirtualRegion,
    alias: ProcessVirtualRegion,
    stack: ProcessVirtualRegion,
    memory_capacity: u64,
}

impl ProcessMemoryLayout {
    fn for_address_space(
        profile: ProcessMemoryLayoutProfile,
        address_space: ProcessAddressSpace,
        process_code_start: u64,
        process_code_end: u64,
        memory_capacity: u64,
    ) -> Result<Self, ProcessBuildError> {
        if profile == ProcessMemoryLayoutProfile::Horizon1
            && address_space == ProcessAddressSpace::Bit64
        {
            return Err(error(
                ProcessBuildStage::Metadata,
                "39-bit process address spaces require Horizon 2.0.0 or newer",
            ));
        }
        let (aslr, alias, heap, stack) = match address_space {
            ProcessAddressSpace::Bit32 => (
                region(0x0020_0000, 0xffe0_0000),
                region(0x4000_0000, 0x4000_0000),
                region(0x8000_0000, 0x4000_0000),
                region(0x0020_0000, 0x3fe0_0000),
            ),
            ProcessAddressSpace::Bit32NoReserved => (
                region(0x0020_0000, 0xffe0_0000),
                region(0x4000_0000, 0),
                region(0x4000_0000, 0x8000_0000),
                region(0x0020_0000, 0x3fe0_0000),
            ),
            ProcessAddressSpace::Bit64Old => (
                region(0x0800_0000, 0xf_f800_0000),
                region(0x8000_0000, 0x1_8000_0000),
                region(0x2_0000_0000, 0x2_0000_0000),
                region(0x0800_0000, 0x7800_0000),
            ),
            ProcessAddressSpace::Bit64 => {
                let code_start = process_code_start & !(HORIZON_REGION_ALIGNMENT - 1);
                let code_end = align_up(process_code_end, HORIZON_REGION_ALIGNMENT)?;
                let aslr = region(0x0800_0000, (1_u64 << 39) - 0x0800_0000);
                if code_start < aslr.base().get() || code_end > aslr.end() {
                    return Err(error(
                        ProcessBuildStage::Placement,
                        "process code is outside the 39-bit Horizon ASLR window",
                    ));
                }

                let [_, stack, alias, heap] = layout_39_bit_regions(aslr, code_start, code_end)?;
                (aslr, alias, heap, stack)
            }
        };
        Ok(Self {
            aslr,
            heap,
            alias,
            stack,
            memory_capacity,
        })
    }

    #[must_use]
    pub const fn aslr(self) -> ProcessVirtualRegion {
        self.aslr
    }

    #[must_use]
    pub const fn heap(self) -> ProcessVirtualRegion {
        self.heap
    }

    #[must_use]
    pub const fn alias(self) -> ProcessVirtualRegion {
        self.alias
    }

    #[must_use]
    pub const fn stack(self) -> ProcessVirtualRegion {
        self.stack
    }

    /// Returns the process commit limit used for memory accounting.
    #[must_use]
    pub const fn memory_capacity(self) -> u64 {
        self.memory_capacity
    }
}

/// Caller-controlled process identities and relocatable image placement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessBuildConfig {
    pub process_id: u64,
    pub address_space_id: AddressSpaceId,
    pub cpu_profile: GuestCpuProfile,
    pub memory_layout_profile: ProcessMemoryLayoutProfile,
    pub image_base: GuestVirtualAddress,
    /// Physical-memory resource limit assigned to the emulated process.
    pub physical_memory_limit: u64,
    /// Frequency exposed by architectural counter registers.
    pub architectural_timer_frequency: u64,
}

impl Default for ProcessBuildConfig {
    fn default() -> Self {
        Self {
            process_id: 1,
            address_space_id: AddressSpaceId::new(1),
            cpu_profile: GuestCpuProfile::switch_1(),
            memory_layout_profile: ProcessMemoryLayoutProfile::Horizon2Plus,
            image_base: GuestVirtualAddress::new(DEFAULT_IMAGE_BASE),
            physical_memory_limit: DEFAULT_PHYSICAL_MEMORY_LIMIT,
            // Horizon exposes the Switch 1 system counter at 19.2 MHz:
            // https://switchbrew.org/w/index.php?title=SVC&oldid=14679#svcGetSystemTick
            architectural_timer_frequency: 19_200_000,
        }
    }
}

const fn region(base: u64, size: u64) -> ProcessVirtualRegion {
    ProcessVirtualRegion::new(GuestVirtualAddress::new(base), size)
}

fn layout_39_bit_regions(
    aslr: ProcessVirtualRegion,
    code_start: u64,
    code_end: u64,
) -> Result<[ProcessVirtualRegion; 4], ProcessBuildError> {
    // Region kinds use Horizon's deterministic no-ASLR ordering:
    // kernel-map, stack, alias, heap.
    let sizes = [0x10_0000_0000, 0x8000_0000, 0x10_0000_0000, 0x2_0000_0000];
    let mut by_descending_size = [(0_usize, sizes[0]); 4];
    for (kind, entry) in by_descending_size.iter_mut().enumerate() {
        *entry = (kind, sizes[kind]);
    }
    by_descending_size.sort_by_key(|entry| std::cmp::Reverse(entry.1));

    let allocation_starts = [aslr.base().get(), code_end];
    let mut allocation_sizes = [code_start - aslr.base().get(), aslr.end() - code_end];
    let mut assignment = [usize::MAX; 4];
    for (kind, size) in by_descending_size {
        let allocation = usize::from(allocation_sizes[1] >= allocation_sizes[0]);
        if allocation_sizes[allocation] < size {
            return Err(error(
                ProcessBuildStage::Placement,
                "39-bit Horizon regions do not fit around process code",
            ));
        }
        allocation_sizes[allocation] -= size;
        assignment[kind] = allocation;
    }

    let mut result = [region(0, 0); 4];
    for (allocation, start) in allocation_starts.into_iter().enumerate() {
        let mut cursor = start;
        for kind in 0..sizes.len() {
            if assignment[kind] == allocation {
                result[kind] = region(cursor, sizes[kind]);
                cursor = cursor.checked_add(sizes[kind]).ok_or_else(|| {
                    error(ProcessBuildStage::Placement, "Horizon region overflows")
                })?;
            }
        }
    }
    Ok(result)
}

/// Fully initialized main thread returned by [`ProcessBuilder`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MainThread {
    object: crate::ThreadObject,
    exit: Option<ThreadExit>,
    pub state: ThreadCpuState,
    pub handle: u32,
    pub stack_bottom: GuestVirtualAddress,
    pub stack_top: GuestVirtualAddress,
    pub tls_base: GuestVirtualAddress,
    pub abi_context: Option<GuestVirtualAddress>,
    /// Runtime-owned guest address installed as the original NRO link register.
    pub loader_return: Option<GuestVirtualAddress>,
}

impl MainThread {
    /// Returns the runtime-owned thread identity independently of guest handles.
    #[must_use]
    pub const fn object(&self) -> crate::ThreadObject {
        self.object
    }

    /// Returns the immutable termination record once this thread has exited.
    #[must_use]
    pub const fn exit(&self) -> Option<ThreadExit> {
        self.exit
    }
}

/// A process whose executable bytes are visible only through process memory.
pub struct RunnableProcess {
    process_id: u64,
    cpu: ProcessCpuContext,
    address_space: ProcessAddressSpace,
    memory_layout: ProcessMemoryLayout,
    random_entropy: [u64; 4],
    heap_size: u64,
    initial_memory_size: u64,
    memory: SyntheticMemory,
    modules: Box<[PreparedModule]>,
    entry_module: usize,
    main_thread: MainThread,
    mounts: crate::ProcessMountNamespace,
    handles: crate::HandleTable,
    execution: crate::execution::ProcessExecutionControl,
}

impl RunnableProcess {
    #[must_use]
    pub const fn process_id(&self) -> u64 {
        self.process_id
    }

    #[must_use]
    pub const fn cpu_context(&self) -> ProcessCpuContext {
        self.cpu
    }

    #[must_use]
    pub const fn address_space(&self) -> ProcessAddressSpace {
        self.address_space
    }

    #[must_use]
    pub const fn memory_layout(&self) -> ProcessMemoryLayout {
        self.memory_layout
    }

    /// Returns the currently committed process heap size.
    #[must_use]
    pub const fn heap_size(&self) -> u64 {
        self.heap_size
    }

    #[must_use]
    pub const fn memory(&self) -> &SyntheticMemory {
        &self.memory
    }

    #[must_use]
    pub fn modules(&self) -> &[PreparedModule] {
        &self.modules
    }

    #[must_use]
    pub fn entry_module(&self) -> &PreparedModule {
        &self.modules[self.entry_module]
    }

    #[must_use]
    pub const fn main_thread(&self) -> &MainThread {
        &self.main_thread
    }

    /// Returns mutable main-thread state for runtime scheduling and ABI setup.
    pub const fn main_thread_mut(&mut self) -> &mut MainThread {
        &mut self.main_thread
    }

    /// Returns the immutable process-local filesystem namespace.
    #[must_use]
    pub const fn mounts(&self) -> &crate::ProcessMountNamespace {
        &self.mounts
    }

    /// Returns the process-local kernel-object handle table.
    #[must_use]
    pub const fn handles(&self) -> &crate::HandleTable {
        &self.handles
    }

    /// Returns mutable handle access for future syscall/IPC dispatch.
    pub const fn handles_mut(&mut self) -> &mut crate::HandleTable {
        &mut self.handles
    }

    /// Borrows the console-neutral resources needed by a platform service layer.
    pub fn mounts_and_handles_mut(
        &mut self,
    ) -> (&crate::ProcessMountNamespace, &mut crate::HandleTable) {
        (&self.mounts, &mut self.handles)
    }

    /// Returns the host-side lifecycle state of this process.
    #[must_use]
    pub const fn execution_status(&self) -> ProcessExecutionStatus {
        self.execution.status()
    }

    /// Returns the process exit record retained until teardown.
    #[must_use]
    pub const fn exit(&self) -> Option<ProcessExit> {
        self.execution.exit()
    }

    /// Requests a stop before the next reference-engine instruction.
    pub fn request_safepoint(&mut self) {
        self.execution.request_safepoint();
    }

    /// Publishes runtime event bits to be observed at the next safepoint.
    pub fn post_event(&self, mask: u32) {
        self.execution.post_event(mask);
    }

    /// Resumes a process suspended by an exception or scheduling instruction.
    pub fn resume(&mut self) -> bool {
        self.execution.resume()
    }

    /// Marks the process exited. Resource release occurs in [`Self::teardown`]
    /// or when the process is dropped.
    pub fn terminate(&mut self) -> bool {
        let exit = ProcessExit {
            cause: ProcessExitCause::HostRequested,
            exit_code: 0,
            source: None,
            thread_id: self.main_thread.object.thread_id(),
        };
        let terminated = self.execution.terminate(exit);
        if terminated {
            self.main_thread.exit = Some(ThreadExit {
                requested_scope: ExceptionTerminationScope::Process,
                exit_code: 0,
                source: None,
            });
        }
        terminated
    }

    /// Runs a bounded slice through the independent reference interpreter.
    ///
    /// This is the executable baseline used before an IR evaluator or native
    /// backend exists. It intentionally does not claim to execute translated IR.
    pub fn run_reference(
        &mut self,
        instruction_budget: u64,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        let report = crate::execution::run_reference(
            &mut self.execution,
            self.cpu,
            &self.memory,
            &mut self.main_thread.state,
            instruction_budget,
            self.main_thread.loader_return,
        )?;
        if let ExecutionStop::LoaderReturn {
            source,
            result_code,
        } = &report.stop
        {
            let thread_id = self.main_thread.object.thread_id();
            let exit = ProcessExit {
                cause: ProcessExitCause::LoaderReturned,
                exit_code: *result_code,
                source: Some(*source),
                thread_id,
            };
            let transitioned = self.execution.terminate(exit);
            debug_assert!(transitioned);
            self.main_thread.exit = Some(ThreadExit {
                requested_scope: ExceptionTerminationScope::Process,
                exit_code: *result_code,
                source: Some(*source),
            });
        }
        Ok(report)
    }

    /// Routes and atomically applies one supervisor-call decision.
    ///
    /// A normal handler must return [`ExceptionResume::Next`]; this method then
    /// advances past the SVC exactly once. Retry is explicit, suspension keeps
    /// its selected continuation non-runnable, and faults retain the SVC source
    /// for deterministic diagnostics.
    pub fn route_supervisor_call<D: ExceptionDispatcher>(
        &mut self,
        stop: &ExecutionStop,
        dispatcher: &mut D,
    ) -> Result<ExceptionHandlingResult<D::Fault>, ExceptionRouteError> {
        let request = stop
            .exception_dispatch_request()
            .filter(|request| request.kind() == nixe_cpu::exception::ExceptionKind::SupervisorCall)
            .ok_or(ExceptionRouteError::NotSupervisorCall)?;
        if self.execution.status() != ProcessExecutionStatus::Suspended {
            return Err(ExceptionRouteError::ProcessNotSuspended {
                status: self.execution.status(),
            });
        }
        let current = crate::execution::current_location(self.cpu, &self.main_thread.state);
        if request.source() != current {
            return Err(ExceptionRouteError::SourceMismatch {
                requested: request.source(),
                current,
            });
        }
        let handle = self.main_thread.handle;
        let object = self.main_thread.object;
        let process = ExceptionProcessContext::new(
            ExceptionProcessMetadata {
                process_id: self.process_id,
                cpu: self.cpu,
                address_space_limit: self.address_space.exclusive_limit(),
                memory_layout: self.memory_layout,
                random_entropy: self.random_entropy,
                initial_memory_size: self.initial_memory_size,
            },
            &mut self.heap_size,
            &self.memory,
            &self.mounts,
            &mut self.handles,
        );
        let thread = ExceptionThreadContext::new(object, handle, &mut self.main_thread.state);
        let mut context = ExceptionDispatchContext::new(process, thread);
        let outcome = dispatcher.dispatch(&mut context, request);
        self.apply_supervisor_call_outcome(request.source(), outcome)
    }

    fn apply_supervisor_call_outcome<F>(
        &mut self,
        source: LocationDescriptor,
        outcome: ExceptionDispatchOutcome<F>,
    ) -> Result<ExceptionHandlingResult<F>, ExceptionRouteError> {
        match outcome {
            ExceptionDispatchOutcome::Resume(continuation) => {
                let target = supervisor_call_continuation(source, continuation)?;
                install_continuation(self.cpu, &mut self.main_thread.state, target)?;
                let transitioned = self.execution.resume();
                debug_assert!(transitioned);
                Ok(ExceptionHandlingResult::Resumed)
            }
            ExceptionDispatchOutcome::Suspend(continuation) => {
                let target = supervisor_call_continuation(source, continuation)?;
                install_continuation(self.cpu, &mut self.main_thread.state, target)?;
                Ok(ExceptionHandlingResult::Suspended)
            }
            ExceptionDispatchOutcome::Reject { diagnostic } => {
                let target = supervisor_call_continuation(source, ExceptionResume::Next)?;
                install_continuation(self.cpu, &mut self.main_thread.state, target)?;
                let transitioned = self.execution.resume();
                debug_assert!(transitioned);
                Ok(ExceptionHandlingResult::Rejected(diagnostic))
            }
            ExceptionDispatchOutcome::Terminate {
                scope,
                exit_code,
                reason,
            } => {
                install_continuation(self.cpu, &mut self.main_thread.state, source)?;
                let thread_id = self.main_thread.object.thread_id();
                let exit = ProcessExit {
                    cause: match reason {
                        ExceptionTerminationReason::Break { reason, info, size } => {
                            ProcessExitCause::GuestBreak { reason, info, size }
                        }
                        ExceptionTerminationReason::Requested => match scope {
                            ExceptionTerminationScope::CurrentThread => {
                                ProcessExitCause::LastThreadExited
                            }
                            ExceptionTerminationScope::Process => {
                                ProcessExitCause::ProcessRequested
                            }
                        },
                    },
                    exit_code,
                    source: Some(source),
                    thread_id,
                };
                let transitioned = self.execution.terminate(exit);
                debug_assert!(transitioned);
                self.main_thread.exit = Some(ThreadExit {
                    requested_scope: scope,
                    exit_code,
                    source: Some(source),
                });
                Ok(ExceptionHandlingResult::Terminated {
                    scope,
                    exit_code,
                    reason,
                })
            }
            ExceptionDispatchOutcome::Fault(fault) => {
                install_continuation(self.cpu, &mut self.main_thread.state, source)?;
                let transitioned = self.execution.fault();
                debug_assert!(transitioned);
                Ok(ExceptionHandlingResult::Fault(fault))
            }
        }
    }

    /// Consumes the process and deterministically releases all process-owned resources.
    #[must_use]
    pub fn teardown(self) -> ProcessTeardownReport {
        ProcessTeardownReport {
            previous_status: self.execution.status(),
            exit: self.execution.exit(),
            threads_released: 1,
            modules_released: self.modules.len(),
            mappings_released: self
                .modules
                .iter()
                .map(|module| module.mappings().len())
                .sum(),
            physical_pages_released: self.memory.physical_page_count(),
            mounts_released: self.mounts.mount_count(),
            handles_released: self.handles.len(),
        }
    }

    /// Translates and verifies the initialized entry block through process memory.
    pub fn translate_entry(&self) -> Result<IrBlock, ProcessBuildError> {
        translate_block(
            BlockTranslationConfig::default(),
            &self.cpu.profile(),
            self.cpu.address_space_id(),
            self.entry_location(),
            &self.memory,
        )
        .map_err(|error| ProcessBuildError::new(ProcessBuildStage::EntryTranslation, error))
    }

    /// Translates the entry block with source disassembly and a structured
    /// failure report. This path is opt-in and never runs during normal build.
    #[must_use]
    pub fn translate_entry_report(&self) -> BlockTranslationReport {
        translate_block_report(
            BlockTranslationConfig::default(),
            &self.cpu.profile(),
            self.cpu.address_space_id(),
            self.entry_location(),
            &self.memory,
        )
    }

    /// Produces the deterministic verified-IR dump used by the first integration milestone.
    pub fn print_entry_ir(&self) -> Result<String, ProcessBuildError> {
        let block = self
            .translate_entry_report()
            .into_result()
            .map_err(|error| ProcessBuildError::new(ProcessBuildStage::EntryTranslation, error))?;
        Ok(print_block(&block, IrPrintOptions::default()))
    }

    /// Produces the compact source, dependency, end-reason, and IR report used
    /// for entry-point bring-up without attaching a native debugger.
    #[must_use]
    pub fn print_entry_report(&self) -> String {
        self.translate_entry_report().print()
    }

    fn entry_location(&self) -> LocationDescriptor {
        LocationDescriptor::new(
            GuestVirtualAddress::new(self.entry_module().entry_address()),
            self.main_thread.state.execution_state(),
            self.cpu.profile().id(),
        )
    }
}

fn supervisor_call_continuation(
    source: LocationDescriptor,
    continuation: ExceptionResume,
) -> Result<LocationDescriptor, ExceptionRouteError> {
    match continuation {
        ExceptionResume::Retry => Ok(source),
        ExceptionResume::At(target) => Ok(target),
        ExceptionResume::Next => {
            let width = match source.execution_state {
                ExecutionState::A64 | ExecutionState::A32 => 4,
                ExecutionState::T32 => 2,
            };
            let pc = source
                .pc
                .checked_add(width)
                .ok_or(ExceptionRouteError::ContinuationAddressOverflow { source })?;
            Ok(LocationDescriptor::new(
                pc,
                source.execution_state,
                source.profile_id,
            ))
        }
    }
}

fn install_continuation(
    cpu: ProcessCpuContext,
    state: &mut ThreadCpuState,
    target: LocationDescriptor,
) -> Result<(), ExceptionRouteError> {
    let current = state.execution_state();
    let expected_profile = cpu.profile().id();
    if target.profile_id != expected_profile {
        return Err(ExceptionRouteError::ContinuationProfileMismatch {
            source: crate::execution::current_location(cpu, state),
            target,
        });
    }
    if !target.is_aligned() {
        return Err(ExceptionRouteError::InvalidContinuationTarget { target });
    }
    match state {
        ThreadCpuState::A64(state) if target.execution_state == ExecutionState::A64 => {
            state.set_pc(target.pc.get());
        }
        ThreadCpuState::A32(state) if target.execution_state != ExecutionState::A64 => {
            let pc = u32::try_from(target.pc.get())
                .map_err(|_| ExceptionRouteError::InvalidContinuationTarget { target })?;
            let cpsr = state
                .cpsr()
                .with_execution_state(target.execution_state)
                .expect("AArch32 continuation state was already validated");
            state.set_cpsr(cpsr);
            state
                .set_instruction_address(pc)
                .map_err(|_| ExceptionRouteError::InvalidContinuationTarget { target })?;
        }
        ThreadCpuState::A64(_) | ThreadCpuState::A32(_) => {
            return Err(ExceptionRouteError::IncompatibleContinuationState {
                current,
                target: target.execution_state,
            });
        }
    }
    Ok(())
}

/// Stage at which process construction failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessBuildStage {
    Metadata,
    Placement,
    Preparation,
    Mapping,
    ThreadInitialization,
    EntryTranslation,
}

/// Fail-closed process construction error.
#[derive(Debug)]
pub struct ProcessBuildError {
    stage: ProcessBuildStage,
    cause: Box<str>,
}

impl ProcessBuildError {
    fn new(stage: ProcessBuildStage, cause: impl Display) -> Self {
        Self {
            stage,
            cause: cause.to_string().into_boxed_str(),
        }
    }

    #[must_use]
    pub const fn stage(&self) -> ProcessBuildStage {
        self.stage
    }

    #[must_use]
    pub const fn cause(&self) -> &str {
        &self.cause
    }
}

impl Display for ProcessBuildError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "cannot build process during {:?}: {}",
            self.stage, self.cause
        )
    }
}

impl Error for ProcessBuildError {}

/// Builds an emulated process from a prepared launch plan.
#[derive(Debug, Default)]
pub struct ProcessBuilder {
    diagnostics: crate::DiagnosticsPolicy,
    config: ProcessBuildConfig,
    virtual_clock: crate::VirtualClock,
}

impl ProcessBuilder {
    /// Creates a process builder using detailed diagnostics and Switch 1 defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn with_diagnostics(mut self, diagnostics: crate::DiagnosticsPolicy) -> Self {
        self.diagnostics = diagnostics;
        self
    }

    #[must_use]
    pub const fn with_config(mut self, config: ProcessBuildConfig) -> Self {
        self.config = config;
        self
    }

    /// Selects the clock source shared by CPU architectural timers and services.
    #[must_use]
    pub fn with_virtual_clock(mut self, virtual_clock: crate::VirtualClock) -> Self {
        self.virtual_clock = virtual_clock;
        self
    }

    #[must_use]
    pub const fn diagnostics(&self) -> crate::DiagnosticsPolicy {
        self.diagnostics
    }

    #[must_use]
    pub const fn cpu_diagnostics(&self) -> nixe_cpu::coverage::CpuDiagnosticsConfig {
        self.diagnostics.cpu()
    }

    /// Prepares, maps, and initializes one runnable process.
    ///
    /// Packaged NSOs retain their dynamic relocations for the guest `rtld`.
    /// Standalone NROs likewise enter through their guest startup ABI.
    pub fn build(&self, plan: &LaunchPlan) -> Result<RunnableProcess, ProcessBuildError> {
        if self.config.architectural_timer_frequency == 0 {
            return Err(ProcessBuildError::new(
                ProcessBuildStage::Metadata,
                "architectural timer frequency must be nonzero",
            ));
        }
        let (execution_state, address_space, stack_size, abi) = process_metadata(plan);
        let random_entropy = generate_process_entropy()?;
        let cpu = ProcessCpuContext::new(self.config.cpu_profile, self.config.address_space_id);
        let thread_configuration = cpu
            .thread_configuration(execution_state)
            .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Metadata, error))?;
        let placements = module_placements(plan, self.config.image_base, address_space)?;
        let modules = prepare_modules(plan, &placements, address_space)?;
        let process_code_start = modules
            .iter()
            .map(PreparedModule::image_base)
            .min()
            .ok_or_else(|| error(ProcessBuildStage::Placement, "launch plan has no modules"))?;
        let process_code_end = modules
            .iter()
            .map(|module| module.image_base().saturating_add(module.image_extent()))
            .max()
            .ok_or_else(|| error(ProcessBuildStage::Placement, "launch plan has no modules"))?;
        let memory_layout = ProcessMemoryLayout::for_address_space(
            self.config.memory_layout_profile,
            address_space,
            process_code_start,
            process_code_end,
            self.config.physical_memory_limit,
        )?;
        let entry_module = plan.entry_module_index();
        // ACI0 is the process-requested policy after ACID authorization has
        // already been checked by the NPDM loader. Horizon uses its
        // HandleTableSize descriptor as the live per-process handle limit:
        // https://switchbrew.org/w/index.php?title=NPDM&oldid=14486#HandleTableSize
        let mut handles = plan
            .effective_policy()
            .and_then(|policy| policy.handle_table_size())
            .map_or_else(crate::HandleTable::new, |size| {
                crate::HandleTable::with_capacity_limit(usize::from(size))
            });
        let main_thread_object = crate::ThreadObject::new(1);
        let main_thread_handle = handles.insert(main_thread_object).map_err(|error| {
            ProcessBuildError::new(ProcessBuildStage::ThreadInitialization, error)
        })?;

        let mut memory = SyntheticMemory::new();
        for module in &modules {
            install_prepared_module(&mut memory, self.config.address_space_id, module)
                .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Mapping, error))?;
            for mapping in module.mappings() {
                let mutable = mapping.permissions().is_writable();
                let purpose = match (matches!(abi, InitialProcessAbi::Homebrew), mutable) {
                    (true, true) => MemoryMappingPurpose::ModuleCodeMutable,
                    (true, false) => MemoryMappingPurpose::ModuleCodeStatic,
                    (false, true) => MemoryMappingPurpose::CodeMutable,
                    (false, false) => MemoryMappingPurpose::CodeStatic,
                };
                if !memory.set_mapping_purpose(
                    self.config.address_space_id,
                    GuestVirtualAddress::new(mapping.guest_address()),
                    mapping.bytes().len() as u64,
                    purpose,
                ) {
                    return Err(error(
                        ProcessBuildStage::Mapping,
                        "installed module mapping could not retain its purpose",
                    ));
                }
            }
        }

        let stack_size = align_up(stack_size.max(SYNTHETIC_PAGE_SIZE as u64), TLS_SIZE)?;
        if stack_size + (RESOURCE_GUARD_SIZE * 3) + (TLS_SIZE * 2) > memory_layout.stack().size() {
            return Err(error(
                ProcessBuildStage::Placement,
                "main-thread resources exceed the reserved stack region",
            ));
        }
        let stack_bottom = memory_layout.stack().base();
        let stack_top = stack_bottom
            .checked_add(stack_size)
            .ok_or_else(|| error(ProcessBuildStage::Placement, "main stack overflows"))?;
        let tls_base = stack_top
            .checked_add(RESOURCE_GUARD_SIZE)
            .ok_or_else(|| error(ProcessBuildStage::Placement, "TLS base overflows"))?;
        validate_range(address_space, tls_base.get(), TLS_SIZE)?;
        install_zero_pages(
            &mut memory,
            self.config.address_space_id,
            stack_bottom,
            stack_size,
        )?;
        install_zero_pages(
            &mut memory,
            self.config.address_space_id,
            tls_base,
            TLS_SIZE,
        )?;
        if !memory.set_mapping_purpose(
            self.config.address_space_id,
            tls_base,
            TLS_SIZE,
            MemoryMappingPurpose::ThreadLocal,
        ) {
            return Err(error(
                ProcessBuildStage::Mapping,
                "installed TLS mapping could not retain its purpose",
            ));
        }
        let (abi_context, loader_return) = if matches!(abi, InitialProcessAbi::Homebrew) {
            let address = tls_base
                .checked_add(TLS_SIZE + RESOURCE_GUARD_SIZE)
                .ok_or_else(|| error(ProcessBuildStage::Placement, "ABI context overflows"))?;
            validate_range(address_space, address.get(), SYNTHETIC_PAGE_SIZE as u64)?;
            install_homebrew_context(
                &mut memory,
                self.config.address_space_id,
                address,
                main_thread_handle,
            )?;
            let loader_return = address
                .checked_add(SYNTHETIC_PAGE_SIZE as u64 + RESOURCE_GUARD_SIZE)
                .ok_or_else(|| error(ProcessBuildStage::Placement, "loader return overflows"))?;
            validate_range(
                address_space,
                loader_return.get(),
                SYNTHETIC_PAGE_SIZE as u64,
            )?;
            install_homebrew_loader_return(
                &mut memory,
                self.config.address_space_id,
                loader_return,
            )?;
            (Some(address), Some(loader_return))
        } else {
            (None, None)
        };

        let entry = GuestVirtualAddress::new(modules[entry_module].entry_address());
        let mut state = ThreadCpuState::new(thread_configuration);
        initialize_thread(
            &mut state,
            entry,
            stack_top,
            tls_base,
            main_thread_handle,
            abi_context,
            loader_return,
        )?;
        let main_thread = MainThread {
            object: main_thread_object,
            exit: None,
            state,
            handle: main_thread_handle,
            stack_bottom,
            stack_top,
            tls_base,
            abi_context,
            loader_return,
        };
        let initial_memory_size = u64::try_from(memory.physical_page_count())
            .ok()
            .and_then(|pages| pages.checked_mul(SYNTHETIC_PAGE_SIZE as u64))
            .ok_or_else(|| {
                error(
                    ProcessBuildStage::Mapping,
                    "process memory accounting overflows",
                )
            })?;
        if initial_memory_size > memory_layout.memory_capacity() {
            return Err(error(
                ProcessBuildStage::Mapping,
                "initial process mappings exceed the configured physical-memory limit",
            ));
        }
        let process = RunnableProcess {
            process_id: self.config.process_id,
            cpu,
            address_space,
            memory_layout,
            random_entropy,
            heap_size: 0,
            initial_memory_size,
            memory,
            modules: modules.into_boxed_slice(),
            entry_module,
            main_thread,
            mounts: crate::ProcessMountNamespace::from_launch_plan(plan),
            handles,
            execution: crate::execution::ProcessExecutionControl::new(
                self.diagnostics,
                self.virtual_clock.clone(),
                self.config.architectural_timer_frequency,
            ),
        };
        process.translate_entry()?;
        Ok(process)
    }
}

fn generate_process_entropy() -> Result<[u64; 4], ProcessBuildError> {
    let mut bytes = [0_u8; size_of::<[u64; 4]>()];
    getrandom::fill(&mut bytes).map_err(|error| {
        ProcessBuildError::new(
            ProcessBuildStage::Metadata,
            format_args!("cannot obtain host entropy for the guest process: {error}"),
        )
    })?;
    Ok(std::array::from_fn(|index| {
        let offset = index * size_of::<u64>();
        u64::from_le_bytes(bytes[offset..offset + size_of::<u64>()].try_into().unwrap())
    }))
}

#[derive(Clone, Copy)]
enum InitialProcessAbi {
    Packaged,
    Homebrew,
}

fn process_metadata(
    plan: &LaunchPlan,
) -> (ExecutionState, ProcessAddressSpace, u64, InitialProcessAbi) {
    match plan.kind() {
        LaunchKind::Packaged(identity) => {
            let npdm = identity.npdm();
            let state = if npdm.flags().is_64_bit_instruction() {
                ExecutionState::A64
            } else {
                ExecutionState::A32
            };
            (
                state,
                ProcessAddressSpace::from_npdm(npdm.flags().address_space()),
                u64::from(npdm.main_thread_stack_size()),
                InitialProcessAbi::Packaged,
            )
        }
        LaunchKind::Homebrew => (
            ExecutionState::A64,
            ProcessAddressSpace::Bit64,
            DEFAULT_HOME_BREW_STACK_SIZE,
            InitialProcessAbi::Homebrew,
        ),
    }
}

fn module_placements(
    plan: &LaunchPlan,
    first_base: GuestVirtualAddress,
    address_space: ProcessAddressSpace,
) -> Result<Vec<PreparationConfig>, ProcessBuildError> {
    let limit = address_space.exclusive_limit();
    let mut next = align_up(first_base.get(), SYNTHETIC_PAGE_SIZE as u64)?;
    let mut placements = Vec::with_capacity(plan.modules().len());
    for module in plan.modules() {
        let extent = image_extent(module.image())?;
        validate_range(address_space, next, extent)?;
        placements.push(PreparationConfig {
            image_base: next,
            address_limit: limit,
        });
        next = align_up(
            next.checked_add(extent)
                .and_then(|end| end.checked_add(MODULE_GUARD_SIZE))
                .ok_or_else(|| error(ProcessBuildStage::Placement, "module placement overflows"))?,
            SYNTHETIC_PAGE_SIZE as u64,
        )?;
    }
    Ok(placements)
}

fn image_extent(image: &LaunchModuleImage) -> Result<u64, ProcessBuildError> {
    let executable = match image {
        LaunchModuleImage::Nso(image) => image.executable(),
        LaunchModuleImage::Nro(image) => image.executable(),
    };
    executable
        .segments()
        .iter()
        .map(|segment| segment.memory_offset().checked_add(segment.mapping_size()))
        .try_fold(0_u64, |extent, end| {
            Ok(extent.max(
                end.ok_or_else(|| error(ProcessBuildStage::Placement, "module extent overflows"))?,
            ))
        })
}

fn prepare_modules(
    plan: &LaunchPlan,
    placements: &[PreparationConfig],
    address_space: ProcessAddressSpace,
) -> Result<Vec<PreparedModule>, ProcessBuildError> {
    let unresolved = |_: ExternalSymbol<'_>| SymbolResolution::Unresolved;
    plan.modules()
        .iter()
        .zip(placements)
        .map(|(module, config)| {
            let prepared = match module.image() {
                LaunchModuleImage::Nso(image) => {
                    image.prepare_for_guest_relocation(*config, &unresolved)
                }
                LaunchModuleImage::Nro(image) => {
                    image.prepare_for_guest_relocation(*config, &unresolved)
                }
            }
            .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Preparation, error))?;
            validate_range(
                address_space,
                prepared.image_base(),
                prepared.image_extent(),
            )?;
            Ok(prepared)
        })
        .collect()
}

fn install_zero_pages(
    memory: &mut SyntheticMemory,
    address_space: AddressSpaceId,
    start: GuestVirtualAddress,
    size: u64,
) -> Result<(), ProcessBuildError> {
    let zero = [0_u8; SYNTHETIC_PAGE_SIZE];
    let page_count = usize::try_from(size / SYNTHETIC_PAGE_SIZE as u64).map_err(|_| {
        error(
            ProcessBuildStage::Mapping,
            "resource page count is too large",
        )
    })?;
    let requests = (0..page_count)
        .map(|index| SyntheticRamPage {
            virtual_address: start
                .checked_add((index * SYNTHETIC_PAGE_SIZE) as u64)
                .expect("validated resource range"),
            bytes: &zero,
            permissions: MemoryPermissions::READ_WRITE,
        })
        .collect::<Vec<_>>();
    memory
        .install_ram_pages_atomic(address_space, &requests)
        .map_err(|failure| ProcessBuildError::new(ProcessBuildStage::Mapping, failure.reason))
}

fn install_homebrew_context(
    memory: &mut SyntheticMemory,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
    main_thread_handle: u32,
) -> Result<(), ProcessBuildError> {
    let mut page = [0_u8; SYNTHETIC_PAGE_SIZE];
    page[..4].copy_from_slice(&HOME_BREW_MAIN_THREAD_HANDLE_KEY.to_le_bytes());
    page[8..16].copy_from_slice(&u64::from(main_thread_handle).to_le_bytes());
    // The following zeroed 24-byte entry is EntryType_EndOfList.
    debug_assert!(HOME_BREW_CONFIG_ENTRY_SIZE * 2 <= page.len());
    memory
        .install_ram_pages_atomic(
            address_space,
            &[SyntheticRamPage {
                virtual_address: address,
                bytes: &page,
                permissions: MemoryPermissions::READ,
            }],
        )
        .map_err(|failure| ProcessBuildError::new(ProcessBuildStage::Mapping, failure.reason))
}

fn install_homebrew_loader_return(
    memory: &mut SyntheticMemory,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
) -> Result<(), ProcessBuildError> {
    let mut page = [0_u8; SYNTHETIC_PAGE_SIZE];
    // If an execution engine misses the runtime return-address boundary, the
    // mapped fallback still performs the ABI-prescribed process exit.
    page[..4].copy_from_slice(&HOME_BREW_EXIT_PROCESS_INSTRUCTION.to_le_bytes());
    memory
        .install_ram_pages_atomic(
            address_space,
            &[SyntheticRamPage {
                virtual_address: address,
                bytes: &page,
                permissions: MemoryPermissions::READ_EXECUTE,
            }],
        )
        .map_err(|failure| ProcessBuildError::new(ProcessBuildStage::Mapping, failure.reason))?;
    if !memory.set_mapping_purpose(
        address_space,
        address,
        SYNTHETIC_PAGE_SIZE as u64,
        MemoryMappingPurpose::CodeStatic,
    ) {
        return Err(error(
            ProcessBuildStage::Mapping,
            "loader return mapping could not retain its purpose",
        ));
    }
    Ok(())
}

fn initialize_thread(
    state: &mut ThreadCpuState,
    entry: GuestVirtualAddress,
    stack_top: GuestVirtualAddress,
    tls_base: GuestVirtualAddress,
    main_thread_handle: u32,
    abi_context: Option<GuestVirtualAddress>,
    loader_return: Option<GuestVirtualAddress>,
) -> Result<(), ProcessBuildError> {
    match state {
        ThreadCpuState::A64(state) => {
            state.set_pc(entry.get());
            state.write_x(A64Register::StackPointer, stack_top.get());
            state.set_tpidr_el0(tls_base.get());
            state.set_tpidrro_el0_from_runtime(tls_base.get());
            state.write_x(
                A64Register::General(a64_register(0)),
                abi_context.map_or(0, GuestVirtualAddress::get),
            );
            state.write_x(
                A64Register::General(a64_register(1)),
                if abi_context.is_some() {
                    u64::MAX
                } else {
                    u64::from(main_thread_handle)
                },
            );
            state.write_x(
                A64Register::General(a64_register(30)),
                loader_return.map_or(0, GuestVirtualAddress::get),
            );
        }
        ThreadCpuState::A32(state) => {
            let entry = u32::try_from(entry.get()).map_err(|_| {
                error(
                    ProcessBuildStage::ThreadInitialization,
                    "A32 PC exceeds 32 bits",
                )
            })?;
            let stack_top = u32::try_from(stack_top.get()).map_err(|_| {
                error(
                    ProcessBuildStage::ThreadInitialization,
                    "A32 SP exceeds 32 bits",
                )
            })?;
            let tls_base = u32::try_from(tls_base.get()).map_err(|_| {
                error(
                    ProcessBuildStage::ThreadInitialization,
                    "A32 TLS exceeds 32 bits",
                )
            })?;
            state.set_instruction_address(entry).map_err(|error| {
                ProcessBuildError::new(ProcessBuildStage::ThreadInitialization, error)
            })?;
            state.write_r(a32_register(13), stack_top);
            state.set_tpidrurw(tls_base);
            state.set_tpidruro_from_runtime(tls_base);
            state.write_r(a32_register(0), 0);
            state.write_r(a32_register(1), main_thread_handle);
        }
    }
    Ok(())
}

fn validate_range(
    address_space: ProcessAddressSpace,
    start: u64,
    size: u64,
) -> Result<(), ProcessBuildError> {
    let end = start
        .checked_add(size)
        .ok_or_else(|| error(ProcessBuildStage::Placement, "guest range overflows"))?;
    if end > address_space.exclusive_limit() {
        return Err(error(
            ProcessBuildStage::Placement,
            "guest range exceeds the NPDM-selected address width",
        ));
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, ProcessBuildError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| error(ProcessBuildStage::Placement, "alignment overflows"))
}

fn a64_register(index: u8) -> nixe_cpu::state::a64::A64GeneralRegister {
    nixe_cpu::state::a64::A64GeneralRegister::new(index).expect("valid ABI register")
}

fn a32_register(index: u8) -> A32GeneralRegister {
    A32GeneralRegister::new(index).expect("valid ABI register")
}

fn error(stage: ProcessBuildStage, cause: impl Display) -> ProcessBuildError {
    ProcessBuildError::new(stage, cause)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use nixe_cpu::exception::ExceptionKind;
    use nixe_cpu::ir::terminator::{ControlTarget, Terminator};
    use nixe_cpu::location::InstructionEncoding;
    use nixe_cpu::memory::{
        CpuMemory, InstructionMemory, MemoryAccess, MemoryAccessSize, MemoryPermissions,
        MemoryValue, SYNTHETIC_PAGE_SIZE,
    };

    use super::*;
    use crate::{Launcher, LauncherInput};

    #[derive(Default)]
    struct RecordingSupervisorCallDispatcher {
        expected_encoding: Option<InstructionEncoding>,
        observed: Option<(crate::ExceptionDispatchRequest, AddressSpaceId, u64, u32)>,
    }

    impl crate::ExceptionDispatcher for RecordingSupervisorCallDispatcher {
        type Fault = &'static str;

        fn dispatch(
            &mut self,
            context: &mut crate::ExceptionDispatchContext<'_>,
            request: crate::ExceptionDispatchRequest,
        ) -> crate::ExceptionDispatchOutcome<Self::Fault> {
            let address_space = context.process().cpu().address_space_id();
            let encoding = match request.source().execution_state {
                ExecutionState::A64 | ExecutionState::A32 => context
                    .process()
                    .memory()
                    .fetch32(address_space, request.source().pc)
                    .map(|value| InstructionEncoding::from_u32(value.bits))
                    .unwrap(),
                ExecutionState::T32 => context
                    .process()
                    .memory()
                    .fetch16(address_space, request.source().pc)
                    .map(|value| InstructionEncoding::from_u16(value.bits))
                    .unwrap(),
            };
            assert_eq!(Some(encoding), self.expected_encoding);
            assert!(
                context
                    .process()
                    .handles()
                    .get_as::<crate::ThreadObject>(context.thread().handle())
                    .is_some()
            );

            let thread_id = context.thread().object().thread_id();
            let handle = context.thread().handle();
            assert_eq!(
                context.thread().state().execution_state(),
                request.source().execution_state
            );
            match context.thread_mut().state_mut() {
                ThreadCpuState::A64(state) => state.write_x(
                    nixe_cpu::state::a64::A64Register::General(a64_register(0)),
                    0xfeed_face,
                ),
                ThreadCpuState::A32(state) => state.write_r(a32_register(0), 0xfeed_face),
            }
            self.observed = Some((request, address_space, thread_id, handle));
            crate::ExceptionDispatchOutcome::Suspend(crate::ExceptionResume::Retry)
        }
    }

    struct FixedSupervisorCallDispatcher<F> {
        outcome: Option<crate::ExceptionDispatchOutcome<F>>,
    }

    impl<F> crate::ExceptionDispatcher for FixedSupervisorCallDispatcher<F> {
        type Fault = F;

        fn dispatch(
            &mut self,
            _context: &mut crate::ExceptionDispatchContext<'_>,
            _request: crate::ExceptionDispatchRequest,
        ) -> crate::ExceptionDispatchOutcome<Self::Fault> {
            self.outcome.take().expect("dispatcher is called once")
        }
    }

    struct PcMutatingSupervisorCallDispatcher<F> {
        outcome: Option<crate::ExceptionDispatchOutcome<F>>,
    }

    impl<F> crate::ExceptionDispatcher for PcMutatingSupervisorCallDispatcher<F> {
        type Fault = F;

        fn dispatch(
            &mut self,
            context: &mut crate::ExceptionDispatchContext<'_>,
            _request: crate::ExceptionDispatchRequest,
        ) -> crate::ExceptionDispatchOutcome<Self::Fault> {
            match context.thread_mut().state_mut() {
                ThreadCpuState::A64(state) => state.set_pc(0x1000),
                ThreadCpuState::A32(state) => state.set_instruction_address(0x1000).unwrap(),
            }
            self.outcome.take().expect("dispatcher is called once")
        }
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn replace_entry_instruction(process: &mut RunnableProcess, encoding: u32) {
        let address_space = process.cpu.address_space_id();
        let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
        let mapping = process.memory.mapping_info(address_space, entry).unwrap();
        let alias_page = GuestVirtualAddress::new(0x6000_0000);
        assert!(process.memory.map_page(
            address_space,
            alias_page,
            mapping.physical_page,
            MemoryPermissions::READ_WRITE,
        ));
        let page_offset = entry.get() % SYNTHETIC_PAGE_SIZE as u64;
        process
            .memory
            .write(
                address_space,
                GuestVirtualAddress::new(alias_page.get() + page_offset),
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(encoding),
            )
            .unwrap();
    }

    fn process_stopped_at_svc(
        execution_state: ExecutionState,
    ) -> (RunnableProcess, crate::ExecutionReport, u64) {
        let encoding = match execution_state {
            ExecutionState::A64 => 0xd400_4681,
            ExecutionState::A32 => 0xef12_3456,
            ExecutionState::T32 => 0xbf00_df7b,
        };
        let (_directory, plan) = plan();
        let mut process = ProcessBuilder::new().build(&plan).unwrap();
        replace_entry_instruction(&mut process, encoding);
        let entry = process.entry_module().entry_address();
        if execution_state != ExecutionState::A64 {
            let mut state = match execution_state {
                ExecutionState::A32 => nixe_cpu::state::A32State::a32(),
                ExecutionState::T32 => nixe_cpu::state::A32State::t32(),
                ExecutionState::A64 => unreachable!(),
            };
            state
                .set_instruction_address(u32::try_from(entry).unwrap())
                .unwrap();
            process.main_thread.state = ThreadCpuState::A32(Box::new(state));
        }
        let report = process.run_reference(1).unwrap();
        assert!(matches!(
            report.stop,
            crate::ExecutionStop::SupervisorCall { .. }
        ));
        (process, report, entry)
    }

    fn instruction_address(state: &ThreadCpuState) -> u64 {
        match state {
            ThreadCpuState::A64(state) => state.pc(),
            ThreadCpuState::A32(state) => u64::from(state.instruction_address()),
        }
    }

    fn synthetic_nro() -> Vec<u8> {
        let mut bytes = vec![0; 0x2800];
        bytes[..4].copy_from_slice(&0x1400_0020_u32.to_le_bytes()); // B entry + 0x80
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

    #[test]
    fn builder_propagates_runtime_diagnostics_to_cpu_resources() {
        let builder = ProcessBuilder::new();
        assert_eq!(
            builder.cpu_diagnostics().report_detail,
            nixe_cpu::coverage::MissingInstructionReportDetail::Detailed
        );
    }

    #[test]
    fn npdm_address_space_values_keep_distinct_runtime_meanings() {
        assert_eq!(
            ProcessAddressSpace::from_npdm(AddressSpaceType::AddressSpace32Bit),
            ProcessAddressSpace::Bit32
        );
        assert_eq!(
            ProcessAddressSpace::from_npdm(AddressSpaceType::AddressSpace32BitNoReserved),
            ProcessAddressSpace::Bit32NoReserved
        );
        assert_eq!(
            ProcessAddressSpace::from_npdm(AddressSpaceType::AddressSpace64BitOld),
            ProcessAddressSpace::Bit64Old
        );
        assert_eq!(
            ProcessAddressSpace::from_npdm(AddressSpaceType::AddressSpace64Bit),
            ProcessAddressSpace::Bit64
        );
        assert!(validate_range(ProcessAddressSpace::Bit32, u64::from(u32::MAX), 2).is_err());
    }

    #[test]
    fn horizon_layout_profiles_keep_allocation_windows_and_resource_limits_distinct() {
        let code_start = 0x7100_0000;
        let code_end = 0x7100_4000;
        let limit = 0x1234_0000;
        let layout = ProcessMemoryLayout::for_address_space(
            ProcessMemoryLayoutProfile::Horizon2Plus,
            ProcessAddressSpace::Bit64,
            code_start,
            code_end,
            limit,
        )
        .unwrap();
        assert_eq!(layout.aslr().base().get(), 0x0800_0000);
        assert_eq!(layout.aslr().end(), 1_u64 << 39);
        assert!(layout.aslr().base().get() <= layout.stack().base().get());
        assert!(layout.stack().end() <= layout.aslr().end());
        assert!(layout.alias().end() <= layout.aslr().end());
        assert!(layout.heap().end() <= layout.aslr().end());
        assert!(layout.stack().base().get() >= 0x7120_0000);
        assert_eq!(layout.alias().size(), 0x10_0000_0000);
        assert_eq!(layout.heap().size(), 0x2_0000_0000);
        assert_eq!(layout.stack().size(), 0x8000_0000);
        assert_eq!(layout.memory_capacity(), limit);

        let high_code_start = 0x64_0000_0000;
        let high_layout = ProcessMemoryLayout::for_address_space(
            ProcessMemoryLayoutProfile::Horizon2Plus,
            ProcessAddressSpace::Bit64,
            high_code_start,
            high_code_start + HORIZON_REGION_ALIGNMENT,
            limit,
        )
        .unwrap();
        assert!(high_layout.heap().end() <= high_code_start);

        let without_alias = ProcessMemoryLayout::for_address_space(
            ProcessMemoryLayoutProfile::Horizon2Plus,
            ProcessAddressSpace::Bit32NoReserved,
            0x0020_0000,
            0x0040_0000,
            limit,
        )
        .unwrap();
        assert_eq!(without_alias.alias().size(), 0);
        assert_eq!(without_alias.heap().base().get(), 0x4000_0000);
        assert_eq!(without_alias.heap().size(), 0x8000_0000);

        let deprecated = ProcessMemoryLayout::for_address_space(
            ProcessMemoryLayoutProfile::Horizon1,
            ProcessAddressSpace::Bit64Old,
            0x0800_0000,
            0x0820_0000,
            limit,
        )
        .unwrap();
        assert_eq!(deprecated.aslr().end(), 1_u64 << 36);
        assert_eq!(deprecated.alias().size(), 0x1_8000_0000);
        assert_eq!(deprecated.heap().size(), 0x2_0000_0000);
        assert!(
            ProcessMemoryLayout::for_address_space(
                ProcessMemoryLayoutProfile::Horizon1,
                ProcessAddressSpace::Bit64,
                code_start,
                code_end,
                limit,
            )
            .is_err()
        );
    }

    #[test]
    fn a32_thread_initialization_uses_32_bit_pc_stack_and_tls() {
        let cpu = ProcessCpuContext::new(GuestCpuProfile::switch_1(), AddressSpaceId::new(7));
        let configuration = cpu.thread_configuration(ExecutionState::A32).unwrap();
        let mut state = ThreadCpuState::new(configuration);
        initialize_thread(
            &mut state,
            GuestVirtualAddress::new(0x0020_0000),
            GuestVirtualAddress::new(0x0080_0000),
            GuestVirtualAddress::new(0x0090_0000),
            1,
            None,
            None,
        )
        .unwrap();
        let ThreadCpuState::A32(state) = state else {
            panic!("A32 metadata must create AArch32 state");
        };
        assert_eq!(state.instruction_address(), 0x0020_0000);
        assert_eq!(state.read_r(a32_register(13)), 0x0080_0000);
        assert_eq!(state.tpidrurw(), 0x0090_0000);
        assert_eq!(state.tpidruro(), 0x0090_0000);
        assert_eq!(state.read_r(a32_register(1)), 1);
    }

    #[test]
    fn synthetic_launch_translates_entry_only_through_process_memory() {
        let (_directory, plan) = plan();
        let process = ProcessBuilder::new().build(&plan).unwrap();
        let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
        assert_eq!(
            process
                .memory()
                .fetch32(process.cpu_context().address_space_id(), entry)
                .unwrap()
                .bits,
            0x1400_0020
        );
        let dump = process.print_entry_ir().unwrap();
        assert!(dump.contains(" A64 "));
        assert!(dump.contains("raw=0x14000020"));
        assert!(dump.contains("guest=\"b imm=#128\""));
        let report = process.print_entry_report();
        assert!(report.starts_with("nixe-frontend-block-report-v1\n"));
        assert!(report.contains("outcome=translated end=direct-branch"));
        assert!(report.contains("ir-dump stage=pre-optimization"));
        assert!(report.contains("dependency page="));
        assert_eq!(
            process.main_thread().state.execution_state(),
            ExecutionState::A64
        );
        let ThreadCpuState::A64(state) = &process.main_thread().state else {
            panic!("homebrew fixture must initialize A64");
        };
        assert_eq!(
            process
                .handles()
                .get_as::<crate::ThreadObject>(process.main_thread().handle),
            Some(&crate::ThreadObject::new(1))
        );
        assert!(process.mounts().primary().is_none());
        assert!(process.mounts().add_ons().is_empty());
        assert_eq!(state.pc(), entry.get());
        assert_eq!(
            state.read_x(A64Register::StackPointer),
            process.main_thread().stack_top.get()
        );
        assert_eq!(state.tpidr_el0(), process.main_thread().tls_base.get());
        let context = process.main_thread().abi_context.unwrap();
        assert_eq!(
            state.read_x(A64Register::General(a64_register(0))),
            context.get()
        );
        assert_eq!(
            state.read_x(A64Register::General(a64_register(1))),
            u64::MAX
        );
        let loader_return = process.main_thread().loader_return.unwrap();
        assert_eq!(
            state.read_x(A64Register::General(a64_register(30))),
            loader_return.get()
        );
        assert_eq!(
            process
                .memory()
                .mapping_info(process.cpu_context().address_space_id(), loader_return)
                .unwrap()
                .permissions,
            MemoryPermissions::READ_EXECUTE
        );
        assert_eq!(
            process
                .memory()
                .fetch32(process.cpu_context().address_space_id(), loader_return)
                .unwrap()
                .bits,
            HOME_BREW_EXIT_PROCESS_INSTRUCTION
        );
        assert_eq!(
            process
                .memory()
                .read(
                    process.cpu_context().address_space_id(),
                    context,
                    MemoryAccess::normal(MemoryAccessSize::Word),
                )
                .unwrap()
                .value,
            MemoryValue::U32(HOME_BREW_MAIN_THREAD_HANDLE_KEY)
        );
        assert_eq!(
            process
                .memory()
                .read(
                    process.cpu_context().address_space_id(),
                    context.checked_add(8).unwrap(),
                    MemoryAccess::normal(MemoryAccessSize::Doubleword),
                )
                .unwrap()
                .value,
            MemoryValue::U64(u64::from(process.main_thread().handle))
        );
        assert_eq!(
            process
                .memory()
                .read(
                    process.cpu_context().address_space_id(),
                    context
                        .checked_add(HOME_BREW_CONFIG_ENTRY_SIZE as u64)
                        .unwrap(),
                    MemoryAccess::normal(MemoryAccessSize::Word),
                )
                .unwrap()
                .value,
            MemoryValue::U32(0)
        );
    }

    #[test]
    fn nro_loader_return_preserves_x0_and_exits_without_executing_the_gateway() {
        let (_directory, plan) = plan();
        let mut process = ProcessBuilder::new().build(&plan).unwrap();
        replace_entry_instruction(&mut process, 0xd65f_03c0); // RET X30
        let loader_return = process.main_thread().loader_return.unwrap();
        let ThreadCpuState::A64(state) = &mut process.main_thread.state else {
            panic!("homebrew fixture must initialize A64");
        };
        state.write_x(A64Register::General(a64_register(0)), 0x1234_5678);

        let report = process.run_reference(1).unwrap();

        assert_eq!(report.instructions_executed, 1);
        assert_eq!(
            report.stop,
            crate::ExecutionStop::LoaderReturn {
                source: LocationDescriptor::new(
                    loader_return,
                    ExecutionState::A64,
                    process.cpu_context().profile().id(),
                ),
                result_code: 0x1234_5678,
            }
        );
        assert_eq!(process.execution_status(), ProcessExecutionStatus::Exited);
        assert_eq!(
            process.exit(),
            Some(ProcessExit {
                cause: ProcessExitCause::LoaderReturned,
                exit_code: 0x1234_5678,
                source: Some(LocationDescriptor::new(
                    loader_return,
                    ExecutionState::A64,
                    process.cpu_context().profile().id(),
                )),
                thread_id: 1,
            })
        );
        assert_eq!(
            process.main_thread().exit(),
            Some(ThreadExit {
                requested_scope: ExceptionTerminationScope::Process,
                exit_code: 0x1234_5678,
                source: Some(LocationDescriptor::new(
                    loader_return,
                    ExecutionState::A64,
                    process.cpu_context().profile().id(),
                )),
            })
        );
        assert!(matches!(
            process.run_reference(1),
            Err(ProcessExecutionError::NotRunnable {
                status: ProcessExecutionStatus::Exited,
                ..
            })
        ));
        let teardown = process.teardown();
        assert_eq!(teardown.exit.unwrap().exit_code, 0x1234_5678);
    }

    #[test]
    fn image_base_is_relocatable_without_changing_pc_relative_translation() {
        let (_directory, plan) = plan();
        let first = ProcessBuilder::new()
            .with_config(ProcessBuildConfig {
                image_base: GuestVirtualAddress::new(0x7100_0000),
                ..ProcessBuildConfig::default()
            })
            .build(&plan)
            .unwrap();
        let second = ProcessBuilder::new()
            .with_config(ProcessBuildConfig {
                image_base: GuestVirtualAddress::new(0x7200_0000),
                ..ProcessBuildConfig::default()
            })
            .build(&plan)
            .unwrap();
        assert_eq!(
            second.entry_module().entry_address() - first.entry_module().entry_address(),
            0x0100_0000
        );
        let first_block = first.translate_entry().unwrap();
        let second_block = second.translate_entry().unwrap();
        let direct_target = |block: &IrBlock| match block.terminator {
            Terminator::Direct {
                target: ControlTarget::Direct { pc, .. },
            } => pc.get(),
            ref terminator => panic!("unexpected terminator {terminator:?}"),
        };
        assert_eq!(
            direct_target(&second_block) - direct_target(&first_block),
            0x0100_0000
        );
        assert_eq!(
            second.modules()[0].mappings()[0].guest_address()
                - first.modules()[0].mappings()[0].guest_address(),
            0x0100_0000
        );
    }

    #[test]
    fn writable_code_alias_updates_the_fetched_generation() {
        let (_directory, plan) = plan();
        let mut process = ProcessBuilder::new().build(&plan).unwrap();
        let space = process.cpu.address_space_id();
        let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
        let before = process.memory.fetch32(space, entry).unwrap().dependencies;
        let mapping = process.memory.mapping_info(space, entry).unwrap();
        let alias = GuestVirtualAddress::new(0x7000_0000);
        assert!(process.memory.map_page(
            space,
            alias,
            mapping.physical_page,
            MemoryPermissions::READ_WRITE
        ));
        process
            .memory
            .write(
                space,
                alias,
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0xd503_201f),
            )
            .unwrap();
        let after = process.memory.fetch32(space, entry).unwrap().dependencies;
        assert_ne!(before, after);
    }

    #[test]
    fn reference_execution_honors_budget_and_preserves_dispatch_pc() {
        let (_directory, plan) = plan();
        let mut process = ProcessBuilder::new().build(&plan).unwrap();
        let entry = process.entry_module().entry_address();

        let report = process.run_reference(1).unwrap();
        assert_eq!(report.instructions_executed, 1);
        assert_eq!(report.stop, crate::ExecutionStop::BudgetExhausted);
        assert!(report.stop.exception_dispatch_request().is_none());
        assert!(!report.trace.enabled());
        assert!(report.trace.entries().is_empty());
        assert_eq!(
            process.execution_status(),
            crate::ProcessExecutionStatus::Ready
        );
        let nixe_cpu::state::RegisterContext::A64(context) = &report.context else {
            panic!("homebrew fixture must report A64 context");
        };
        assert_eq!(context.pc.get(), entry + 0x80);
        assert!(report.to_string().contains("flags=N0Z0C0V0"));
        let ThreadCpuState::A64(state) = &process.main_thread().state else {
            panic!("homebrew fixture must initialize A64");
        };
        assert_eq!(state.pc(), entry + 0x80);
    }

    #[test]
    fn reference_execution_observes_safepoints_before_fetch() {
        let (_directory, plan) = plan();
        let mut process = ProcessBuilder::new().build(&plan).unwrap();
        let entry = process.entry_module().entry_address();
        process.request_safepoint();

        let report = process.run_reference(10).unwrap();
        assert_eq!(report.instructions_executed, 0);
        assert_eq!(report.stop, crate::ExecutionStop::Safepoint);
        let ThreadCpuState::A64(state) = &process.main_thread().state else {
            panic!("homebrew fixture must initialize A64");
        };
        assert_eq!(state.pc(), entry);
    }

    #[test]
    fn reference_execution_observes_pending_events_before_fetch() {
        let (_directory, plan) = plan();
        let mut process = ProcessBuilder::new().build(&plan).unwrap();
        process.post_event(0b0101);

        let report = process.run_reference(10).unwrap();
        assert_eq!(report.instructions_executed, 0);
        assert_eq!(
            report.stop,
            crate::ExecutionStop::PendingEvent { mask: 0b0101 }
        );
    }

    #[test]
    fn reference_execution_reports_instruction_fetch_faults_as_a_distinct_stop() {
        let (_directory, plan) = plan();
        let mut process = ProcessBuilder::new().build(&plan).unwrap();
        let ThreadCpuState::A64(state) = &mut process.main_thread.state else {
            panic!("homebrew fixture must initialize A64");
        };
        state.set_pc(0x1000);

        let report = process.run_reference(1).unwrap();
        assert_eq!(report.instructions_executed, 0);
        assert!(matches!(
            report.stop,
            crate::ExecutionStop::FetchFault { .. }
        ));
        let nixe_cpu::state::RegisterContext::A64(context) = &report.context else {
            panic!("homebrew fixture must report A64 context");
        };
        assert_eq!(context.pc.get(), 0x1000);
        assert!(report.to_string().contains("fetch-fault"));
        assert_eq!(
            process.execution_status(),
            crate::ProcessExecutionStatus::Faulted
        );
    }

    #[test]
    fn unallocated_encoding_suspends_until_runtime_resumes_thread() {
        let (_directory, plan) = plan();
        let mut process = ProcessBuilder::new().build(&plan).unwrap();

        let report = process.run_reference(2).unwrap();
        assert!(matches!(
            report.stop,
            crate::ExecutionStop::UnallocatedEncoding { .. }
        ));
        assert_eq!(
            process.execution_status(),
            crate::ProcessExecutionStatus::Suspended
        );
        assert!(matches!(
            process.run_reference(1),
            Err(crate::ProcessExecutionError::NotRunnable {
                status: crate::ProcessExecutionStatus::Suspended,
                ..
            })
        ));
        assert!(process.resume());
        assert_eq!(
            process.execution_status(),
            crate::ProcessExecutionStatus::Ready
        );
    }

    #[test]
    fn reference_execution_distinguishes_unsupported_profile_and_unallocated_code() {
        let (_directory, plan) = plan();

        let mut unsupported = ProcessBuilder::new().build(&plan).unwrap();
        replace_entry_instruction(&mut unsupported, 0xd503_203f); // YIELD
        let report = unsupported.run_reference(1).unwrap();
        assert!(matches!(
            report.stop,
            crate::ExecutionStop::UnsupportedSemantics { .. }
        ));
        assert_eq!(
            unsupported.execution_status(),
            ProcessExecutionStatus::Faulted
        );
        assert!(report.to_string().contains("unsupported-semantics"));

        let mut profile_disabled = ProcessBuilder::new()
            .with_config(ProcessBuildConfig {
                cpu_profile: GuestCpuProfile::switch_2_native(),
                ..ProcessBuildConfig::default()
            })
            .build(&plan)
            .unwrap();
        replace_entry_instruction(&mut profile_disabled, 0x4e22_1c20);
        let report = profile_disabled.run_reference(1).unwrap();
        assert_eq!(
            report.stop.exception_dispatch_request().unwrap().kind(),
            nixe_cpu::exception::ExceptionKind::UndefinedInstruction
        );
        assert!(matches!(
            report.stop,
            crate::ExecutionStop::ProfileDisabled { .. }
        ));
        assert!(report.to_string().contains("profile-disabled"));

        let mut unallocated = ProcessBuilder::new().build(&plan).unwrap();
        replace_entry_instruction(&mut unallocated, 0);
        let report = unallocated.run_reference(1).unwrap();
        assert_eq!(
            report.stop.exception_dispatch_request().unwrap().kind(),
            nixe_cpu::exception::ExceptionKind::UndefinedInstruction
        );
        assert!(matches!(
            report.stop,
            crate::ExecutionStop::UnallocatedEncoding { .. }
        ));
        assert!(report.to_string().contains("unallocated-encoding"));
    }

    #[test]
    fn reference_execution_distinguishes_svc_architectural_and_data_fault_stops() {
        let (_directory, plan) = plan();

        let mut svc = ProcessBuilder::new().build(&plan).unwrap();
        replace_entry_instruction(&mut svc, 0xd400_0841); // SVC #0x42
        let report = svc.run_reference(1).unwrap();
        let dispatch = report.stop.exception_dispatch_request().unwrap();
        assert_eq!(
            dispatch.kind(),
            nixe_cpu::exception::ExceptionKind::SupervisorCall
        );
        assert_eq!(dispatch.syndrome(), Some(0x42));
        assert!(matches!(
            report.stop,
            crate::ExecutionStop::SupervisorCall {
                immediate: 0x42,
                ..
            }
        ));
        assert!(report.to_string().contains("supervisor-call"));

        let mut breakpoint = ProcessBuilder::new().build(&plan).unwrap();
        replace_entry_instruction(&mut breakpoint, 0xd420_2460); // BRK #0x123
        let report = breakpoint.run_reference(1).unwrap();
        let dispatch = report.stop.exception_dispatch_request().unwrap();
        assert_eq!(
            dispatch.kind(),
            nixe_cpu::exception::ExceptionKind::Breakpoint
        );
        assert_eq!(dispatch.syndrome(), Some(0x123));
        assert!(matches!(
            report.stop,
            crate::ExecutionStop::ArchitecturalException {
                kind: nixe_cpu::exception::ExceptionKind::Breakpoint,
                syndrome: Some(0x123),
                ..
            }
        ));
        assert!(report.to_string().contains("architectural-exception"));

        let mut data_fault = ProcessBuilder::new().build(&plan).unwrap();
        replace_entry_instruction(&mut data_fault, 0xf940_0020); // LDR X0,[X1]
        let ThreadCpuState::A64(state) = &mut data_fault.main_thread.state else {
            panic!("homebrew fixture must initialize A64");
        };
        state.write_x(
            nixe_cpu::state::a64::A64Register::General(a64_register(1)),
            0x1000,
        );
        let report = data_fault.run_reference(1).unwrap();
        assert_eq!(
            report.stop.exception_dispatch_request().unwrap().kind(),
            nixe_cpu::exception::ExceptionKind::DataAbort
        );
        assert!(matches!(
            report.stop,
            crate::ExecutionStop::DataFault { .. }
        ));
        assert!(report.to_string().contains("data-fault"));
    }

    #[test]
    fn supervisor_calls_route_a64_a32_and_t32_with_current_runtime_context() {
        let cases = [
            (ExecutionState::A64, 0xd400_4681, 0x234),
            (ExecutionState::A32, 0xef12_3456, 0x12_3456),
            (ExecutionState::T32, 0xbf00_df7b, 0x7b),
        ];

        for (execution_state, encoding, immediate) in cases {
            let (_directory, plan) = plan();
            let mut process = ProcessBuilder::new().build(&plan).unwrap();
            replace_entry_instruction(&mut process, encoding);
            let entry = process.entry_module().entry_address();
            if execution_state != ExecutionState::A64 {
                let mut state = match execution_state {
                    ExecutionState::A32 => nixe_cpu::state::A32State::a32(),
                    ExecutionState::T32 => nixe_cpu::state::A32State::t32(),
                    ExecutionState::A64 => unreachable!(),
                };
                state
                    .set_instruction_address(u32::try_from(entry).unwrap())
                    .unwrap();
                process.main_thread.state = ThreadCpuState::A32(Box::new(state));
            }

            let report = process.run_reference(1).unwrap();
            let expected_encoding = match execution_state {
                ExecutionState::T32 => InstructionEncoding::from_u16(encoding as u16),
                ExecutionState::A64 | ExecutionState::A32 => {
                    InstructionEncoding::from_u32(encoding)
                }
            };
            let mut dispatcher = RecordingSupervisorCallDispatcher {
                expected_encoding: Some(expected_encoding),
                observed: None,
            };
            let outcome = process
                .route_supervisor_call(&report.stop, &mut dispatcher)
                .unwrap();

            assert_eq!(outcome, crate::ExceptionHandlingResult::Suspended);
            let (request, address_space, thread_id, handle) = dispatcher.observed.unwrap();
            assert_eq!(request.kind(), ExceptionKind::SupervisorCall);
            assert_eq!(request.syndrome(), Some(immediate));
            assert_eq!(request.source().pc.get(), entry);
            assert_eq!(request.source().execution_state, execution_state);
            assert_eq!(address_space, process.cpu_context().address_space_id());
            assert_eq!(thread_id, 1);
            assert_eq!(handle, process.main_thread().handle);
            match &process.main_thread().state {
                ThreadCpuState::A64(state) => assert_eq!(
                    state.read_x(nixe_cpu::state::a64::A64Register::General(a64_register(0))),
                    0xfeed_face
                ),
                ThreadCpuState::A32(state) => {
                    assert_eq!(state.read_r(a32_register(0)), 0xfeed_face)
                }
            }
        }
    }

    #[test]
    fn handled_supervisor_calls_advance_once_in_a64_a32_and_t32() {
        let cases = [
            (ExecutionState::A64, 4_u64),
            (ExecutionState::A32, 4_u64),
            (ExecutionState::T32, 2_u64),
        ];

        for (execution_state, width) in cases {
            let (mut process, report, entry) = process_stopped_at_svc(execution_state);
            let mut dispatcher = FixedSupervisorCallDispatcher {
                outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Resume(
                    crate::ExceptionResume::Next,
                )),
            };

            let result = process
                .route_supervisor_call(&report.stop, &mut dispatcher)
                .unwrap();

            assert_eq!(result, crate::ExceptionHandlingResult::Resumed);
            assert_eq!(process.execution_status(), ProcessExecutionStatus::Ready);
            assert_eq!(
                instruction_address(&process.main_thread.state),
                entry + width
            );
            let next = process.run_reference(1).unwrap();
            assert!(!matches!(
                next.stop,
                crate::ExecutionStop::SupervisorCall { source, .. } if source.pc.get() == entry
            ));
        }
    }

    #[test]
    fn supervisor_call_retry_is_explicit_and_reexecutes_the_source() {
        for execution_state in [
            ExecutionState::A64,
            ExecutionState::A32,
            ExecutionState::T32,
        ] {
            let (mut process, report, entry) = process_stopped_at_svc(execution_state);
            let mut dispatcher = PcMutatingSupervisorCallDispatcher {
                outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Resume(
                    crate::ExceptionResume::Retry,
                )),
            };

            assert_eq!(
                process
                    .route_supervisor_call(&report.stop, &mut dispatcher)
                    .unwrap(),
                crate::ExceptionHandlingResult::Resumed
            );
            assert_eq!(instruction_address(&process.main_thread.state), entry);
            let retried = process.run_reference(1).unwrap();
            assert!(matches!(
                retried.stop,
                crate::ExecutionStop::SupervisorCall { source, .. } if source.pc.get() == entry
            ));
        }
    }

    #[test]
    fn suspended_supervisor_call_installs_continuation_without_becoming_runnable() {
        let (mut process, report, entry) = process_stopped_at_svc(ExecutionState::A64);
        let mut dispatcher = FixedSupervisorCallDispatcher {
            outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Suspend(
                crate::ExceptionResume::Next,
            )),
        };

        assert_eq!(
            process
                .route_supervisor_call(&report.stop, &mut dispatcher)
                .unwrap(),
            crate::ExceptionHandlingResult::Suspended
        );
        assert_eq!(instruction_address(&process.main_thread.state), entry + 4);
        assert_eq!(
            process.execution_status(),
            ProcessExecutionStatus::Suspended
        );
        assert!(matches!(
            process.run_reference(1),
            Err(crate::ProcessExecutionError::NotRunnable {
                status: ProcessExecutionStatus::Suspended,
                ..
            })
        ));
        assert!(process.resume());
        assert_eq!(instruction_address(&process.main_thread.state), entry + 4);
    }

    #[test]
    fn faulted_supervisor_call_retains_source_and_cannot_run() {
        let (mut process, report, entry) = process_stopped_at_svc(ExecutionState::A64);
        let mut dispatcher = PcMutatingSupervisorCallDispatcher {
            outcome: Some(crate::ExceptionDispatchOutcome::Fault(
                "svc dispatch failed",
            )),
        };

        assert_eq!(
            process
                .route_supervisor_call(&report.stop, &mut dispatcher)
                .unwrap(),
            crate::ExceptionHandlingResult::Fault("svc dispatch failed")
        );
        assert_eq!(instruction_address(&process.main_thread.state), entry);
        assert_eq!(process.execution_status(), ProcessExecutionStatus::Faulted);
        assert!(matches!(
            process.run_reference(1),
            Err(crate::ProcessExecutionError::NotRunnable {
                status: ProcessExecutionStatus::Faulted,
                ..
            })
        ));
        assert!(!process.resume());
    }

    #[test]
    fn detailed_instruction_trace_is_opt_in_bounded_and_persistent_across_slices() {
        let (_directory, plan) = plan();
        let mut process = ProcessBuilder::new()
            .with_diagnostics(crate::DiagnosticsPolicy {
                instruction_trace: true,
                ..crate::DiagnosticsPolicy::default()
            })
            .build(&plan)
            .unwrap();
        replace_entry_instruction(&mut process, 0x1400_0000); // B #0

        let first = process
            .run_reference(crate::MAX_INSTRUCTION_TRACE_ENTRIES as u64 + 3)
            .unwrap();
        assert!(first.trace.enabled());
        assert_eq!(
            first.trace.entries().len(),
            crate::MAX_INSTRUCTION_TRACE_ENTRIES
        );
        assert_eq!(first.trace.discarded(), 3);
        assert_eq!(first.trace.entries()[0].sequence, 3);
        assert_eq!(
            first.trace.entries().last().unwrap().sequence,
            crate::MAX_INSTRUCTION_TRACE_ENTRIES as u64 + 2
        );
        assert!(
            first
                .trace
                .entries()
                .iter()
                .all(|entry| entry.disassembly.as_deref() == Some("b imm=#0"))
        );
        assert!(first.trace.to_string().len() <= crate::MAX_INSTRUCTION_TRACE_EXPORT_BYTES);

        let second = process.run_reference(1).unwrap();
        assert_eq!(second.trace.discarded(), 4);
        assert_eq!(second.trace.entries()[0].sequence, 4);
        assert_eq!(
            second.trace.entries().last().unwrap().sequence,
            crate::MAX_INSTRUCTION_TRACE_ENTRIES as u64 + 3
        );
    }

    #[test]
    fn sanitized_instruction_trace_omits_detailed_disassembly() {
        let (_directory, plan) = plan();
        let mut process = ProcessBuilder::new()
            .with_diagnostics(crate::DiagnosticsPolicy {
                report_detail: crate::ReportDetail::Sanitized,
                instruction_trace: true,
                ..crate::DiagnosticsPolicy::default()
            })
            .build(&plan)
            .unwrap();

        let report = process.run_reference(1).unwrap();
        assert_eq!(report.trace.entries().len(), 1);
        assert!(report.trace.entries()[0].disassembly.is_none());
        assert!(!report.trace.to_string().contains("disassembly="));
    }

    #[test]
    fn teardown_reports_resources_owned_by_the_process() {
        let (_directory, plan) = plan();
        let mut process = ProcessBuilder::new().build(&plan).unwrap();
        assert!(process.terminate());
        assert_eq!(
            process.exit().unwrap().cause,
            crate::ProcessExitCause::HostRequested
        );
        assert_eq!(
            process.main_thread().exit().unwrap().requested_scope,
            crate::ExceptionTerminationScope::Process
        );

        let report = process.teardown();
        assert_eq!(
            report.previous_status,
            crate::ProcessExecutionStatus::Exited
        );
        assert_eq!(
            report.exit.unwrap().cause,
            crate::ProcessExitCause::HostRequested
        );
        assert_eq!(report.threads_released, 1);
        assert_eq!(report.modules_released, 1);
        assert!(report.mappings_released > 0);
        assert!(report.physical_pages_released > 0);
        assert_eq!(report.mounts_released, 0);
        assert_eq!(report.handles_released, 1);
    }
}
