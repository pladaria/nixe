//! Construction of a runnable CPU process from an immutable launch plan.

mod builder;
mod execution;
mod layout;
mod thread;

pub use builder::ProcessBuilder;
use builder::{ThreadPolicy, align_up, error, initialize_created_thread};
#[cfg(test)]
use builder::{a32_register, a64_register, initialize_thread, validate_range};
pub use execution::{
    ExecutionReport, ExecutionStop, InstructionTrace, InstructionTraceEntry,
    MAX_INSTRUCTION_TRACE_ENTRIES, MAX_INSTRUCTION_TRACE_EXPORT_BYTES, MAX_TRACE_DISASSEMBLY_BYTES,
    ProcessExecutionError, ProcessExecutionStatus, ProcessExit, ProcessExitCause,
    ProcessTeardownReport, ThreadExit,
};
pub use layout::{
    ProcessAddressSpace, ProcessBuildConfig, ProcessMemoryLayout, ProcessMemoryLayoutProfile,
    ProcessVirtualRegion,
};
pub use thread::{
    GuestThread, MainThread, ThreadCreateError, ThreadCreateRequest, ThreadCreation, ThreadTable,
    ThreadTableError,
};

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::sync::Arc;

use nixe_cpu::ir::block::IrBlock;
use nixe_cpu::ir::print::{IrPrintOptions, print_block};
use nixe_cpu::location::{ExecutionState, LocationDescriptor};
use nixe_cpu::memory::{
    CpuMemory, ExecutionMemory, MemoryMappingPurpose, MemoryPermissions, ProcessMemory,
    SYNTHETIC_PAGE_SIZE, SyntheticRamPage,
};
use nixe_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::{ThreadCpuState, a32::A32GeneralRegister, a64::A64Register};
use nixe_cpu::translate::{
    BlockTranslationConfig, BlockTranslationReport, translate_block, translate_block_report,
};
use nixe_loader_executable::{
    AddressSpaceType, ExternalSymbol, PreparationConfig, PreparedModule, SymbolResolution,
};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

use crate::exception_dispatch::ExceptionProcessMetadata;
use crate::{
    ExceptionDispatchContext, ExceptionDispatchOutcome, ExceptionDispatcher,
    ExceptionHandlingResult, ExceptionProcessContext, ExceptionResume, ExceptionRouteError,
    ExceptionTerminationReason, ExceptionTerminationScope, ExceptionThreadContext, LaunchKind,
    LaunchModuleImage, LaunchPlan, install_prepared_module,
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

/// A process whose executable bytes are visible only through process memory.
pub struct RunnableProcess {
    process_id: u64,
    lifecycle: nixe_scheduler::ProcessLifecycle,
    process_exit: Option<ProcessExit>,
    cpu: ProcessCpuContext,
    address_space: ProcessAddressSpace,
    memory_layout: ProcessMemoryLayout,
    random_entropy: [u64; 4],
    heap_size: u64,
    initial_memory_size: u64,
    memory: ExecutionMemory,
    modules: Box<[PreparedModule]>,
    entry_module: usize,
    main_thread_id: nixe_scheduler::GuestThreadId,
    initial_thread_priority: i32,
    initial_ideal_vcpu: nixe_scheduler::VirtualCpuId,
    thread_policy: Option<ThreadPolicy>,
    next_thread_tls: GuestVirtualAddress,
    free_thread_tls: std::collections::BTreeSet<GuestVirtualAddress>,
    threads: crate::ThreadTable,
    mounts: crate::ProcessMountNamespace,
    handles: crate::HandleTable,
    address_waits: crate::AddressWaitRegistry,
    execution: execution::ProcessExecutionControl,
}

impl RunnableProcess {
    #[must_use]
    pub const fn process_id(&self) -> u64 {
        self.process_id
    }

    pub(crate) const fn assign_process_id(&mut self, id: nixe_scheduler::ProcessId) {
        self.process_id = id.get();
    }

    #[must_use]
    pub const fn lifecycle(&self) -> nixe_scheduler::ProcessLifecycle {
        self.lifecycle
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
    pub const fn memory(&self) -> &ExecutionMemory {
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
    pub fn main_thread(&self) -> &crate::MainThread {
        self.threads
            .get(self.main_thread_id)
            .expect("a runnable process always retains its main thread")
    }

    /// Returns mutable main-thread state for runtime scheduling and ABI setup.
    pub fn main_thread_mut(&mut self) -> &mut crate::MainThread {
        self.threads
            .get_mut(self.main_thread_id)
            .expect("a runnable process always retains its main thread")
    }

    #[must_use]
    pub const fn main_thread_id(&self) -> nixe_scheduler::GuestThreadId {
        self.main_thread_id
    }

    #[must_use]
    pub const fn initial_thread_priority(&self) -> i32 {
        self.initial_thread_priority
    }

    #[must_use]
    pub const fn initial_ideal_vcpu(&self) -> nixe_scheduler::VirtualCpuId {
        self.initial_ideal_vcpu
    }

    pub(crate) fn validate_thread_request(
        &self,
        request: &ThreadCreateRequest,
    ) -> Result<(), ThreadCreateError> {
        let execution_state = match &self.main_thread().state {
            ThreadCpuState::A64(_) => ExecutionState::A64,
            ThreadCpuState::A32(state) => state.execution_state(),
        };
        let entry_alignment = if execution_state == ExecutionState::T32 {
            2
        } else {
            4
        };
        if request.entry.get() >= self.address_space.exclusive_limit()
            || !request.entry.get().is_multiple_of(entry_alignment)
        {
            return Err(ThreadCreateError::InvalidEntry);
        }
        let limit = GuestVirtualAddress::new(self.address_space.exclusive_limit());
        let entry_mapping =
            self.memory
                .query_memory(self.cpu.address_space_id(), request.entry, limit);
        if entry_mapping.is_none_or(|mapping| {
            mapping.region.is_none() || !mapping.permissions.contains(MemoryPermissions::EXECUTE)
        }) {
            return Err(ThreadCreateError::InvalidEntry);
        }
        let stack_alignment = if execution_state == ExecutionState::A64 {
            16
        } else {
            8
        };
        let Some(stack_byte) = request.stack_top.get().checked_sub(1) else {
            return Err(ThreadCreateError::InvalidStack);
        };
        if request.stack_top.get() > self.address_space.exclusive_limit()
            || !request.stack_top.get().is_multiple_of(stack_alignment)
        {
            return Err(ThreadCreateError::InvalidStack);
        }
        let stack_mapping = self.memory.query_memory(
            self.cpu.address_space_id(),
            GuestVirtualAddress::new(stack_byte),
            limit,
        );
        if stack_mapping.is_none_or(|mapping| {
            mapping.region.is_none() || !mapping.permissions.contains(MemoryPermissions::WRITE)
        }) {
            return Err(ThreadCreateError::InvalidStack);
        }
        if let Some(policy) = self.thread_policy
            && (request.priority < policy.highest_priority
                || request.priority > policy.lowest_priority
                || request
                    .affinity
                    .iter()
                    .any(|vcpu| vcpu.get() < policy.min_core || vcpu.get() > policy.max_core))
        {
            return Err(ThreadCreateError::PolicyDenied);
        }
        Ok(())
    }

    pub(crate) fn validate_thread_policy(
        &self,
        priority: i32,
        affinity: &nixe_scheduler::CoreSet,
    ) -> Result<(), ThreadCreateError> {
        if let Some(policy) = self.thread_policy
            && (priority < policy.highest_priority
                || priority > policy.lowest_priority
                || affinity
                    .iter()
                    .any(|vcpu| vcpu.get() < policy.min_core || vcpu.get() > policy.max_core))
        {
            return Err(ThreadCreateError::PolicyDenied);
        }
        Ok(())
    }

    pub(crate) fn commit_created_thread(
        &mut self,
        id: nixe_scheduler::GuestThreadId,
        request: &ThreadCreateRequest,
    ) -> Result<ThreadCreation, ThreadCreateError> {
        if self
            .initial_memory_size
            .saturating_add(self.heap_size)
            .checked_add(TLS_SIZE)
            .is_none_or(|size| size > self.memory_layout.memory_capacity())
        {
            return Err(ThreadCreateError::ResourceLimit);
        }
        let recycled_tls = self.free_thread_tls.pop_first();
        let tls_base = match recycled_tls {
            Some(tls_base) => tls_base,
            None => self
                .next_thread_tls
                .checked_sub(TLS_SIZE)
                .ok_or(ThreadCreateError::ResourceLimit)?,
        };
        if tls_base.get() < self.memory_layout.stack().base().get() {
            return Err(ThreadCreateError::ResourceLimit);
        }
        self.memory
            .resize_zeroed_mapping(
                self.cpu.address_space_id(),
                tls_base,
                0,
                TLS_SIZE,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::ThreadLocal,
            )
            .map_err(|_| ThreadCreateError::ResourceLimit)?;
        let object = crate::ThreadObject::new(id.get());
        let handle = match self.handles.insert(object.clone()) {
            Ok(handle) => handle,
            Err(_) => {
                self.rollback_thread_tls(tls_base);
                if recycled_tls.is_some() {
                    self.free_thread_tls.insert(tls_base);
                }
                return Err(ThreadCreateError::ResourceLimit);
            }
        };
        let execution_state = match &self.main_thread().state {
            ThreadCpuState::A64(_) => ExecutionState::A64,
            ThreadCpuState::A32(state) => state.execution_state(),
        };
        let configuration = self
            .cpu
            .thread_configuration(execution_state)
            .map_err(|_| ThreadCreateError::Internal)?;
        let mut state = ThreadCpuState::new(configuration);
        if initialize_created_thread(&mut state, request, tls_base).is_err() {
            let _ = self.handles.close(handle);
            self.rollback_thread_tls(tls_base);
            if recycled_tls.is_some() {
                self.free_thread_tls.insert(tls_base);
            }
            return Err(ThreadCreateError::Internal);
        }
        let thread = GuestThread {
            id,
            object,
            exit: None,
            lifecycle: nixe_scheduler::ThreadLifecycle::Created,
            wait_reason: None,
            continuation: None,
            state,
            handle,
            stack_bottom: request.stack_top,
            stack_top: request.stack_top,
            tls_base,
            abi_context: None,
            loader_return: None,
        };
        if self.threads.insert(thread).is_err() {
            let _ = self.handles.close(handle);
            self.rollback_thread_tls(tls_base);
            if recycled_tls.is_some() {
                self.free_thread_tls.insert(tls_base);
            }
            return Err(ThreadCreateError::Internal);
        }
        if recycled_tls.is_none() {
            self.next_thread_tls = tls_base;
        }
        self.initial_memory_size += TLS_SIZE;
        Ok(ThreadCreation { id, handle })
    }

    pub(crate) fn rollback_created_thread(&mut self, id: nixe_scheduler::GuestThreadId) {
        if let Ok(thread) = self.threads.remove(id) {
            let _ = self.handles.close(thread.handle);
            self.release_thread_tls(thread.tls_base);
        }
    }

    pub(crate) fn reap_exited_thread(
        &mut self,
        id: nixe_scheduler::GuestThreadId,
    ) -> Result<(), ThreadCreateError> {
        let thread = self.threads.get(id).ok_or(ThreadCreateError::Internal)?;
        if !matches!(
            thread.lifecycle,
            nixe_scheduler::ThreadLifecycle::Exited | nixe_scheduler::ThreadLifecycle::Faulted
        ) {
            return Err(ThreadCreateError::Internal);
        }
        let thread = self
            .threads
            .remove(id)
            .map_err(|_| ThreadCreateError::Internal)?;
        self.release_thread_tls(thread.tls_base);
        Ok(())
    }

    fn release_thread_tls(&mut self, tls_base: GuestVirtualAddress) {
        self.rollback_thread_tls(tls_base);
        self.initial_memory_size = self.initial_memory_size.saturating_sub(TLS_SIZE);
        if tls_base == self.next_thread_tls {
            self.next_thread_tls = self
                .next_thread_tls
                .checked_add(TLS_SIZE)
                .expect("allocated TLS lies below the finite stack-region end");
            while self.free_thread_tls.remove(&self.next_thread_tls) {
                self.next_thread_tls = self
                    .next_thread_tls
                    .checked_add(TLS_SIZE)
                    .expect("freed TLS lies below the finite stack-region end");
            }
        } else {
            self.free_thread_tls.insert(tls_base);
        }
    }

    fn rollback_thread_tls(&self, tls_base: GuestVirtualAddress) {
        let _ = self.memory.resize_zeroed_mapping(
            self.cpu.address_space_id(),
            tls_base,
            TLS_SIZE,
            0,
            MemoryPermissions::READ_WRITE,
            MemoryMappingPurpose::ThreadLocal,
        );
    }

    /// Replaces the provisional main-thread identity before publication. The
    /// shared kernel object receives the same runtime-global identity so copied
    /// handles cannot alias process-local thread numbers.
    pub(crate) fn assign_main_thread_id(
        &mut self,
        id: nixe_scheduler::GuestThreadId,
    ) -> Result<(), crate::ThreadTableError> {
        let old = self.main_thread_id;
        if old == id {
            self.main_thread().object.assign_thread_id(id.get());
            return Ok(());
        }
        self.threads.rekey(old, id)?;
        self.main_thread_id = id;
        self.main_thread().object.assign_thread_id(id.get());
        Ok(())
    }

    #[must_use]
    pub const fn threads(&self) -> &crate::ThreadTable {
        &self.threads
    }

    #[must_use]
    pub fn thread(&self, id: nixe_scheduler::GuestThreadId) -> Option<&crate::GuestThread> {
        self.threads.get(id)
    }

    pub fn thread_mut(
        &mut self,
        id: nixe_scheduler::GuestThreadId,
    ) -> Option<&mut crate::GuestThread> {
        self.threads.get_mut(id)
    }

    pub(crate) fn start_created_thread(
        &mut self,
        id: nixe_scheduler::GuestThreadId,
    ) -> Result<(), nixe_scheduler::LifecycleTransitionError<nixe_scheduler::ThreadLifecycle>> {
        let thread = self
            .threads
            .get_mut(id)
            .expect("the coordinator validated the target thread");
        nixe_scheduler::transition_thread(
            &mut thread.lifecycle,
            nixe_scheduler::ThreadLifecycle::Ready,
        )
    }

    pub(crate) fn set_thread_activity_from_coordinator(
        &mut self,
        id: nixe_scheduler::GuestThreadId,
        paused: bool,
    ) {
        let thread = self
            .threads
            .get_mut(id)
            .expect("the coordinator validated the target thread");
        let target = if paused {
            nixe_scheduler::ThreadLifecycle::Suspended
        } else {
            nixe_scheduler::ThreadLifecycle::Ready
        };
        nixe_scheduler::transition_thread(&mut thread.lifecycle, target)
            .expect("scheduler and runtime activity transitions remain synchronized");
    }

    pub(crate) fn terminate_thread_from_coordinator(
        &mut self,
        id: nixe_scheduler::GuestThreadId,
        exit_code: u64,
        requested_scope: ExceptionTerminationScope,
    ) {
        let thread = self
            .threads
            .get_mut(id)
            .expect("the coordinator owns every scheduled thread");
        nixe_scheduler::transition_thread(
            &mut thread.lifecycle,
            nixe_scheduler::ThreadLifecycle::Terminating,
        )
        .expect("only a live non-running thread is terminated by the coordinator");
        nixe_scheduler::transition_thread(
            &mut thread.lifecycle,
            nixe_scheduler::ThreadLifecycle::Exited,
        )
        .expect("a terminating thread can exit");
        thread.wait_reason = None;
        thread.continuation = None;
        thread.exit = Some(ThreadExit {
            requested_scope,
            exit_code,
            source: None,
        });
        thread.object.signal();
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

    #[must_use]
    pub const fn address_waits(&self) -> &crate::AddressWaitRegistry {
        &self.address_waits
    }

    pub const fn address_waits_mut(&mut self) -> &mut crate::AddressWaitRegistry {
        &mut self.address_waits
    }

    /// Borrows the console-neutral resources needed by a platform service layer.
    pub fn mounts_and_handles_mut(
        &mut self,
    ) -> (&crate::ProcessMountNamespace, &mut crate::HandleTable) {
        (&self.mounts, &mut self.handles)
    }

    /// Returns the host-side lifecycle state of this process.
    #[must_use]
    pub fn execution_status(&self) -> ProcessExecutionStatus {
        match self.lifecycle {
            nixe_scheduler::ProcessLifecycle::Exited => ProcessExecutionStatus::Exited,
            nixe_scheduler::ProcessLifecycle::Faulted => ProcessExecutionStatus::Faulted,
            _ => match self.main_thread().lifecycle {
                nixe_scheduler::ThreadLifecycle::Running => ProcessExecutionStatus::Running,
                nixe_scheduler::ThreadLifecycle::Waiting => ProcessExecutionStatus::Suspended,
                nixe_scheduler::ThreadLifecycle::Suspended => ProcessExecutionStatus::Suspended,
                nixe_scheduler::ThreadLifecycle::Exited => ProcessExecutionStatus::Exited,
                nixe_scheduler::ThreadLifecycle::Faulted => ProcessExecutionStatus::Faulted,
                _ => ProcessExecutionStatus::Ready,
            },
        }
    }

    /// Returns the process exit record retained until teardown.
    #[must_use]
    pub const fn exit(&self) -> Option<ProcessExit> {
        self.process_exit
    }

    /// Returns the selected execution-engine descriptor.
    #[must_use]
    pub fn engine_descriptor(&self) -> nixe_cpu_engine::EngineDescriptor {
        self.execution.engine_descriptor()
    }

    /// Requests a stop before the next reference-engine instruction.
    pub fn request_safepoint(&mut self) {
        self.execution.request_safepoint();
    }

    /// Publishes runtime event bits to be observed at the next safepoint.
    pub fn post_event(&self, mask: u32) {
        self.execution.post_event(mask);
    }

    pub(crate) fn clear_local_exclusive_reservation(&mut self, vcpu: nixe_scheduler::VirtualCpuId) {
        self.execution.clear_local_exclusive_reservation(vcpu);
    }

    /// Resumes a process suspended by an exception or scheduling instruction.
    pub fn resume(&mut self) -> bool {
        self.resume_thread(self.main_thread_id)
    }

    /// Resumes one explicit guest thread. Scheduler-facing code must use this
    /// operation rather than the main-thread compatibility adapter.
    pub(crate) fn resume_thread(&mut self, id: nixe_scheduler::GuestThreadId) -> bool {
        if self.lifecycle != nixe_scheduler::ProcessLifecycle::Running {
            return false;
        }
        let Some(thread) = self.threads.get_mut(id) else {
            return false;
        };
        if thread.lifecycle != nixe_scheduler::ThreadLifecycle::Waiting {
            return false;
        }
        nixe_scheduler::transition_thread(
            &mut thread.lifecycle,
            nixe_scheduler::ThreadLifecycle::Ready,
        )
        .expect("runtime and compatibility execution lifecycles remain synchronized");
        thread.wait_reason = None;
        true
    }

    /// Marks the process exited. Resource release occurs in [`Self::teardown`]
    /// or when the process is dropped.
    pub fn terminate(&mut self) -> bool {
        let thread_id = self.main_thread().object.thread_id();
        let exit = ProcessExit {
            cause: ProcessExitCause::HostRequested,
            exit_code: 0,
            source: None,
            thread_id,
        };
        let terminated = !matches!(
            self.lifecycle,
            nixe_scheduler::ProcessLifecycle::Exited | nixe_scheduler::ProcessLifecycle::Faulted
        );
        if terminated {
            nixe_scheduler::transition_process(
                &mut self.lifecycle,
                nixe_scheduler::ProcessLifecycle::Terminating,
            )
            .expect("a live process can terminate");
            nixe_scheduler::transition_process(
                &mut self.lifecycle,
                nixe_scheduler::ProcessLifecycle::Exited,
            )
            .expect("a terminating process can exit");
            let thread_lifecycle = self.main_thread().lifecycle;
            if !matches!(
                thread_lifecycle,
                nixe_scheduler::ThreadLifecycle::Exited | nixe_scheduler::ThreadLifecycle::Faulted
            ) {
                nixe_scheduler::transition_thread(
                    &mut self.main_thread_mut().lifecycle,
                    nixe_scheduler::ThreadLifecycle::Terminating,
                )
                .expect("a live main thread can terminate");
                nixe_scheduler::transition_thread(
                    &mut self.main_thread_mut().lifecycle,
                    nixe_scheduler::ThreadLifecycle::Exited,
                )
                .expect("a terminating main thread can exit");
            }
            self.process_exit = Some(exit);
            self.main_thread_mut().exit = Some(ThreadExit {
                requested_scope: ExceptionTerminationScope::Process,
                exit_code: 0,
                source: None,
            });
        }
        terminated
    }

    /// Runs one bounded slice through the injected CPU engine domain.
    pub fn run(
        &mut self,
        instruction_budget: u64,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        self.run_thread(
            self.main_thread_id,
            nixe_scheduler::VirtualCpuId::new(0),
            instruction_budget,
        )
    }

    /// Runs one bounded slice for the thread and emulated vCPU selected by the
    /// scheduler. This is the production execution entry point.
    pub(crate) fn run_thread(
        &mut self,
        thread_id: nixe_scheduler::GuestThreadId,
        vcpu: nixe_scheduler::VirtualCpuId,
        instruction_budget: u64,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        let Some(selected) = self.threads.get(thread_id) else {
            return Err(ProcessExecutionError::UnknownThread(thread_id));
        };
        if selected.lifecycle != nixe_scheduler::ThreadLifecycle::Ready {
            return Err(ProcessExecutionError::NotRunnable {
                status: self.execution_status(),
                context: Box::new(selected.state.register_context()),
            });
        }
        let loader_return = selected.loader_return;
        let thread = self
            .threads
            .get_mut(thread_id)
            .expect("the selected thread was validated");
        nixe_scheduler::transition_thread(
            &mut thread.lifecycle,
            nixe_scheduler::ThreadLifecycle::Running,
        )
        .expect("a ready thread can run");
        let report_result = execution::run_engine(
            &mut self.execution,
            vcpu,
            self.cpu,
            &self.memory,
            &mut thread.state,
            instruction_budget,
            loader_return,
        );
        let report = match report_result {
            Ok(report) => report,
            Err(error) => {
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Faulted,
                )
                .expect("a running thread may fault");
                nixe_scheduler::transition_process(
                    &mut self.lifecycle,
                    nixe_scheduler::ProcessLifecycle::Faulted,
                )
                .expect("a running process may fault");
                return Err(error);
            }
        };
        let target = match report.stop {
            ExecutionStop::BudgetExhausted
            | ExecutionStop::Safepoint
            | ExecutionStop::PendingEvent { .. } => nixe_scheduler::ThreadLifecycle::Ready,
            ExecutionStop::FetchFault { .. } | ExecutionStop::UnsupportedSemantics { .. } => {
                nixe_scheduler::ThreadLifecycle::Faulted
            }
            ExecutionStop::LoaderReturn { .. } => nixe_scheduler::ThreadLifecycle::Exited,
            _ => nixe_scheduler::ThreadLifecycle::Waiting,
        };
        if target == nixe_scheduler::ThreadLifecycle::Exited {
            nixe_scheduler::transition_thread(
                &mut thread.lifecycle,
                nixe_scheduler::ThreadLifecycle::Terminating,
            )
            .expect("a running thread may terminate");
        }
        nixe_scheduler::transition_thread(&mut thread.lifecycle, target)
            .expect("engine exits define legal running-thread transitions");
        thread.wait_reason = (thread.lifecycle == nixe_scheduler::ThreadLifecycle::Waiting)
            .then_some(nixe_scheduler::WaitReason::Scheduler);
        if let ExecutionStop::LoaderReturn {
            source,
            result_code,
        } = &report.stop
        {
            let object_thread_id = thread.object.thread_id();
            let exit = ProcessExit {
                cause: ProcessExitCause::LoaderReturned,
                exit_code: *result_code,
                source: Some(*source),
                thread_id: object_thread_id,
            };
            nixe_scheduler::transition_process(
                &mut self.lifecycle,
                nixe_scheduler::ProcessLifecycle::Terminating,
            )
            .expect("a running process may terminate");
            nixe_scheduler::transition_process(
                &mut self.lifecycle,
                nixe_scheduler::ProcessLifecycle::Exited,
            )
            .expect("a terminating process may exit");
            self.process_exit = Some(exit);
            let thread = self
                .threads
                .get_mut(thread_id)
                .expect("the executed thread remains registered");
            thread.exit = Some(ThreadExit {
                requested_scope: ExceptionTerminationScope::Process,
                exit_code: *result_code,
                source: Some(*source),
            });
        }
        Ok(report)
    }

    /// Deprecated compatibility name for [`Self::run`]. New orchestration must
    /// use the engine-neutral method; this wrapper is removed after migration.
    pub fn run_reference(
        &mut self,
        instruction_budget: u64,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        self.run(instruction_budget)
    }

    /// Failure-atomically replaces the process execution domain at a canonical
    /// state boundary. The old domain remains installed if preparation fails.
    pub fn switch_engine(
        &mut self,
        provider: &dyn nixe_cpu_engine::EngineProvider,
    ) -> Result<nixe_cpu_engine::StateCommitBarrier, nixe_cpu_engine::HandoffFailure> {
        self.execution.switch_provider(self.cpu, provider)
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
        self.route_supervisor_call_for(
            self.main_thread_id,
            nixe_scheduler::VirtualCpuId::new(0),
            stop,
            dispatcher,
        )
    }

    /// Routes an exception to the explicit thread and vCPU from a completed
    /// scheduler lease.
    pub fn route_supervisor_call_for<D: ExceptionDispatcher>(
        &mut self,
        thread_id: nixe_scheduler::GuestThreadId,
        vcpu: nixe_scheduler::VirtualCpuId,
        stop: &ExecutionStop,
        dispatcher: &mut D,
    ) -> Result<ExceptionHandlingResult<D::Fault>, ExceptionRouteError> {
        let request = stop
            .exception_dispatch_request()
            .filter(|request| request.kind() == nixe_cpu::exception::ExceptionKind::SupervisorCall)
            .ok_or(ExceptionRouteError::NotSupervisorCall)?;
        let selected = self
            .threads
            .get(thread_id)
            .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
        if selected.lifecycle != nixe_scheduler::ThreadLifecycle::Waiting {
            return Err(ExceptionRouteError::ProcessNotSuspended {
                status: self.execution_status(),
            });
        }
        let current = execution::current_location(self.cpu, &selected.state);
        if request.source() != current {
            return Err(ExceptionRouteError::SourceMismatch {
                requested: request.source(),
                current,
            });
        }
        let thread = self
            .threads
            .get_mut(thread_id)
            .expect("the selected thread was validated");
        let handle = thread.handle;
        let object = thread.object.clone();
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
            &self.memory,
            &self.mounts,
            &mut self.handles,
            &mut self.address_waits,
        );
        let thread =
            ExceptionThreadContext::new(thread_id, vcpu, object, handle, &mut thread.state);
        let mut context = ExceptionDispatchContext::new(process, thread);
        let outcome = dispatcher.dispatch(&mut context, request);
        self.apply_supervisor_call_outcome(thread_id, request.source(), outcome)
    }

    fn apply_supervisor_call_outcome<F>(
        &mut self,
        thread_id: nixe_scheduler::GuestThreadId,
        source: LocationDescriptor,
        outcome: ExceptionDispatchOutcome<F>,
    ) -> Result<ExceptionHandlingResult<F>, ExceptionRouteError> {
        match outcome {
            ExceptionDispatchOutcome::Resume(continuation) => {
                let target = supervisor_call_continuation(source, continuation)?;
                let cpu = self.cpu;
                install_continuation(
                    cpu,
                    &mut self
                        .thread_mut(thread_id)
                        .ok_or(ExceptionRouteError::UnknownThread(thread_id))?
                        .state,
                    target,
                )?;
                let thread = self
                    .thread_mut(thread_id)
                    .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Ready,
                )
                .expect("a waiting exception thread may resume");
                thread.wait_reason = None;
                Ok(ExceptionHandlingResult::Resumed)
            }
            ExceptionDispatchOutcome::Suspend(continuation) => {
                let target = supervisor_call_continuation(source, continuation)?;
                let cpu = self.cpu;
                let thread = self
                    .thread_mut(thread_id)
                    .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
                install_continuation(cpu, &mut thread.state, target)?;
                thread.continuation = Some(match continuation {
                    ExceptionResume::Retry => nixe_scheduler::Continuation::Retry,
                    ExceptionResume::Next => nixe_scheduler::Continuation::Next,
                    ExceptionResume::At(target) => {
                        nixe_scheduler::Continuation::Address(target.pc.get())
                    }
                });
                Ok(ExceptionHandlingResult::Suspended)
            }
            ExceptionDispatchOutcome::Reject { diagnostic } => {
                let target = supervisor_call_continuation(source, ExceptionResume::Next)?;
                let cpu = self.cpu;
                install_continuation(
                    cpu,
                    &mut self
                        .thread_mut(thread_id)
                        .ok_or(ExceptionRouteError::UnknownThread(thread_id))?
                        .state,
                    target,
                )?;
                let thread = self
                    .thread_mut(thread_id)
                    .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Ready,
                )
                .expect("a rejected call resumes its waiting thread");
                thread.wait_reason = None;
                Ok(ExceptionHandlingResult::Rejected(diagnostic))
            }
            ExceptionDispatchOutcome::Terminate {
                scope,
                exit_code,
                reason,
            } => {
                let cpu = self.cpu;
                install_continuation(
                    cpu,
                    &mut self
                        .thread_mut(thread_id)
                        .ok_or(ExceptionRouteError::UnknownThread(thread_id))?
                        .state,
                    source,
                )?;
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
                    thread_id: thread_id.get(),
                };
                let terminate_process = scope == ExceptionTerminationScope::Process
                    || !self.threads.iter().any(|(id, thread)| {
                        *id != thread_id
                            && !matches!(
                                thread.lifecycle,
                                nixe_scheduler::ThreadLifecycle::Exited
                                    | nixe_scheduler::ThreadLifecycle::Faulted
                            )
                    });
                if terminate_process {
                    nixe_scheduler::transition_process(
                        &mut self.lifecycle,
                        nixe_scheduler::ProcessLifecycle::Terminating,
                    )
                    .expect("a live process may terminate");
                    nixe_scheduler::transition_process(
                        &mut self.lifecycle,
                        nixe_scheduler::ProcessLifecycle::Exited,
                    )
                    .expect("a terminating process may exit");
                    self.process_exit = Some(exit);
                }
                let thread = self
                    .thread_mut(thread_id)
                    .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Terminating,
                )
                .expect("a waiting exception thread may terminate");
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Exited,
                )
                .expect("a terminating exception thread may exit");
                thread.exit = Some(ThreadExit {
                    requested_scope: scope,
                    exit_code,
                    source: Some(source),
                });
                thread.object.signal();
                Ok(ExceptionHandlingResult::Terminated {
                    scope,
                    exit_code,
                    reason,
                })
            }
            ExceptionDispatchOutcome::Fault(fault) => {
                let cpu = self.cpu;
                install_continuation(
                    cpu,
                    &mut self
                        .thread_mut(thread_id)
                        .ok_or(ExceptionRouteError::UnknownThread(thread_id))?
                        .state,
                    source,
                )?;
                let has_other_live = self.threads.iter().any(|(id, thread)| {
                    *id != thread_id
                        && !matches!(
                            thread.lifecycle,
                            nixe_scheduler::ThreadLifecycle::Exited
                                | nixe_scheduler::ThreadLifecycle::Faulted
                        )
                });
                if !has_other_live {
                    nixe_scheduler::transition_process(
                        &mut self.lifecycle,
                        nixe_scheduler::ProcessLifecycle::Faulted,
                    )
                    .expect("a live process may fault");
                }
                let thread = self
                    .thread_mut(thread_id)
                    .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Faulted,
                )
                .expect("a waiting exception thread may fault");
                thread.object.signal();
                Ok(ExceptionHandlingResult::Fault(fault))
            }
        }
    }

    /// Consumes the process and deterministically releases all process-owned resources.
    #[must_use]
    pub fn teardown(self) -> ProcessTeardownReport {
        let previous_status = self.execution_status();
        ProcessTeardownReport {
            previous_status,
            exit: self.process_exit,
            threads_released: self.threads.len(),
            modules_released: self.modules.len(),
            mappings_released: self
                .modules
                .iter()
                .map(|module| module.mappings().len())
                .sum(),
            physical_pages_released: self.memory.physical_page_count(),
            mounts_released: self.mounts.mount_count(),
            handles_released: self.handles.len(),
            address_waiters_released: self.address_waits.waiter_count(),
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
            self.main_thread().state.execution_state(),
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
            source: execution::current_location(cpu, state),
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
    EngineInitialization,
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

#[cfg(test)]
pub(crate) mod tests;
