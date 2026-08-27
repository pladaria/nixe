//! Construction of a runnable CPU process from an immutable launch plan.

mod builder;
mod dispatch;
mod exception;
pub(crate) mod execution;
mod layout;
mod thread;

pub use builder::ProcessBuilder;
#[cfg(test)]
use builder::a64_register;
use builder::{ThreadPolicy, align_up, error, initialize_created_thread};
pub use execution::{
    CpuBackendConfig, ExecutionReport, ExecutionStop, ProcessExecutionError, ProcessExit,
    ProcessExitCause, ProcessTeardownFailure, ProcessTeardownReport, ThreadExit,
};
pub use layout::{
    ProcessAddressSpace, ProcessBuildConfig, ProcessMemoryLayout, ProcessMemoryLayoutProfile,
    ProcessVirtualRegion,
};
pub use thread::{
    GuestThread, ThreadCreateError, ThreadCreateRequest, ThreadCreation, ThreadTable,
    ThreadTableError,
};

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use nixe_cpu::location::LocationDescriptor;
use nixe_cpu::memory::{
    CpuMemory, ExecutionMemory, MappingEpoch, MemoryAttributes, MemoryMappingError,
    MemoryMappingPurpose, MemoryPermissions, MemoryProtectionError, ProcessMemory,
    SYNTHETIC_PAGE_SIZE, SyntheticRamPage,
};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::{ThreadCpuState, a64::A64Register};
use nixe_loader_executable::{
    AddressSpaceType, ExternalSymbol, PreparationConfig, PreparedModule, SymbolResolution,
};
use nixe_memory::MemoryInvalidationSource;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

use crate::exception_dispatch::{ExceptionProcessMetadata, ExceptionProcessResources};
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
const HOME_BREW_ARGV_KEY: u32 = 5;
const HOME_BREW_ARGV_OFFSET: usize = 0x100;
const _: () = assert!(HOME_BREW_CONFIG_ENTRY_SIZE * 3 <= HOME_BREW_ARGV_OFFSET);
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
    memory: std::sync::Arc<ExecutionMemory>,
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
    pub fn memory(&self) -> &ExecutionMemory {
        self.memory.as_ref()
    }

    /// Applies one runtime mapping resize after closing new execution leases,
    /// then publishes the committed epoch to every CPU-thread control path.
    pub fn resize_memory_mapping(
        &self,
        start: GuestVirtualAddress,
        old_size: u64,
        new_size: u64,
        permissions: MemoryPermissions,
        purpose: MemoryMappingPurpose,
    ) -> Result<(), MemoryMappingError> {
        self.execution.request_mapping_safepoint();
        self.memory.resize_zeroed_mapping(
            self.cpu.address_space_id(),
            start,
            old_size,
            new_size,
            permissions,
            purpose,
        )?;
        self.execution
            .publish_memory_invalidation(self.memory.invalidation_cursor());
        Ok(())
    }

    pub fn set_memory_permissions(
        &self,
        start: GuestVirtualAddress,
        size: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryProtectionError> {
        self.execution.request_mapping_safepoint();
        self.memory
            .set_permissions(self.cpu.address_space_id(), start, size, permissions)?;
        self.execution
            .publish_memory_invalidation(self.memory.invalidation_cursor());
        Ok(())
    }

    pub fn set_memory_attributes(
        &self,
        start: GuestVirtualAddress,
        size: u64,
        mask: MemoryAttributes,
        value: MemoryAttributes,
    ) -> Result<(), MemoryProtectionError> {
        self.execution.request_mapping_safepoint();
        self.memory
            .set_attributes(self.cpu.address_space_id(), start, size, mask, value)?;
        self.execution
            .publish_memory_invalidation(self.memory.invalidation_cursor());
        Ok(())
    }

    #[must_use]
    pub fn mapping_epoch(&self) -> MappingEpoch {
        self.memory.mapping_epoch()
    }

    #[must_use]
    pub fn memory_invalidation_acknowledged(
        &self,
        cursor: nixe_memory::MemoryInvalidationCursor,
    ) -> bool {
        self.execution.memory_invalidation_acknowledged(cursor)
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
    pub fn main_thread(&self) -> &crate::GuestThread {
        self.threads
            .get(self.main_thread_id)
            .expect("a runnable process always retains its main thread")
    }

    /// Returns mutable main-thread state for runtime scheduling and ABI setup.
    pub fn main_thread_mut(&mut self) -> &mut crate::GuestThread {
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
        if request.entry.get() >= self.address_space.exclusive_limit()
            || !request.entry.get().is_multiple_of(4)
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
        let Some(stack_byte) = request.stack_top.get().checked_sub(1) else {
            return Err(ThreadCreateError::InvalidStack);
        };
        if request.stack_top.get() > self.address_space.exclusive_limit()
            || !request.stack_top.get().is_multiple_of(16)
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
        self.resize_memory_mapping(
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
        let mut state = ThreadCpuState::default();
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
            state: Some(state),
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
        let _ = self.resize_memory_mapping(
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

    /// Consumes the process and deterministically releases all process-owned resources.
    pub fn try_teardown(mut self) -> Result<ProcessTeardownReport, ProcessTeardownFailure> {
        let previous_lifecycle = self.lifecycle;
        let report = ProcessTeardownReport {
            previous_lifecycle,
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
        };
        self.execution
            .shutdown()
            .map_err(|fault| ProcessTeardownFailure {
                report: Box::new(report),
                fault: Box::new(fault),
            })?;
        Ok(report)
    }

    pub(crate) fn request_execution_safepoint(&self) {
        self.execution.request_safepoint();
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
            let pc = source
                .pc
                .checked_add(4)
                .ok_or(ExceptionRouteError::ContinuationAddressOverflow { source })?;
            Ok(LocationDescriptor::new(pc, source.profile_id))
        }
    }
}

fn install_continuation(
    cpu: ProcessCpuContext,
    state: &mut ThreadCpuState,
    target: LocationDescriptor,
) -> Result<(), ExceptionRouteError> {
    let expected_profile = cpu.profile_id();
    if target.profile_id != expected_profile {
        return Err(ExceptionRouteError::ContinuationProfileMismatch {
            source: execution::current_location(cpu, state),
            target,
        });
    }
    if !target.is_aligned() {
        return Err(ExceptionRouteError::InvalidContinuationTarget { target });
    }
    state.set_pc(target.pc.get());
    Ok(())
}

/// Stage at which process construction failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessBuildStage {
    Metadata,
    CpuInitialization,
    Placement,
    Preparation,
    Mapping,
    ThreadInitialization,
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
