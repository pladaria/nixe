//! Construction of a runnable CPU process from an immutable launch plan.

mod builder;
mod dispatch;
mod exception;
pub(crate) mod execution;
mod layout;
mod memory_accounting;
mod thread;
mod thread_local;

pub use builder::ProcessBuilder;
pub(crate) use builder::ThreadPolicy;
#[cfg(test)]
use builder::a64_register;
use builder::{align_up, error, initialize_created_thread};
pub use execution::{
    ExecutionReport, ExecutionStop, GuestStackFrame, ProcessExecutionError, ProcessExit,
    ProcessExitCause, ProcessTeardownFailure, ProcessTeardownReport, ThreadExit,
};
pub use layout::{
    ProcessAddressSpace, ProcessBuildConfig, ProcessMemoryLayout, ProcessMemoryLayoutProfile,
    ProcessVirtualRegion,
};
pub use memory_accounting::ProcessMemoryAccounting;
pub use thread::{
    ThreadCreateError, ThreadCreateRequest, ThreadCreation, ThreadObject, ThreadRegisters,
    ThreadStartError, ThreadTable, ThreadTableError, ThreadWait,
};

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::GuestCpuState;
use crate::memory::ProcessMemory;
use nixe_cpu::location::LocationDescriptor;
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::A64Register;
use nixe_loader_executable::{
    AddressSpaceType, ExternalSymbol, PreparationConfig, PreparedModule, SymbolResolution,
};
use nixe_memory::GuestVirtualAddress;
use nixe_memory::{MemoryPermissions, PAGE_SIZE};

use crate::{
    ExceptionDispatchContext, ExceptionDispatchOutcome, ExceptionDispatcher,
    ExceptionHandlingResult, ExceptionResume, ExceptionRouteError, ExceptionTerminationReason,
    ExceptionTerminationScope, ExceptionThreadContext, LaunchKind, LaunchModuleImage, LaunchPlan,
};

const DEFAULT_IMAGE_BASE: u64 = 0x7100_0000;
const DEFAULT_HOME_BREW_STACK_SIZE: u64 = 0x10_0000;
const MODULE_GUARD_SIZE: u64 = 0x1_0000;
const RESOURCE_GUARD_SIZE: u64 = 0x1_0000;
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
    state: Mutex<ProcessState>,
    generation: std::sync::atomic::AtomicU64,
    cpu: ProcessCpuContext,
    address_space: ProcessAddressSpace,
    memory_layout: ProcessMemoryLayout,
    random_entropy: [u64; 4],
    memory: std::sync::Arc<ProcessMemory>,
    modules: Box<[PreparedModule]>,
    entry_module: usize,
    main_thread_id: nixe_scheduler::GuestThreadId,
    initial_thread_priority: i32,
    initial_ideal_vcpu: nixe_scheduler::VirtualCpuId,
    thread_policy: Option<ThreadPolicy>,
    thread_local: Arc<Mutex<thread_local::ThreadLocalPages>>,
    process_local_region: GuestVirtualAddress,
    threads: crate::ThreadTable,
    mounts: std::sync::Arc<crate::ProcessMountNamespace>,
    handles: std::sync::Arc<crate::HandleTable>,
    address_waits: std::sync::Arc<crate::AddressWaitRegistry>,
    pub(crate) scheduler: std::sync::Arc<nixe_scheduler::SchedulerState>,
    execution: Mutex<execution::ProcessExecutionControl>,
    pub(crate) events: std::sync::OnceLock<crate::ExternalEventSender>,
}

impl std::fmt::Debug for RunnableProcess {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Process")
            .field("id", &self.process_id)
            .finish_non_exhaustive()
    }
}

struct ProcessState {
    lifecycle: nixe_scheduler::ProcessLifecycle,
    exit: Option<ProcessExit>,
    num_running_threads: usize,
    termination: Option<nixe_scheduler::ProcessLifecycle>,
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
    pub fn lifecycle(&self) -> nixe_scheduler::ProcessLifecycle {
        self.state.lock().unwrap().lifecycle
    }

    #[must_use]
    pub const fn memory_layout(&self) -> ProcessMemoryLayout {
        self.memory_layout
    }

    /// Kernel-owned exception context, retained independently of thread TLS.
    #[must_use]
    pub const fn user_exception_context_address(&self) -> GuestVirtualAddress {
        self.process_local_region
    }

    /// Returns the currently committed process heap size.
    #[must_use]
    pub fn heap_size(&self) -> u64 {
        self.memory.heap_size()
    }

    /// Returns the coherent physical-memory counters used by Horizon queries.
    #[must_use]
    pub fn memory_accounting(&self) -> ProcessMemoryAccounting {
        self.memory.accounting()
    }

    #[must_use]
    pub fn memory(&self) -> &std::sync::Arc<ProcessMemory> {
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
    pub fn main_thread(&self) -> Arc<crate::ThreadObject> {
        self.threads
            .get(self.main_thread_id)
            .expect("main thread remains retained")
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

    pub async fn create_thread(
        self: &Arc<Self>,
        request: ThreadCreateRequest,
    ) -> Result<ThreadCreation, ThreadCreateError> {
        if self.lifecycle() != nixe_scheduler::ProcessLifecycle::Running {
            return Err(ThreadCreateError::Terminating);
        }
        let scheduler = self.scheduler();
        if let Some(vcpu) = request.affinity.iter().find(|core| {
            !scheduler.profile().contains(*core)
                || self.thread_policy.is_some_and(|policy| {
                    core.get() < policy.min_core || core.get() > policy.max_core
                })
        }) {
            return Err(ThreadCreateError::InvalidVirtualCpu(vcpu));
        }
        if let Some(ideal) = request.ideal_vcpu
            && !request.affinity.contains(ideal)
        {
            return Err(ThreadCreateError::InvalidVirtualCpu(ideal));
        }
        if !scheduler.profile().priorities().contains(request.priority)
            || self.thread_policy.is_some_and(|policy| {
                request.priority < policy.highest_priority
                    || request.priority > policy.lowest_priority
            })
        {
            return Err(ThreadCreateError::InvalidPriority(request.priority));
        }
        let allocation =
            thread::ThreadLocalRegion::allocate(self.thread_local.clone(), self.memory.clone())
                .map_err(|error| {
                    if self.lifecycle() != nixe_scheduler::ProcessLifecycle::Running {
                        ThreadCreateError::Terminating
                    } else {
                        match error {
                            crate::memory::MappingError::Memory(fault) => match fault.as_ref() {
                                nixe_memory::MemoryError::ProcessLimit { .. }
                                | nixe_memory::MemoryError::PhysicalExhaustion { .. }
                                | nixe_memory::MemoryError::RetainedAllocations { .. } => {
                                    ThreadCreateError::OutOfMemory
                                }
                                _ => ThreadCreateError::Memory(fault),
                            },
                            crate::memory::MappingError::InvalidRange => {
                                ThreadCreateError::OutOfMemory
                            }
                            _ => ThreadCreateError::Internal,
                        }
                    }
                })?;
        let tls_base = allocation.address;
        let mut state = GuestCpuState::default();
        initialize_created_thread(&mut state, &request, tls_base);
        // This object is private until initialization and scheduler registration
        // finish; reserved handles cannot resolve the provisional identity.
        let object = Arc::new(crate::ThreadObject::new(
            &self.scheduler,
            nixe_scheduler::GuestThreadId::new(0),
            state,
            request.stack_top,
            request.stack_top,
            None,
            allocation,
        ));
        object.bind_owner(self);
        let reservation = self
            .handles
            .reserve(Some(crate::HandleObject::from_arc(object.clone())))
            .map_err(|error| match error {
                crate::HandleError::Closed => ThreadCreateError::Terminating,
                crate::HandleError::Exhausted => ThreadCreateError::OutOfHandles,
                _ => ThreadCreateError::Internal,
            })?;
        let handle = reservation.handle();
        // Zero the reserved region and install its self handle in one coherent
        // write, without holding the TLS allocation lock across device access.
        let access = {
            let mut bytes = [0; thread_local::REGION_SIZE as usize];
            bytes[0x110..0x114].copy_from_slice(&handle.to_le_bytes());
            self.memory
                .space()
                .write(tls_base, &bytes)
                .map_err(|fault| ThreadCreateError::Memory(Arc::new(fault)))?
        };
        if let nixe_memory::Access::Pending(pending) = access {
            match pending.await.map_err(ThreadCreateError::Memory)? {
                nixe_memory::AccessResult::Written => {}
                _ => return Err(ThreadCreateError::Internal),
            }
        }
        let id = scheduler
            .create_thread(nixe_scheduler::ScheduledThreadConfig {
                process: nixe_scheduler::ProcessId::new(self.process_id),
                base_priority: request.priority,
                ideal_vcpu: request.ideal_vcpu,
                affinity: request.affinity,
            })
            .map_err(|fault| match fault {
                nixe_scheduler::SchedulerError::ThreadIdExhausted => {
                    ThreadCreateError::IdentityExhausted
                }
                _ => ThreadCreateError::Internal,
            })?;
        object.assign_thread_id(id.get());
        let result = {
            let process = self.state.lock().unwrap();
            if process.lifecycle != nixe_scheduler::ProcessLifecycle::Running {
                Err(ThreadCreateError::Terminating)
            } else if self.threads.insert(&object, false).is_err() {
                Err(ThreadCreateError::Internal)
            } else if self.handles.publish(&mut [reservation]).is_err() {
                self.threads
                    .remove(id)
                    .expect("unpublished thread membership");
                Err(ThreadCreateError::Terminating)
            } else {
                object.bind_retirement(
                    nixe_scheduler::ProcessId::new(self.process_id),
                    self.events.get().expect("registered process").clone(),
                );
                Ok(ThreadCreation { id, handle })
            }
        };
        if result.is_err() {
            scheduler
                .unregister(id)
                .expect("unpublished thread has no register owner");
        }
        result
    }

    pub(crate) fn retire_thread(
        &self,
        id: nixe_scheduler::GuestThreadId,
    ) -> Result<(), ThreadCreateError> {
        self.threads
            .remove(id)
            .map_err(|_| ThreadCreateError::Internal)
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
            self.main_thread().assign_thread_id(id.get());
            return Ok(());
        }
        self.threads.rekey(old, id)?;
        self.main_thread_id = id;
        self.main_thread().assign_thread_id(id.get());
        Ok(())
    }

    #[must_use]
    pub const fn threads(&self) -> &crate::ThreadTable {
        &self.threads
    }

    #[must_use]
    pub fn thread(&self, id: nixe_scheduler::GuestThreadId) -> Option<Arc<crate::ThreadObject>> {
        self.threads.get(id)
    }

    pub(crate) fn finish_thread(&self, thread: &Arc<crate::ThreadObject>) -> bool {
        let id = thread.id();
        {
            let mut process = self.state.lock().unwrap();
            if self.threads.finish(id) {
                process.num_running_threads -= 1;
                let last = process.num_running_threads == 0;
                if last && process.lifecycle == nixe_scheduler::ProcessLifecycle::Running {
                    process.lifecycle = nixe_scheduler::ProcessLifecycle::Terminating;
                    process.termination = Some(nixe_scheduler::ProcessLifecycle::Exited);
                    self.generation
                        .fetch_add(1, std::sync::atomic::Ordering::Release);
                }
                last
            } else {
                false
            }
        }
    }

    pub(crate) fn terminate_thread_from_coordinator(
        &self,
        id: nixe_scheduler::GuestThreadId,
        exit_code: u64,
        requested_scope: ExceptionTerminationScope,
    ) -> Option<Arc<crate::ThreadObject>> {
        let thread = self.thread(id)?;
        thread.set_exit(ThreadExit {
            requested_scope,
            exit_code,
            source: None,
        });
        self.finish_thread(&thread);
        Some(thread)
    }

    pub fn scheduler(&self) -> &Arc<nixe_scheduler::SchedulerState> {
        &self.scheduler
    }
    pub fn allowed_core_mask(&self) -> Option<u64> {
        self.thread_policy?.allowed_core_mask()
    }
    pub fn allowed_priority_mask(&self) -> Option<u64> {
        self.thread_policy?.allowed_priority_mask()
    }
    pub fn address_space_limit(&self) -> u64 {
        self.address_space.exclusive_limit()
    }
    pub fn cpu(&self) -> ProcessCpuContext {
        self.cpu
    }
    pub fn random_entropy(&self, index: usize) -> Option<u64> {
        self.random_entropy.get(index).copied()
    }
    pub fn total_user_physical_memory_size(&self) -> u64 {
        self.memory.accounting().total_user_physical_memory_size()
    }
    pub fn used_user_physical_memory_size(&self) -> u64 {
        self.memory.accounting().used_user_physical_memory_size()
    }
    pub fn total_non_system_user_physical_memory_size(&self) -> u64 {
        self.memory
            .accounting()
            .total_non_system_user_physical_memory_size()
    }
    pub fn used_non_system_user_physical_memory_size(&self) -> u64 {
        self.memory
            .accounting()
            .used_non_system_user_physical_memory_size()
    }
    pub fn total_system_resource_size(&self) -> u64 {
        self.memory.accounting().total_system_resource_size()
    }
    pub fn used_system_resource_size(&self) -> u64 {
        self.memory.accounting().used_system_resource_size()
    }
    pub fn generation(&self) -> u64 {
        self.generation.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Returns the immutable process-local filesystem namespace.
    #[must_use]
    pub const fn mounts(&self) -> &std::sync::Arc<crate::ProcessMountNamespace> {
        &self.mounts
    }

    /// Returns the process-local kernel-object handle table.
    #[must_use]
    pub const fn handles(&self) -> &std::sync::Arc<crate::HandleTable> {
        &self.handles
    }

    /// The accepted continuation owns its mailbox; ingress only wakes the execution owner.
    pub fn notify_kernel_completion(&self) {
        self.events
            .get()
            .expect("executing process is registered")
            .notify_host_service(crate::ExternalEventSource::Device);
    }

    #[must_use]
    pub const fn address_waits(&self) -> &std::sync::Arc<crate::AddressWaitRegistry> {
        &self.address_waits
    }

    /// Closes process resources after native contexts have quiesced. Retained
    /// kernel objects continue to own their identity and pinned physical inputs.
    pub(crate) fn try_teardown(&self) -> Result<ProcessTeardownReport, ProcessTeardownFailure> {
        let previous_lifecycle = self.lifecycle();
        let report = ProcessTeardownReport {
            previous_lifecycle,
            exit: self.state.lock().unwrap().exit.clone(),
            threads_released: self.threads.len(),
            modules_released: self.modules.len(),
            mappings_released: self
                .modules
                .iter()
                .map(|module| module.mappings().len())
                .sum(),
            physical_pages_released: (self.memory.account().used_bytes() / PAGE_SIZE) as usize,
            mounts_released: self.mounts.mount_count(),
            handles_released: self.handles.len(),
            address_waiters_released: self.address_waits.lock().waiter_count(),
        };
        self.cancel_waits();
        let execution = self.execution.lock().unwrap().shutdown();
        self.thread_local.lock().unwrap().closed = true;
        let memory = self.memory.close();
        self.handles.close_all();
        self.threads.release_running();
        self.state.lock().unwrap().num_running_threads = 0;
        if let Err(fault) = execution {
            return Err(ProcessTeardownFailure {
                report: Box::new(report),
                fault: Box::new(fault),
            });
        }
        memory.map_err(|error| ProcessTeardownFailure {
            report: Box::new(report.clone()),
            fault: Box::new(nixe_cpu::execution::CpuFault {
                backend: "memory",
                kind: nixe_cpu::execution::CpuFaultKind::Internal,
                message: error.to_string().into(),
                context: None,
            }),
        })?;
        Ok(report)
    }

    pub(crate) fn request_vcpu_safepoint(&self, vcpu: nixe_scheduler::VirtualCpuId) {
        self.execution.lock().unwrap().request_vcpu_safepoint(vcpu);
    }

    pub(crate) fn request_execution_safepoint(&self) {
        self.execution.lock().unwrap().request_safepoint();
    }
}

impl Drop for RunnableProcess {
    fn drop(&mut self) {
        let _ = self.execution.get_mut().unwrap().request_stop();
        self.thread_local.lock().unwrap().closed = true;
        let _ = self.memory.close();
        self.handles.close_all();
    }
}

fn supervisor_call_pc(source: LocationDescriptor, continuation: ExceptionResume) -> u64 {
    // Native entry supplies the immutable profile and an executed A64 SVC
    // boundary; no arbitrary target/profile is accepted by this continuation.
    // https://github.com/eden-emulator/mirror/blob/master/src/core/hle/kernel/physical_core.cpp
    match continuation {
        ExceptionResume::Retry => source.pc.get(),
        ExceptionResume::Next => source.pc.get().wrapping_add(4),
    }
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
