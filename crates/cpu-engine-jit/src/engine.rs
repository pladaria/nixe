//! Engine provider, process domain, and executor-local native state.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings;
use nixe_cpu::location::{ExecutionState, LocationDescriptor};
use nixe_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu_engine::{
    CapabilityRejection, CapabilityRejectionReason, CapabilityReport, CrossVcpuRequest,
    DomainMemoryBinding, DomainRequest, EngineCapabilities, EngineControl, EngineDescriptor,
    EngineDomain, EngineDomainId, EngineExecutor, EngineExecutorId, EngineFault, EngineFaultKind,
    EngineId, EngineKind, EngineProvider, ExecutionReport, RunRequest,
};
use nixe_memory::GuestVirtualAddress;

use crate::abi::{
    EXIT_BUDGET_EXHAUSTED, EXIT_INTERPRET_ONE, EXIT_LOADER_RETURN, EXIT_NONE, EXIT_PENDING_EVENT,
    EXIT_SAFEPOINT, ExecutionFrame, FrameError, NativeExit,
};
use crate::compiler::CompilerContext;
use crate::executable_memory::{
    ExecutableMemoryError, SharedExecutableMemory, process_executable_memory,
};

pub const JIT_ENGINE_ID: EngineId = EngineId::new(2);

const CONTROL_PREEMPT: u32 = 1 << 0;
const CONTROL_CODE_INVALIDATION: u32 = 1 << 1;

enum HostSupport {
    Available(OwnedTargetIsa),
    Unavailable {
        reason: CapabilityRejectionReason,
        detail: Box<str>,
    },
}

/// Cranelift JIT provider and owner of the process-wide executable arena.
pub struct JitProvider {
    host: HostSupport,
    executable_memory: Result<SharedExecutableMemory, ExecutableMemoryError>,
}

impl JitProvider {
    #[must_use]
    pub fn new() -> Self {
        Self {
            host: probe_host(),
            executable_memory: process_executable_memory(),
        }
    }

    #[cfg(test)]
    fn with_executable_error(error: ExecutableMemoryError) -> Self {
        Self {
            host: probe_host(),
            executable_memory: Err(error),
        }
    }

    fn availability_rejections(
        &self,
        profile: GuestCpuProfile,
        required: EngineCapabilities,
    ) -> Vec<CapabilityRejection> {
        let capabilities = capabilities();
        let mut rejections = Vec::new();
        if profile.id() != GuestCpuProfile::SWITCH_1_ID
            || !capabilities.supports_profile(profile, required)
        {
            rejections.push(rejection(
                CapabilityRejectionReason::GuestProfileUnsupported,
                "JIT supports only the verified Switch 1 CPU profile",
            ));
        }
        if !capabilities.contains(required) {
            rejections.push(rejection(
                CapabilityRejectionReason::MissingCapabilities,
                "required capability set is unavailable",
            ));
        }
        if let Err(error) = &self.executable_memory {
            rejections.push(CapabilityRejection {
                engine: JIT_ENGINE_ID,
                reason: error.rejection_reason(),
                detail: error.detail().into(),
            });
        }
        if let HostSupport::Unavailable { reason, detail } = &self.host {
            rejections.push(CapabilityRejection {
                engine: JIT_ENGINE_ID,
                reason: *reason,
                detail: detail.clone(),
            });
        }
        rejections
    }

    fn available_resources(
        &self,
        cpu: ProcessCpuContext,
    ) -> Result<(OwnedTargetIsa, SharedExecutableMemory), EngineFault> {
        let rejections = self.availability_rejections(cpu.profile(), EngineCapabilities::default());
        if !rejections.is_empty() {
            return Err(fault(
                EngineFaultKind::Unavailable,
                0,
                format!(
                    "JIT domain creation rejected: {}",
                    rejections
                        .iter()
                        .map(|rejection| rejection.detail.as_ref())
                        .collect::<Vec<_>>()
                        .join("; ")
                ),
                &ThreadCpuState::new(
                    cpu.thread_configuration(first_execution_state(cpu.profile()))
                        .expect("the selected state belongs to the profile"),
                ),
            ));
        }
        let isa = match &self.host {
            HostSupport::Available(isa) => Arc::clone(isa),
            HostSupport::Unavailable { .. } => unreachable!("rejections handled host failure"),
        };
        let executable_memory = self
            .executable_memory
            .as_ref()
            .expect("rejections handled executable-memory failure")
            .clone();
        Ok((isa, executable_memory))
    }
}

impl Default for JitProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineProvider for JitProvider {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn probe(&self, profile: GuestCpuProfile, required: EngineCapabilities) -> CapabilityReport {
        let rejections = self.availability_rejections(profile, required);
        CapabilityReport {
            descriptor: descriptor(),
            available: rejections.is_empty(),
            rejections: rejections.into_boxed_slice(),
        }
    }

    fn create_domain(&self, request: DomainRequest) -> Result<Box<dyn EngineDomain>, EngineFault> {
        let (isa, executable_memory) = self.available_resources(request.cpu)?;
        Ok(Box::new(JitDomain {
            id: request.domain,
            cpu: request.cpu,
            isa,
            executable_memory: Some(executable_memory),
            binding: None,
            shutdown: false,
        }))
    }
}

#[derive(Clone, Copy)]
struct BoundMemory {
    end_exclusive: GuestVirtualAddress,
    invalidation_epoch: u64,
}

struct JitDomain {
    id: EngineDomainId,
    cpu: ProcessCpuContext,
    isa: OwnedTargetIsa,
    // The provider owns the process-wide arena; each live domain retains the
    // owner so code can never outlive its OS mapping.
    executable_memory: Option<SharedExecutableMemory>,
    binding: Option<BoundMemory>,
    shutdown: bool,
}

impl EngineDomain for JitDomain {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn domain_id(&self) -> EngineDomainId {
        self.id
    }

    fn create_executor(
        &mut self,
        executor: EngineExecutorId,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
        if self.shutdown {
            return Err(domain_fault(
                self.cpu,
                EngineFaultKind::Unavailable,
                "JIT domain is shut down",
            ));
        }
        if self.executable_memory.is_none() {
            return Err(domain_fault(
                self.cpu,
                EngineFaultKind::Unavailable,
                "JIT executable-memory owner is unavailable",
            ));
        }
        let Some(binding) = self.binding else {
            return Err(domain_fault(
                self.cpu,
                EngineFaultKind::InvalidRequest,
                "canonical memory must be bound before creating a JIT executor",
            ));
        };
        Ok(Box::new(JitExecutor {
            id: executor,
            cpu: self.cpu,
            address_space_end: binding.end_exclusive,
            frame: ExecutionFrame::default(),
            _compiler: CompilerContext::new(Arc::clone(&self.isa)),
            _executable_memory: self
                .executable_memory
                .as_ref()
                .expect("live domain retains executable memory")
                .clone(),
            control: EngineControl::default(),
            acknowledged_epoch: binding.invalidation_epoch,
        }))
    }

    fn bind_memory(&mut self, binding: DomainMemoryBinding<'_>) -> Result<(), EngineFault> {
        if self.shutdown {
            return Err(domain_fault(
                self.cpu,
                EngineFaultKind::Unavailable,
                "cannot bind memory after JIT domain shutdown",
            ));
        }
        if binding.address_space != self.cpu.address_space_id() {
            return Err(domain_fault(
                self.cpu,
                EngineFaultKind::InvalidRequest,
                "canonical memory binding belongs to a different address space",
            ));
        }
        self.binding = Some(BoundMemory {
            end_exclusive: binding.end_exclusive,
            invalidation_epoch: binding.invalidation_generation,
        });
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), EngineFault> {
        self.shutdown = true;
        self.binding = None;
        self.executable_memory = None;
        Ok(())
    }
}

struct JitExecutor {
    id: EngineExecutorId,
    cpu: ProcessCpuContext,
    address_space_end: GuestVirtualAddress,
    frame: ExecutionFrame,
    // Retained and deliberately inert until JIT-006 performs real lowering.
    _compiler: CompilerContext,
    // Executors retain the mapping independently because the neutral contract
    // permits them to finish teardown after their domain handle is released.
    _executable_memory: SharedExecutableMemory,
    control: EngineControl,
    acknowledged_epoch: u64,
}

impl EngineExecutor for JitExecutor {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn executor_id(&self) -> EngineExecutorId {
        self.id
    }

    fn run_slice(&mut self, request: RunRequest<'_>) -> Result<ExecutionReport, EngineFault> {
        if request.cpu != self.cpu {
            return Err(fault(
                EngineFaultKind::InvalidRequest,
                0,
                "run request CPU context differs from the JIT domain",
                request.state,
            ));
        }
        let current_pc = current_pc(request.state);
        if current_pc >= self.address_space_end.get() {
            return Err(fault(
                EngineFaultKind::InvalidRequest,
                0,
                "canonical PC lies outside the bound address space",
                request.state,
            ));
        }

        self.frame.import_state(request.state);
        self.frame.memory.address_space = request.cpu.address_space_id().get();
        self.frame.memory.mapping_epoch = self.acknowledged_epoch;
        self.frame.control.instruction_budget = request.instruction_budget;
        let timer = request.timer.snapshot();
        self.frame.control.timer_counter = timer.counter;
        self.frame.control.timer_frequency = timer.frequency;
        self.frame.control.invalidation_epoch = self.acknowledged_epoch;
        self.frame.control.loader_return_valid = u32::from(request.loader_return.is_some());
        self.frame.control.loader_return =
            request.loader_return.map_or(0, GuestVirtualAddress::get);

        let pending = self.control.take_pending();
        if let Some(snapshot) = pending {
            self.frame.control.request_flags =
                (u32::from(snapshot.contains(CrossVcpuRequest::Preempt)) * CONTROL_PREEMPT)
                    | (u32::from(snapshot.contains(CrossVcpuRequest::CodeInvalidation))
                        * CONTROL_CODE_INVALIDATION);
            self.frame.control.event_mask = snapshot.event_mask;
            self.frame.control.invalidation_epoch = snapshot.invalidation_epoch;
        } else {
            self.frame.control.request_flags = 0;
            self.frame.control.event_mask = 0;
        }

        let dispatch = contain_rust_boundary(|| self.dispatch_without_published_code());
        let native_exit = match dispatch {
            Ok(exit) => exit,
            Err(_) => {
                self.commit_or_fault(request.state, 0)?;
                return Err(fault(
                    EngineFaultKind::Internal,
                    0,
                    "panic was contained at the JIT native-entry boundary",
                    request.state,
                ));
            }
        };
        self.frame.exit = native_exit;
        self.commit_or_fault(request.state, native_exit.instructions_executed)?;

        if let Some(snapshot) = pending {
            if snapshot.contains(CrossVcpuRequest::CodeInvalidation) {
                self.acknowledged_epoch = self.acknowledged_epoch.max(snapshot.invalidation_epoch);
                self.frame.memory.mapping_epoch = self.acknowledged_epoch;
            }
            self.control.acknowledge(snapshot);
        }

        Ok(ExecutionReport {
            instructions_executed: native_exit.instructions_executed,
            stop: normalize_exit(native_exit, request.cpu, request.state)?,
            context: request.state.register_context(),
        })
    }

    fn synchronize_invalidation(
        &mut self,
        epoch: u64,
        _state: &ThreadCpuState,
        _memory: &dyn nixe_cpu::memory::CpuMemory,
    ) -> Result<(), EngineFault> {
        self.acknowledged_epoch = self.acknowledged_epoch.max(epoch);
        self.frame.memory.mapping_epoch = self.acknowledged_epoch;
        self.control.acknowledge_invalidation(epoch);
        Ok(())
    }

    fn synchronize_address_space(
        &mut self,
        binding: DomainMemoryBinding<'_>,
        state: &ThreadCpuState,
    ) -> Result<(), EngineFault> {
        if binding.address_space != self.cpu.address_space_id() {
            return Err(fault(
                EngineFaultKind::InvalidRequest,
                0,
                "address-space synchronization belongs to a different domain",
                state,
            ));
        }
        self.address_space_end = binding.end_exclusive;
        self.synchronize_invalidation(binding.invalidation_generation, state, binding.memory)
    }

    fn control(&self) -> Option<EngineControl> {
        Some(self.control.clone())
    }

    fn clear_local_exclusive_reservation(&mut self) {}
}

impl JitExecutor {
    fn dispatch_without_published_code(&mut self) -> NativeExit {
        // JIT-006 supplies lowering. Until then the only honest execution
        // result is the explicit semantic fallback; no compilation work is
        // fabricated on this path.
        self.frame
            .execution_state()
            .expect("imported native frame has a valid execution state");
        let source_pc = self.frame.current_pc();
        if self.frame.control.event_mask != 0 {
            return NativeExit {
                kind: EXIT_PENDING_EVENT,
                detail: self.frame.control.event_mask,
                source_pc,
                ..NativeExit::default()
            };
        }
        if self.frame.control.request_flags & CONTROL_PREEMPT != 0 {
            return NativeExit {
                kind: EXIT_SAFEPOINT,
                source_pc,
                ..NativeExit::default()
            };
        }
        if self.frame.control.instruction_budget == 0 {
            return NativeExit {
                kind: EXIT_BUDGET_EXHAUSTED,
                source_pc,
                ..NativeExit::default()
            };
        }
        if self.frame.control.loader_return_valid != 0
            && source_pc == self.frame.control.loader_return
            && let Some(result_code) = self.frame.a64_result_code()
        {
            return NativeExit {
                kind: EXIT_LOADER_RETURN,
                source_pc,
                payload0: result_code,
                ..NativeExit::default()
            };
        }
        NativeExit {
            kind: EXIT_INTERPRET_ONE,
            source_pc,
            ..NativeExit::default()
        }
    }

    fn commit_or_fault(
        &self,
        state: &mut ThreadCpuState,
        instructions_executed: u64,
    ) -> Result<(), EngineFault> {
        self.frame.commit_state(state).map_err(|error| {
            fault(
                EngineFaultKind::Internal,
                instructions_executed,
                match error {
                    FrameError::StateKindChanged => {
                        "native frame attempted to change the canonical state representation"
                    }
                    FrameError::InconsistentA32ExecutionState => {
                        "native frame AArch32 state disagrees with CPSR.T"
                    }
                    FrameError::InvalidA32InstructionAddress => {
                        "native frame contained an invalid AArch32 instruction address"
                    }
                },
                state,
            )
        })
    }
}

fn normalize_exit(
    exit: NativeExit,
    cpu: ProcessCpuContext,
    state: &ThreadCpuState,
) -> Result<nixe_cpu_engine::EngineExit, EngineFault> {
    let execution_state = state.execution_state();
    let source = LocationDescriptor::new(
        GuestVirtualAddress::new(exit.source_pc),
        execution_state,
        cpu.profile().id(),
    );
    match exit.kind {
        EXIT_INTERPRET_ONE => Ok(nixe_cpu_engine::EngineExit::InterpretOne { source }),
        EXIT_BUDGET_EXHAUSTED => Ok(nixe_cpu_engine::EngineExit::BudgetExhausted),
        EXIT_SAFEPOINT => Ok(nixe_cpu_engine::EngineExit::Safepoint),
        EXIT_PENDING_EVENT => Ok(nixe_cpu_engine::EngineExit::PendingEvent { mask: exit.detail }),
        EXIT_LOADER_RETURN => Ok(nixe_cpu_engine::EngineExit::LoaderReturn {
            source,
            result_code: exit.payload0,
        }),
        EXIT_NONE => Err(fault(
            EngineFaultKind::Internal,
            exit.instructions_executed,
            "native frame returned without a normalized exit",
            state,
        )),
        _ => Err(fault(
            EngineFaultKind::Internal,
            exit.instructions_executed,
            format!("native frame returned unknown exit kind {}", exit.kind),
            state,
        )),
    }
}

fn descriptor() -> EngineDescriptor {
    EngineDescriptor {
        id: JIT_ENGINE_ID,
        name: "cranelift-jit".into(),
        kind: EngineKind::Jit,
        capabilities: capabilities(),
    }
}

/// Contains every Rust-side operation adjacent to native entry. Published
/// functions and helper slots use `extern "C"`, so unwind is not an ABI outcome;
/// helpers must perform the same containment before returning to native code.
fn contain_rust_boundary<T>(operation: impl FnOnce() -> T) -> Result<T, ()> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|_| ())
}

const fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        a64: true,
        a32: true,
        t32: true,
        interpret_one_fallback: true,
        concurrent_executors: true,
        max_safepoint_instructions: std::num::NonZeroU64::new(1),
        acknowledged_invalidation: true,
        deterministic_execution: true,
        // Direct canonical backing leases arrive in JIT-009.
        canonical_memory_binding: false,
        max_concurrent_executors: None,
    }
}

fn probe_host() -> HostSupport {
    if !cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
        return HostSupport::Unavailable {
            reason: CapabilityRejectionReason::PlatformUnsupported,
            detail: format!(
                "Cranelift JIT supports only x86-64 and AArch64 hosts, not {}",
                std::env::consts::ARCH
            )
            .into_boxed_str(),
        };
    }
    let builder = match cranelift_native::builder() {
        Ok(builder) => builder,
        Err(detail) => {
            return HostSupport::Unavailable {
                reason: CapabilityRejectionReason::HostUnavailable,
                detail: format!("Cranelift rejected the native host ISA: {detail}")
                    .into_boxed_str(),
            };
        }
    };
    let flags = settings::Flags::new(settings::builder());
    match builder.finish(flags) {
        Ok(isa) => HostSupport::Available(isa),
        Err(error) => HostSupport::Unavailable {
            reason: CapabilityRejectionReason::HostUnavailable,
            detail: format!("Cranelift native ISA configuration failed: {error}").into_boxed_str(),
        },
    }
}

fn rejection(reason: CapabilityRejectionReason, detail: &'static str) -> CapabilityRejection {
    CapabilityRejection {
        engine: JIT_ENGINE_ID,
        reason,
        detail: detail.into(),
    }
}

fn first_execution_state(profile: GuestCpuProfile) -> ExecutionState {
    [
        ExecutionState::A64,
        ExecutionState::A32,
        ExecutionState::T32,
    ]
    .into_iter()
    .find(|state| profile.allowed_execution_states().contains(*state))
    .expect("a guest CPU profile has at least one execution state")
}

fn current_pc(state: &ThreadCpuState) -> u64 {
    match state {
        ThreadCpuState::A64(state) => state.pc(),
        ThreadCpuState::A32(state) => u64::from(state.instruction_address()),
    }
}

fn domain_fault(
    cpu: ProcessCpuContext,
    kind: EngineFaultKind,
    message: impl Into<Box<str>>,
) -> EngineFault {
    let state = ThreadCpuState::new(
        cpu.thread_configuration(first_execution_state(cpu.profile()))
            .expect("the selected state belongs to the profile"),
    );
    fault(kind, 0, message, &state)
}

fn fault(
    kind: EngineFaultKind,
    instructions_executed: u64,
    message: impl Into<Box<str>>,
    state: &ThreadCpuState,
) -> EngineFault {
    EngineFault {
        engine: JIT_ENGINE_ID,
        kind,
        instructions_executed,
        message: message.into(),
        context: Box::new(state.register_context()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nixe_cpu::memory::ExecutionMemory;
    use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State};
    use nixe_cpu_engine::{EngineProvider, EngineTimer, TimerSnapshot};
    use nixe_memory::AddressSpaceId;

    use super::*;

    const SPACE: AddressSpaceId = AddressSpaceId::new(7);

    struct FixedTimer;

    impl EngineTimer for FixedTimer {
        fn snapshot(&self) -> TimerSnapshot {
            TimerSnapshot {
                counter: 123,
                frequency: 19_200_000,
            }
        }
    }

    fn cpu() -> ProcessCpuContext {
        ProcessCpuContext::new(GuestCpuProfile::switch_1(), SPACE)
    }

    fn bound_executor() -> (Box<dyn EngineDomain>, Box<dyn EngineExecutor>) {
        let provider = JitProvider::new();
        let mut domain = provider
            .create_domain(DomainRequest {
                domain: EngineDomainId::new(9),
                cpu: cpu(),
            })
            .unwrap();
        let memory = ExecutionMemory::new();
        domain
            .bind_memory(DomainMemoryBinding {
                address_space: SPACE,
                end_exclusive: GuestVirtualAddress::new(1 << 39),
                memory: &memory,
                invalidation_generation: 3,
            })
            .unwrap();
        let executor = domain.create_executor(EngineExecutorId::new(11)).unwrap();
        (domain, executor)
    }

    #[test]
    fn native_host_and_executable_memory_are_capability_gates() {
        let permitted = JitProvider::new();
        let report = permitted.probe(GuestCpuProfile::switch_1(), EngineCapabilities::default());
        if cfg!(all(
            any(target_arch = "x86_64", target_arch = "aarch64"),
            any(
                all(unix, not(target_vendor = "apple")),
                target_os = "macos",
                windows
            )
        )) {
            assert!(report.available, "{:?}", report.rejections);
        } else {
            assert!(!report.available);
            assert!(report.rejections.iter().any(|rejection| {
                rejection.reason == CapabilityRejectionReason::PlatformUnsupported
            }));
        }

        let denied = JitProvider::with_executable_error(
            ExecutableMemoryError::privilege_denied_for_test("sandbox forbids JIT"),
        );
        let report = denied.probe(GuestCpuProfile::switch_1(), EngineCapabilities::default());
        assert!(!report.available);
        assert!(report.rejections.iter().any(|rejection| {
            rejection.reason == CapabilityRejectionReason::PrivilegeUnavailable
                && rejection.detail.as_ref() == "sandbox forbids JIT"
        }));
        let error = match denied.create_domain(DomainRequest {
            domain: EngineDomainId::new(1),
            cpu: cpu(),
        }) {
            Ok(_) => panic!("denied executable-code policy created a domain"),
            Err(error) => error,
        };
        assert_eq!(error.kind, EngineFaultKind::Unavailable);
    }

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        any(all(unix, not(target_vendor = "apple")), target_os = "macos", windows)
    ))]
    #[test]
    fn providers_share_the_single_process_executable_memory_owner() {
        let first = JitProvider::new();
        let second = JitProvider::new();
        let first = first.executable_memory.as_ref().unwrap();
        let second = second.executable_memory.as_ref().unwrap();
        assert!(Arc::ptr_eq(first, second));
    }

    #[test]
    fn unverified_guest_profile_is_rejected() {
        let provider = JitProvider::new();
        let report = provider.probe(
            GuestCpuProfile::switch_2_native(),
            EngineCapabilities::default(),
        );
        assert!(!report.available);
        assert!(report.rejections.iter().any(|rejection| {
            rejection.reason == CapabilityRejectionReason::GuestProfileUnsupported
        }));
    }

    #[test]
    fn domain_requires_the_matching_canonical_memory_binding() {
        let provider = JitProvider::new();
        let mut domain = provider
            .create_domain(DomainRequest {
                domain: EngineDomainId::new(1),
                cpu: cpu(),
            })
            .unwrap();
        assert!(domain.create_executor(EngineExecutorId::new(1)).is_err());

        let memory = ExecutionMemory::new();
        let error = domain
            .bind_memory(DomainMemoryBinding {
                address_space: AddressSpaceId::new(99),
                end_exclusive: GuestVirtualAddress::new(1 << 39),
                memory: &memory,
                invalidation_generation: 0,
            })
            .unwrap_err();
        assert_eq!(error.kind, EngineFaultKind::InvalidRequest);
    }

    #[test]
    fn pre_lowering_executor_round_trips_state_and_requests_explicit_fallback() {
        let (_domain, mut executor) = bound_executor();
        let memory = ExecutionMemory::new();
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        let x0 = A64Register::General(A64GeneralRegister::new(0).unwrap());
        a64.write_x(x0, 0x1234_5678_9abc_def0);
        let expected = ThreadCpuState::A64(Box::new(a64));
        let mut state = expected.clone();

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 10,
                loader_return: None,
                timer: &FixedTimer,
            })
            .unwrap();

        assert_eq!(state, expected);
        assert_eq!(report.instructions_executed, 0);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::InterpretOne { source }
                if source.pc == GuestVirtualAddress::new(0x1000)
        ));
    }

    #[test]
    fn zero_budget_and_control_requests_are_normalized_without_fallback() {
        let (_domain, mut executor) = bound_executor();
        let memory = ExecutionMemory::new();
        let mut state = ThreadCpuState::A64(Box::new({
            let mut state = A64State::default();
            state.set_pc(0x1000);
            state
        }));
        let run = |executor: &mut dyn EngineExecutor, state: &mut ThreadCpuState, budget| {
            executor.run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state,
                instruction_budget: budget,
                loader_return: None,
                timer: &FixedTimer,
            })
        };

        assert_eq!(
            run(executor.as_mut(), &mut state, 0).unwrap().stop,
            nixe_cpu_engine::EngineExit::BudgetExhausted
        );
        let control = executor.control().unwrap();
        control.post_event(0x20);
        assert_eq!(
            run(executor.as_mut(), &mut state, 10).unwrap().stop,
            nixe_cpu_engine::EngineExit::PendingEvent { mask: 0x20 }
        );
        control.request(CrossVcpuRequest::Preempt);
        assert_eq!(
            run(executor.as_mut(), &mut state, 10).unwrap().stop,
            nixe_cpu_engine::EngineExit::Safepoint
        );
    }

    #[test]
    fn native_boundary_contains_panics() {
        let result = contain_rust_boundary(|| -> NativeExit {
            panic!("synthetic generated-code boundary failure")
        });
        assert!(result.is_err());
    }

    #[test]
    fn provider_is_usable_behind_the_neutral_trait() {
        let provider: Arc<dyn EngineProvider> = Arc::new(JitProvider::new());
        assert_eq!(provider.descriptor().kind, EngineKind::Jit);
    }
}
