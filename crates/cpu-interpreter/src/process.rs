//! Concrete interpreter process, thread, and bounded execution loop.

use std::cell::RefCell;

use nixe_cpu::decode::{self, DecodeResult};
use nixe_cpu::execution::{
    ArchitecturalTimer, ControlRequest, CpuControl, CpuExit, CpuFault, CpuFaultKind, CpuThreadId,
    ExecutionReport, MemoryBinding, VcpuEventState,
};
use nixe_cpu::location::{InstructionEncoding, LocationDescriptor};
use nixe_cpu::memory::CpuMemory;
use nixe_cpu::memory::ExecutionMemoryLease;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::A64State;
use nixe_memory::{CpuMemoryBackend, DirectAddressSpaceView, GuestVirtualAddress};

use crate::interpreter::{InstructionStep, InterpreterContext, InterpreterError, execute_decoded};

pub struct InterpreterProcess {
    cpu: ProcessCpuContext,
    memory_backend: Option<InterpreterMemoryBackend>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InterpreterMemoryBackend {
    Checked,
    LinuxDirect {
        address_space: nixe_memory::AddressSpaceId,
        view: DirectAddressSpaceView,
    },
}

impl InterpreterProcess {
    #[must_use]
    pub const fn new(cpu: ProcessCpuContext) -> Self {
        Self {
            cpu,
            memory_backend: None,
        }
    }

    pub fn bind_memory(&mut self, binding: MemoryBinding<'_>) -> Result<(), CpuFault> {
        let backend = match binding.memory.cpu_memory_backend(binding.address_space) {
            CpuMemoryBackend::Checked => InterpreterMemoryBackend::Checked,
            CpuMemoryBackend::LinuxDirect => InterpreterMemoryBackend::LinuxDirect {
                address_space: binding.address_space,
                view: binding
                    .memory
                    .direct_address_space_view(binding.address_space)
                    .ok_or_else(|| backend_fault("LinuxDirect binding has no arena view"))?,
            },
        };
        if self
            .memory_backend
            .is_some_and(|current| current != backend)
        {
            return Err(backend_fault(
                "interpreter memory backend cannot change after binding",
            ));
        }
        self.memory_backend = Some(backend);
        Ok(())
    }

    pub fn create_thread(&mut self, id: CpuThreadId) -> Result<InterpreterThread, CpuFault> {
        InterpreterThread::new(id, self.cpu, self.memory_backend)
    }

    pub fn request_stop(&mut self) -> Result<(), CpuFault> {
        Ok(())
    }
    pub fn shutdown(&mut self) -> Result<(), CpuFault> {
        Ok(())
    }
}

pub struct InterpreterThread {
    id: CpuThreadId,
    cpu: ProcessCpuContext,
    memory_backend: Option<InterpreterMemoryBackend>,
    exclusive_monitor: RefCell<nixe_cpu::exclusive::ExclusiveMonitorState>,
    control: CpuControl,
    direct_memory: Option<RefCell<nixe_cpu_direct_memory::DirectScalarFrontend>>,
}

struct InterpreterDirectSlice<'a> {
    frontend: Option<&'a RefCell<nixe_cpu_direct_memory::DirectScalarFrontend>>,
}

impl<'a> InterpreterDirectSlice<'a> {
    fn begin(
        frontend: Option<&'a RefCell<nixe_cpu_direct_memory::DirectScalarFrontend>>,
    ) -> Result<Self, nixe_cpu_direct_memory::FaultRuntimeError> {
        if let Some(frontend) = frontend {
            frontend.borrow_mut().begin_slice()?;
        }
        Ok(Self { frontend })
    }
}

impl Drop for InterpreterDirectSlice<'_> {
    fn drop(&mut self) {
        if let Some(frontend) = self.frontend {
            frontend
                .borrow_mut()
                .end_slice()
                .expect("active interpreter direct-memory slice ends on its worker TID");
        }
    }
}

pub struct InterpreterRunRequest<'a> {
    pub memory: &'a dyn CpuMemory,
    /// Live mapping-stability proof required by a LinuxDirect backend.
    pub memory_lease: Option<ExecutionMemoryLease<'a>>,
    pub state: &'a mut A64State,
    pub instruction_budget: u64,
    pub loader_return: Option<GuestVirtualAddress>,
    pub timer: &'a dyn ArchitecturalTimer,
    pub events: VcpuEventState,
}

impl InterpreterThread {
    fn new(
        id: CpuThreadId,
        cpu: ProcessCpuContext,
        memory_backend: Option<InterpreterMemoryBackend>,
    ) -> Result<Self, CpuFault> {
        let direct_memory = match memory_backend {
            Some(InterpreterMemoryBackend::LinuxDirect {
                address_space,
                view,
            }) => Some(RefCell::new(
                unsafe { nixe_cpu_direct_memory::DirectScalarFrontend::new(view, address_space) }
                    .map_err(|error| backend_fault(error.to_string()))?,
            )),
            Some(InterpreterMemoryBackend::Checked) | None => None,
        };
        Ok(Self {
            id,
            cpu,
            memory_backend,
            exclusive_monitor: RefCell::new(Default::default()),
            control: CpuControl::default(),
            direct_memory,
        })
    }

    #[must_use]
    pub const fn id(&self) -> CpuThreadId {
        self.id
    }

    pub fn run_slice(
        &mut self,
        request: InterpreterRunRequest<'_>,
    ) -> Result<ExecutionReport, CpuFault> {
        let mut remaining = request.instruction_budget;
        let mut executed = 0_u64;
        self.validate_memory_backend(request.memory)?;
        self.validate_memory_lease(request.memory, request.memory_lease.as_ref())?;
        let _direct_slice = InterpreterDirectSlice::begin(self.direct_memory.as_ref())
            .map_err(|error| instruction_fault(CpuFaultKind::Internal, 0, request.state, error))?;
        let context = InterpreterContext::new(
            self.cpu,
            request.memory,
            &self.exclusive_monitor,
            request.timer,
            &request.events,
        )
        .with_direct_memory(self.direct_memory.as_ref());
        loop {
            if let Some((source, result_code)) =
                loader_return_observation(self.cpu, request.state, request.loader_return)
            {
                return Ok(self.report(
                    executed,
                    CpuExit::LoaderReturn {
                        source,
                        result_code,
                    },
                    request.state,
                ));
            }
            let pending_interrupts = request.events.take_pending_interrupts();
            if pending_interrupts != 0 {
                return Ok(self.report(
                    executed,
                    CpuExit::PendingEvent {
                        mask: pending_interrupts,
                    },
                    request.state,
                ));
            }
            if let Some(control) = self.control.take_pending() {
                self.control.acknowledge(control);
                if control.contains(ControlRequest::Preempt) {
                    return Ok(self.report(executed, CpuExit::Safepoint, request.state));
                }
                continue;
            }
            if remaining == 0 {
                return Ok(self.report(executed, CpuExit::BudgetExhausted, request.state));
            }
            let source = LocationDescriptor::new(
                GuestVirtualAddress::new(request.state.pc()),
                self.cpu.profile_id(),
            );
            let encoding = request
                .memory
                .fetch32(self.cpu.address_space_id(), source.pc)
                .map(|fetched| InstructionEncoding::from_u32(fetched.bits));
            let encoding = match encoding {
                Ok(encoding) => encoding,
                Err(fault) => {
                    return Ok(self.report(executed, CpuExit::FetchFault { fault }, request.state));
                }
            };
            let decoded = match decode::decode(self.cpu.decoder(), source, encoding) {
                DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
                    decoded
                }
                DecodeResult::Unallocated { reason, .. } => {
                    return Err(instruction_fault(
                        CpuFaultKind::InvalidRequest,
                        executed,
                        request.state,
                        format!(
                            "invalid instruction stream: {source} encoding={encoding} reason={reason}"
                        ),
                    ));
                }
                DecodeResult::Reserved { name, reason, .. } => {
                    return Err(instruction_fault(
                        CpuFaultKind::InvalidRequest,
                        executed,
                        request.state,
                        format!(
                            "invalid instruction stream: {source} encoding={encoding} reason={name}: reserved: {reason}"
                        ),
                    ));
                }
            };
            let step = execute_decoded(context, request.state, &decoded).map_err(|error| {
                let kind = match error {
                    InterpreterError::UnsupportedInstruction { .. } => CpuFaultKind::Unavailable,
                    InterpreterError::InvalidInstructionStream { .. }
                    | InterpreterError::DirectMemory { .. } => CpuFaultKind::Internal,
                };
                instruction_fault(kind, executed, request.state, error)
            })?;
            executed += 1;
            remaining -= 1;
            match step {
                InstructionStep::Continue => continue,
                InstructionStep::Exit(stop) => {
                    return Ok(self.report(executed, stop, request.state));
                }
            };
        }
    }

    pub fn synchronize_address_space(
        &mut self,
        binding: MemoryBinding<'_>,
    ) -> Result<(), CpuFault> {
        self.validate_memory_backend(binding.memory)?;
        self.control
            .acknowledge_invalidation(binding.invalidation_cursor.get());
        Ok(())
    }

    #[must_use]
    pub fn control(&self) -> CpuControl {
        self.control.clone()
    }

    pub fn prepare_shutdown(&mut self, binding: MemoryBinding<'_>) -> Result<(), CpuFault> {
        self.synchronize_address_space(binding)?;
        self.clear_local_exclusive_reservation();
        Ok(())
    }

    pub fn clear_local_exclusive_reservation(&mut self) {
        *self.exclusive_monitor.get_mut() = Default::default();
    }

    fn validate_memory_backend(&self, memory: &dyn CpuMemory) -> Result<(), CpuFault> {
        let expected = self
            .memory_backend
            .unwrap_or(InterpreterMemoryBackend::Checked);
        let actual = match memory.cpu_memory_backend(self.cpu.address_space_id()) {
            CpuMemoryBackend::Checked => InterpreterMemoryBackend::Checked,
            CpuMemoryBackend::LinuxDirect => InterpreterMemoryBackend::LinuxDirect {
                address_space: self.cpu.address_space_id(),
                view: memory
                    .direct_address_space_view(self.cpu.address_space_id())
                    .ok_or_else(|| backend_fault("LinuxDirect execution has no arena view"))?,
            },
        };
        if actual != expected {
            return Err(backend_fault(
                "interpreter execution memory differs from its immutable process binding",
            ));
        }
        Ok(())
    }

    fn validate_memory_lease(
        &self,
        memory: &dyn CpuMemory,
        lease: Option<&ExecutionMemoryLease<'_>>,
    ) -> Result<(), CpuFault> {
        if matches!(
            self.memory_backend,
            Some(InterpreterMemoryBackend::LinuxDirect { .. })
        ) && !lease.is_some_and(|lease| lease.authorizes(memory))
        {
            return Err(backend_fault(
                "LinuxDirect interpreter execution requires its live mapping lease",
            ));
        }
        Ok(())
    }

    fn report(&self, progress: u64, stop: CpuExit, state: &A64State) -> ExecutionReport {
        ExecutionReport {
            progress,
            stop,
            context: state.register_context(),
        }
    }
}

fn instruction_fault(
    kind: CpuFaultKind,
    progress: u64,
    state: &A64State,
    message: impl ToString,
) -> CpuFault {
    CpuFault {
        backend: "interpreter",
        kind,
        progress,
        message: message.to_string().into(),
        context: Box::new(state.register_context()),
    }
}

fn backend_fault(message: impl ToString) -> CpuFault {
    CpuFault {
        backend: "interpreter",
        kind: CpuFaultKind::Internal,
        progress: 0,
        message: message.to_string().into(),
        context: Box::new(A64State::default().register_context()),
    }
}

fn loader_return_observation(
    cpu: ProcessCpuContext,
    state: &A64State,
    loader_return: Option<GuestVirtualAddress>,
) -> Option<(LocationDescriptor, u64)> {
    let return_address = loader_return?;
    (state.pc() == return_address.get()).then(|| {
        let source = LocationDescriptor::new(return_address, cpu.profile_id());
        let result_code = state.read_x(nixe_cpu::state::a64::A64Register::General(
            nixe_cpu::state::a64::A64GeneralRegister::new(0).expect("valid result register"),
        ));
        (source, result_code)
    })
}
