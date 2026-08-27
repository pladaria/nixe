//! Concrete interpreter process, thread, and bounded execution loop.

use std::cell::RefCell;

use nixe_cpu::decode::{self, DecodeResult};
use nixe_cpu::execution::{
    ControlRequest, CpuControl, CpuExit, CpuFault, CpuFaultKind, CpuThreadId, ExecutionReport,
    MemoryBinding, RunRequest,
};
use nixe_cpu::location::{
    ExecutionState, InstructionEncoding, LocationDescriptor, current_location,
};
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::ThreadCpuState;
use nixe_memory::GuestVirtualAddress;

use crate::interpreter::{InstructionStep, InterpreterContext, InterpreterError, execute_decoded};

pub struct InterpreterProcess {
    cpu: ProcessCpuContext,
}

impl InterpreterProcess {
    #[must_use]
    pub const fn new(cpu: ProcessCpuContext) -> Self {
        Self { cpu }
    }

    pub fn bind_memory(&mut self, _binding: MemoryBinding<'_>) -> Result<(), CpuFault> {
        Ok(())
    }

    pub fn create_thread(&mut self, id: CpuThreadId) -> Result<InterpreterThread, CpuFault> {
        Ok(InterpreterThread::new(id, self.cpu))
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
    exclusive_monitor: RefCell<nixe_cpu::exclusive::ExclusiveMonitorState>,
    control: CpuControl,
}

impl InterpreterThread {
    fn new(id: CpuThreadId, cpu: ProcessCpuContext) -> Self {
        Self {
            id,
            cpu,
            exclusive_monitor: RefCell::new(Default::default()),
            control: CpuControl::default(),
        }
    }

    #[must_use]
    pub const fn id(&self) -> CpuThreadId {
        self.id
    }

    pub fn run_slice(&mut self, request: RunRequest<'_>) -> Result<ExecutionReport, CpuFault> {
        debug_assert_eq!(request.cpu, self.cpu);
        let mut remaining = request.instruction_budget;
        let mut executed = 0_u64;
        let context = InterpreterContext::new(
            request.cpu,
            request.memory,
            &self.exclusive_monitor,
            request.timer,
            &request.events,
        );
        loop {
            if let Some((source, result_code)) =
                loader_return_observation(request.cpu, request.state, request.loader_return)
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
            let source = current_location(request.cpu, request.state);
            let encoding = match source.execution_state {
                ExecutionState::A64 | ExecutionState::A32 => request
                    .memory
                    .fetch32(request.cpu.address_space_id(), source.pc)
                    .map(|fetched| InstructionEncoding::from_u32(fetched.bits)),
                ExecutionState::T32 => request
                    .memory
                    .fetch16(request.cpu.address_space_id(), source.pc)
                    .and_then(|first| {
                        if source.execution_state.instruction_size(first.bits)
                            == nixe_cpu::location::InstructionSize::Bits16
                        {
                            Ok(InstructionEncoding::from_u16(first.bits))
                        } else {
                            request
                                .memory
                                .fetch_t32_32(request.cpu.address_space_id(), source.pc)
                                .map(|fetched| InstructionEncoding::from_u32(fetched.bits))
                        }
                    }),
            };
            let encoding = match encoding {
                Ok(encoding) => encoding,
                Err(fault) => {
                    return Ok(self.report(executed, CpuExit::FetchFault { fault }, request.state));
                }
            };
            let decoded = match decode::decode(request.cpu.decoder(), source, encoding) {
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
                    InterpreterError::ContextMismatch { .. }
                    | InterpreterError::InvalidInstructionStream { .. } => CpuFaultKind::Internal,
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
        _state: &ThreadCpuState,
    ) -> Result<(), CpuFault> {
        self.control
            .acknowledge_invalidation(binding.invalidation_cursor.get());
        Ok(())
    }

    #[must_use]
    pub fn control(&self) -> CpuControl {
        self.control.clone()
    }

    pub fn prepare_shutdown(
        &mut self,
        binding: MemoryBinding<'_>,
        state: &ThreadCpuState,
    ) -> Result<(), CpuFault> {
        self.synchronize_address_space(binding, state)?;
        self.clear_local_exclusive_reservation();
        Ok(())
    }

    pub fn clear_local_exclusive_reservation(&mut self) {
        *self.exclusive_monitor.get_mut() = Default::default();
    }

    fn report(
        &self,
        instructions_executed: u64,
        stop: CpuExit,
        state: &ThreadCpuState,
    ) -> ExecutionReport {
        ExecutionReport {
            instructions_executed,
            stop,
            context: state.register_context(),
        }
    }
}

fn instruction_fault(
    kind: CpuFaultKind,
    instructions_executed: u64,
    state: &ThreadCpuState,
    message: impl ToString,
) -> CpuFault {
    CpuFault {
        backend: "interpreter",
        kind,
        instructions_executed,
        message: message.to_string().into(),
        context: Box::new(state.register_context()),
    }
}

fn loader_return_observation(
    cpu: ProcessCpuContext,
    state: &ThreadCpuState,
    loader_return: Option<GuestVirtualAddress>,
) -> Option<(LocationDescriptor, u64)> {
    let return_address = loader_return?;
    let ThreadCpuState::A64(state) = state else {
        return None;
    };
    (state.pc() == return_address.get()).then(|| {
        let source =
            LocationDescriptor::new(return_address, ExecutionState::A64, cpu.profile().id());
        let result_code = state.read_x(nixe_cpu::state::a64::A64Register::General(
            nixe_cpu::state::a64::A64GeneralRegister::new(0).expect("valid result register"),
        ));
        (source, result_code)
    })
}
