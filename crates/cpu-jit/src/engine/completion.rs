//! Owned cold exits: canonical state and caller FP are already restored, and
//! no invocation or execution lease may survive into these semantic providers.

use super::*;
use crate::lcq::{fp, system};
use crate::lifetime::unit::{EdgeKind, Instruction};
use nixe_cpu::{
    decode::{self, DecodeResult},
    error::{InstructionDiagnostic, UnallocatedEncoding},
    exception::ExceptionKind,
    execution::{ArchitecturalTimer, CpuExit, CpuFault, CpuFaultKind, VcpuEventState},
    location::{InstructionEncoding, LocationDescriptor},
    memory::DataAccessFaultReason,
};

impl JitThread {
    /// Complete exactly the owned exit returned by this vCPU's invocation.
    /// `progress` is work already charged before this completion. A stop/error
    /// retains its source state; a successful helper adds one instruction even
    /// when the native prefix already exhausted the slice. No native re-entry
    /// is permitted until the outer loop has handled that budget/control state.
    pub(crate) fn complete(
        &mut self,
        exit: invocation::Exit,
        state: &mut A64State,
        budget: &mut PollBudget,
        timer: &dyn ArchitecturalTimer,
        events: &VcpuEventState,
        progress: u64,
    ) -> Result<Option<CpuExit>, CpuFault> {
        let completion_sample = exit.completion_sample();
        let (stop, instruction) = match exit {
            invocation::Exit::Memory {
                instruction,
                outcome,
                ..
            } => {
                let stop = outcome.complete(
                    instruction,
                    state,
                    &*self.process.memory,
                    &mut self.exclusive,
                    progress,
                )?;
                if stop.is_some() {
                    return Ok(stop);
                }
                (None, instruction)
            }
            invocation::Exit::Native {
                guest, instruction, ..
            } => {
                let source = source(instruction);
                let stop = match guest.kind {
                    // Native execution already committed and charged these.
                    EdgeKind::Static
                    | EdgeKind::Taken
                    | EdgeKind::NotTaken
                    | EdgeKind::Call
                    | EdgeKind::Indirect
                    | EdgeKind::Return
                    | EdgeKind::FragmentLimit => return Ok(None),
                    // Deliver the explicit guest exception once. Keep PC at its
                    // source; the runtime owns exception/SVC dispatch and return.
                    EdgeKind::SupervisorCall(immediate) => Some(CpuExit::SupervisorCall {
                        source,
                        immediate: u32::from(immediate),
                    }),
                    EdgeKind::Breakpoint(immediate) => Some(CpuExit::ArchitecturalException {
                        source,
                        kind: ExceptionKind::Breakpoint,
                        syndrome: Some(u64::from(immediate)),
                    }),
                    EdgeKind::Unsupported | EdgeKind::InvalidInstruction => {
                        return Ok(Some(decode_stop(instruction)));
                    }
                    EdgeKind::FpSystem(operation) => {
                        system::complete_fp(operation, state)
                            .map_err(|error| internal(instruction, error, progress, state))?;
                        None
                    }
                    EdgeKind::RuntimeSystem(operation) => {
                        match system::complete_runtime(
                            operation,
                            state,
                            &mut system::RuntimeServices {
                                address_space: self.process.cpu.address_space_id(),
                                memory: &*self.process.memory,
                                timer,
                                events,
                                exclusive: &mut self.exclusive,
                            },
                        ) {
                            Ok(scheduled) => {
                                scheduled.map(|request| CpuExit::Scheduled { source, request })
                            }
                            Err(system::CompletionError::Invalid(error)) => {
                                return Err(internal(instruction, error, progress, state));
                            }
                            Err(system::CompletionError::Memory(fault)) => {
                                if let DataAccessFaultReason::HostBacking(detail) = &fault.reason {
                                    return Err(internal(instruction, detail, progress, state));
                                }
                                return Ok(Some(CpuExit::DataFault { source, fault }));
                            }
                        }
                    }
                    EdgeKind::ExclusiveStore(_) => {
                        return Err(internal(
                            instruction,
                            "exclusive-store exit bypassed the owned memory boundary",
                            progress,
                            state,
                        ));
                    }
                    kind @ (EdgeKind::FpCompare(_)
                    | EdgeKind::FpRound(_)
                    | EdgeKind::FpAdd(_)
                    | EdgeKind::FpDivide(_)
                    | EdgeKind::VectorFpDivide(_)
                    | EdgeKind::VectorFpMultiplyElement(_)
                    | EdgeKind::FpMultiply(_)
                    | EdgeKind::FpFused(_)
                    | EdgeKind::FpUnary(_)
                    | EdgeKind::FpToInteger(_)
                    | EdgeKind::IntegerToFp(_)
                    | EdgeKind::VectorIntegerToFp(_)) => {
                        let result = match kind {
                            EdgeKind::FpCompare(op) => fp::complete_compare(op, state),
                            EdgeKind::FpRound(op) => fp::complete_round(op, state),
                            EdgeKind::FpAdd(op) => fp::complete_add(op, state),
                            EdgeKind::FpDivide(op) => fp::complete_divide(op, state),
                            EdgeKind::VectorFpDivide(op) => fp::complete_vector_divide(op, state),
                            EdgeKind::VectorFpMultiplyElement(op) => {
                                fp::complete_vector_multiply_element(op, state)
                            }
                            EdgeKind::FpMultiply(op) => fp::complete_multiply(op, state),
                            EdgeKind::FpFused(op) => fp::complete_fused(op, state),
                            EdgeKind::FpUnary(op) => fp::complete_unary(op, state),
                            EdgeKind::FpToInteger(op) => fp::complete_to_integer(op, state),
                            EdgeKind::IntegerToFp(op) => fp::complete_from_integer(op, state),
                            EdgeKind::VectorIntegerToFp(op) => {
                                fp::complete_from_vector_integer(op, state)
                            }
                            _ => unreachable!("non-FP exits handled above"),
                        };
                        match result {
                            Ok(()) => None,
                            Err(fp::CompletionError::Invalid(error)) => {
                                return Err(internal(instruction, error, progress, state));
                            }
                            Err(fp::CompletionError::Trap(status)) => {
                                return Ok(Some(CpuExit::ArchitecturalException {
                                    source,
                                    kind: ExceptionKind::FloatingPoint,
                                    syndrome: Some(u64::from(
                                        nixe_cpu::semantics::a64_fp_simd::fp_status_bits(status),
                                    )),
                                }));
                            }
                        }
                    }
                };
                (stop, instruction)
            }
        };
        // The gateway has already reconciled native work. armed_span may be
        // zero after overshoot; this still charges exactly one cold completion.
        let poll = budget
            .reconcile(budget.armed_span - 1, false)
            .map_err(|error| {
                internal(
                    instruction,
                    format!("cold completion budget: {error:?}"),
                    progress,
                    state,
                )
            })?;
        if poll.sample
            && let Some(source) = completion_sample
        {
            self.process
                .lifetime
                .sample_completion(source, &mut self.samples)
                .map_err(|error| internal(instruction, error, progress.saturating_add(1), state))?;
        }
        Ok(stop)
    }
}

fn source(instruction: Instruction) -> LocationDescriptor {
    let key = instruction.key.block_key();
    LocationDescriptor::new(key.pc, key.profile)
}

fn internal(
    instruction: Instruction,
    error: impl std::fmt::Display,
    progress: u64,
    state: &A64State,
) -> CpuFault {
    CpuFault {
        backend: "jit",
        kind: CpuFaultKind::Internal,
        progress,
        message: format!(
            "LCQ completion source=[{}] encoding=0x{:08x}: {error}",
            source(instruction),
            instruction.bits
        )
        .into(),
        context: Box::new(state.register_context()),
    }
}

fn decode_stop(instruction: Instruction) -> CpuExit {
    let source = source(instruction);
    let encoding = InstructionEncoding::from_u32(instruction.bits);
    match decode::decode(instruction.key.block_key().platform, source, encoding) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
            CpuExit::UnsupportedSemantics {
                source,
                encoding,
                disassembly: decode::disassemble(&decoded.instruction).to_string().into(),
                coverage_id: decoded.instruction.coverage_id(),
            }
        }
        DecodeResult::Unallocated { reason, .. } => CpuExit::UnallocatedEncoding {
            error: UnallocatedEncoding::new(InstructionDiagnostic::new(source, encoding), reason),
        },
        DecodeResult::Reserved { name, reason, .. } => CpuExit::UnallocatedEncoding {
            error: UnallocatedEncoding::new(
                InstructionDiagnostic::new(source, encoding),
                format!("reserved {name}: {reason}"),
            ),
        },
    }
}
