//! Canonical slice loop around native LCQ execution. Demand and semantic
//! completion run only after invocation, FP ownership and memory lease release.

use super::*;
use nixe_cpu::execution::{
    ArchitecturalTimer, ControlRequest, CpuExit, CpuFault, CpuFaultKind, ExecutionReport,
    VcpuEventState,
};

impl JitThread {
    /// Memory and CPU identity come from this vCPU's immutable process binding;
    /// callers must not retain an execution lease across this cold loop.
    /// `returns` belongs to the scheduled guest and must follow it across slices
    /// and vCPU migration, independently of this worker's PIC registration.
    pub fn run_slice(
        &mut self,
        returns: &mut crate::ReturnStack,
        worker: &mut NativeWorker,
        state: &mut A64State,
        instruction_budget: u64,
        timer: &dyn ArchitecturalTimer,
        events: &VcpuEventState,
    ) -> Result<ExecutionReport, CpuFault> {
        let initial = i64::try_from(instruction_budget).map_err(|_| {
            fault(
                CpuFaultKind::InvalidRequest,
                "LCQ slice budget exceeds i64::MAX",
                0,
                state,
            )
        })?;
        if initial == 0 {
            return Ok(report(CpuExit::BudgetExhausted, 0, state));
        }
        let mut budget = PollBudget::new(self.sample_remaining, initial)
            .expect("vCPU retains a valid poll phase and positive slice");
        let result = (|| -> Result<CpuExit, CpuFault> {
            // At most one reclamation pass per miss, including soft pressure. A
            // failed retry reports capacity instead of recompiling/evicting forever.
            let mut capacity_pass = false;
            loop {
                // Check host notifications whenever execution is canonical.
                // Native chains observe requests through bounded cold polls,
                // independently of the caller's full slice budget.
                if let Some(control) = self.control.take_pending() {
                    // This is a notification, not authority to invalidate code:
                    // the bound memory observer unlinks before publishing changes.
                    // We hold no reader/lease and retain no entry across this ack.
                    self.control.acknowledge(control);
                    if control.contains(ControlRequest::Preempt) {
                        return Ok(CpuExit::Safepoint);
                    }
                }
                let mask = events.take_pending_interrupts();
                if mask != 0 {
                    return Ok(CpuExit::PendingEvent { mask });
                }
                let pc = GuestVirtualAddress::new(state.pc());
                if budget.slice_remaining <= 0 {
                    return Ok(CpuExit::BudgetExhausted);
                }
                let progress = initial.abs_diff(budget.slice_remaining);
                let exit = match self.invoke(returns, worker, state, budget, events) {
                    Ok((exit, reconciled)) => {
                        budget = reconciled;
                        exit
                    }
                    // No invocation/lease survives failed admission. Service
                    // owned link/cutover work if quiescent, otherwise yield to
                    // the remaining readers or the foreign maintenance owner.
                    Err(invocation::Error::Lifetime(lifetime::Error::Closed)) => {
                        if self.service_links(state, progress)? {
                            continue;
                        }
                        return Ok(CpuExit::Safepoint);
                    }
                    Err(invocation::Error::Lifetime(lifetime::Error::Shutdown)) => {
                        return Err(fault(
                            CpuFaultKind::Unavailable,
                            "LCQ process is shutting down",
                            progress,
                            state,
                        ));
                    }
                    Err(error) => {
                        return Err(fault(
                            CpuFaultKind::Internal,
                            format!("LCQ invocation: {error}"),
                            progress,
                            state,
                        ));
                    }
                };
                if let Some(exit) = exit {
                    capacity_pass = false;
                    let control = matches!(&exit, invocation::Exit::Native { returned, .. }
                        if returned.reason == crate::abi::NativeExitReason::Control);
                    // Finish an already-started instruction even when its native
                    // prefix exhausted the slice. Never execute it again merely
                    // to handle control or budget; completion returns owned stops.
                    let progress = initial.abs_diff(budget.slice_remaining);
                    if let Some(stop) =
                        self.complete(exit, state, &mut budget, timer, events, progress)?
                    {
                        return Ok(stop);
                    }
                    if control {
                        // Complete the instruction before servicing maintenance.
                        // This also handles deferred LinkPatch requests which
                        // deliberately reopened admission between batches.
                        self.service_links(state, initial.abs_diff(budget.slice_remaining))?;
                    }
                } else {
                    if !capacity_pass
                        && self
                            .process
                            .lifetime
                            .executable_cache()
                            .usage()
                            .map_err(|error| {
                                fault(CpuFaultKind::Internal, error.to_string(), progress, state)
                            })?
                            .needs_reclamation()
                    {
                        capacity_pass = true;
                        self.recover_capacity(state, progress)?;
                        continue;
                    }
                    match self.demand(pc) {
                        Ok(Demand::Ready | Demand::Retry)
                        | Err(PublishError::StaleCapture)
                        | Err(PublishError::Lifetime(lifetime::Error::StalePublication)) => {}
                        Ok(Demand::FetchFault(fault)) => return Ok(CpuExit::FetchFault { fault }),
                        Err(PublishError::Lifetime(lifetime::Error::Closed)) => {
                            if self.service_links(state, progress)? {
                                continue;
                            }
                            return Ok(CpuExit::Safepoint);
                        }
                        Err(PublishError::Lifetime(lifetime::Error::Shutdown)) => {
                            return Err(fault(
                                CpuFaultKind::Unavailable,
                                "LCQ process is shutting down",
                                progress,
                                state,
                            ));
                        }
                        Err(error) if error.capacity().is_some() => {
                            if capacity_pass {
                                return Err(fault(
                                    CpuFaultKind::Unavailable,
                                    format!("LCQ capacity at {pc}: {}", error.capacity().unwrap()),
                                    progress,
                                    state,
                                ));
                            }
                            // demand has dropped its claim, image and unpublished
                            // output before we close admission or reclaim storage.
                            capacity_pass = true;
                            self.recover_capacity(state, progress)?;
                        }
                        Err(error) => {
                            return Err(fault(
                                CpuFaultKind::Internal,
                                format!("LCQ demand: {error}"),
                                progress,
                                state,
                            ));
                        }
                    }
                    // Compilation/waiting may have received control or replaced
                    // admission. Restart all canonical checks, not a saved entry.
                }
            }
        })();
        self.sample_remaining = budget.sample_remaining;
        result
            .map(|stop| report(stop, initial.abs_diff(budget.slice_remaining), state))
            .map_err(|mut fault| {
                // Enrich terminal internal failure only after the canonical
                // path captured exact progress/state and restored host FP.
                // No extra lookup or lock is added to successful execution.
                if fault.kind == CpuFaultKind::Internal
                    && let Some(error) = self.process.lifetime.background_failure()
                {
                    fault.message = error.detail;
                }
                fault
            })
    }

    fn service_links(&self, state: &A64State, progress: u64) -> Result<bool, CpuFault> {
        self.process.lifetime.try_service_links().map_err(|error| {
            fault(
                if matches!(
                    error,
                    lifetime::Error::Capacity(_) | lifetime::Error::Shutdown
                ) {
                    CpuFaultKind::Unavailable
                } else {
                    CpuFaultKind::Internal
                },
                format!("LCQ link maintenance at {:#x}: {error}", state.pc()),
                progress,
                state,
            )
        })
    }

    fn recover_capacity(&self, state: &A64State, progress: u64) -> Result<(), CpuFault> {
        self.process.lifetime.recover_capacity().map_err(|error| {
            fault(
                if matches!(
                    error,
                    lifetime::Error::Capacity(_) | lifetime::Error::Shutdown
                ) {
                    CpuFaultKind::Unavailable
                } else {
                    CpuFaultKind::Internal
                },
                format!("LCQ reclamation at {:#x}: {error}", state.pc()),
                progress,
                state,
            )
        })
    }
}

fn report(stop: CpuExit, progress: u64, state: &A64State) -> ExecutionReport {
    ExecutionReport {
        progress,
        stop,
        context: Some(state.register_context()),
    }
}

fn fault(
    kind: CpuFaultKind,
    message: impl Into<Box<str>>,
    progress: u64,
    state: &A64State,
) -> CpuFault {
    CpuFault {
        backend: "jit",
        kind,
        progress,
        message: message.into(),
        context: Box::new(state.register_context()),
    }
}
