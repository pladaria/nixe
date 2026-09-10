//! Canonical slice loop for unlinked LCQ fragments. Demand and semantic
//! completion run only after invocation, FP ownership and memory lease release.

use super::*;
use nixe_cpu::{
    execution::{
        ArchitecturalTimer, ControlRequest, CpuExit, CpuFault, CpuFaultKind, ExecutionReport,
        VcpuEventState,
    },
    location::LocationDescriptor,
    state::a64::{A64GeneralRegister, A64Register},
};

impl JitThread {
    /// Memory and CPU identity come from this vCPU's immutable process binding;
    /// callers must not retain an execution lease across this cold loop.
    pub fn run_slice(
        &mut self,
        worker: &mut NativeWorker,
        state: &mut A64State,
        instruction_budget: u64,
        loader_return: Option<GuestVirtualAddress>,
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
                // Each LCQ terminal edge (including a backedge) returns canonical.
                // Control is independent of the sample/slice deadline. Linked
                // execution must preserve this check at its native boundaries.
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
                if loader_return == Some(pc) {
                    return Ok(CpuExit::LoaderReturn {
                        source: LocationDescriptor::new(pc, self.process.cpu.profile_id()),
                        result_code: state
                            .read_x(A64Register::General(A64GeneralRegister::new(0).unwrap())),
                    });
                }
                if budget.slice_remaining <= 0 {
                    return Ok(CpuExit::BudgetExhausted);
                }
                let progress = initial.abs_diff(budget.slice_remaining);
                let exit = match self.invoke(worker, state, budget) {
                    Ok((exit, reconciled)) => {
                        budget = reconciled;
                        exit
                    }
                    // Admission failed before native execution. Yield to the
                    // maintenance owner; do not spin, re-enter or acknowledge its
                    // outstanding work. Capacity/shutdown draining is separate.
                    Err(invocation::Error::Lifetime(lifetime::Error::Closed)) => {
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
                    // Finish an already-started instruction even when its native
                    // prefix exhausted the slice. Never execute it again merely
                    // to handle control or budget; completion returns owned stops.
                    let progress = initial.abs_diff(budget.slice_remaining);
                    if let Some(stop) =
                        self.complete(exit, state, &mut budget, timer, events, progress)?
                    {
                        return Ok(stop);
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
        result.map(|stop| report(stop, initial.abs_diff(budget.slice_remaining), state))
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
