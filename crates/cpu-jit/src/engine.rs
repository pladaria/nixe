//! Process-bound LCQ kernel; the runtime slice/exit loop is built on these
//! owners, without a legacy compiler, lookup or fault registry.

use crate::abi::{BlockKey, FpSpecialization, HostAbi, NativeFrame, PollBudget};
use crate::executable::Cache;
use crate::jit_error::Error;
use crate::lcq::{
    Compilation,
    compiler::{Compiler, PublishError},
    invocation,
};
use crate::lifetime::{self, Lifetime, Reader, compile::Request};
use nixe_cpu::{
    error::{InstructionFetchFault, InstructionFetchFaultReason},
    exclusive::ExclusiveMonitorState,
    execution::{CpuControl, VcpuEventState},
    memory::ExecutionMemory,
    profile::ProcessCpuContext,
    state::a64::A64State,
};
use nixe_cpu_direct_memory::NativeWorker;
use nixe_memory::{CpuMemoryBackend, GuestVirtualAddress};
use std::sync::Arc;

mod completion;
mod execution;

pub struct JitProcess {
    cpu: ProcessCpuContext,
    memory: Arc<ExecutionMemory>,
    lifetime: Arc<Lifetime>,
}

impl JitProcess {
    /// Stop new native/compile admission without waiting for this process's
    /// workers. Their next canonical boundary observes terminal closure.
    pub fn request_stop(&self) -> Result<(), Error> {
        self.lifetime
            .request_shutdown()
            .map_err(|error| Error::internal(error.to_string()))
    }

    /// Call again after outstanding worker/compiler owners have drained. A
    /// pending result is not successful teardown; no retained owner is ignored.
    pub fn try_shutdown(&self) -> Result<bool, Error> {
        self.lifetime
            .try_shutdown()
            .map_err(|error| Error::internal(error.to_string()))
    }

    /// Bind the complete memory authority before creating execution workers.
    /// No rebinding, borrowed owner, checked fallback or background tier exists.
    pub fn new(cpu: ProcessCpuContext, memory: Arc<ExecutionMemory>) -> Result<Self, Error> {
        if memory.cpu_memory_backend(cpu.address_space_id()) != Some(CpuMemoryBackend::LinuxDirect)
        {
            return Err(Error::unsupported(
                "LCQ JIT requires a bound LinuxDirect memory backend",
            ));
        }
        if let Some(error) = memory.direct_backend_failure() {
            return Err(Error::internal(error));
        }
        if memory
            .direct_address_space_view(cpu.address_space_id())
            .is_none()
        {
            return Err(Error::internal("LCQ JIT memory has no direct arena"));
        }
        let cache = Cache::new().map_err(|error| Error::internal(error.to_string()))?;
        let lifetime =
            Arc::new(Lifetime::new(cache).map_err(|error| Error::internal(error.to_string()))?);
        // Install last: failed construction must not strand a bound observer.
        // Memory owns Lifetime, not JitProcess, so there is no ownership cycle.
        memory
            .set_mutation_observer(lifetime.clone())
            .map_err(|error| Error::internal(error.to_string()))?;
        Ok(Self {
            cpu,
            memory,
            lifetime,
        })
    }
}

pub struct JitThread {
    process: Arc<JitProcess>,
    reader: Reader,
    compiler: Compiler,
    control: CpuControl,
    exclusive: ExclusiveMonitorState,
    // Preserve the poll phase across slices, without enabling functional samples.
    sample_remaining: i64,
}

pub(crate) enum Demand {
    Ready,
    /// A competing owner completed/canceled; restart admission in canonical mode.
    Retry,
    /// A fault on the first demanded word is reported, never cached as code.
    FetchFault(InstructionFetchFault),
}

impl JitThread {
    pub fn new(process: Arc<JitProcess>) -> Result<Self, Error> {
        let arena = process
            .memory
            .direct_address_space_view(process.cpu.address_space_id())
            .ok_or_else(|| Error::internal("LCQ JIT memory lost its direct arena"))?;
        let compiler = Compiler::for_arena(
            if cfg!(target_arch = "x86_64") {
                HostAbi::X86_64
            } else {
                HostAbi::Aarch64
            },
            arena.address_space_size,
        )?;
        let reader = process
            .lifetime
            .register()
            .map_err(|error| Error::internal(error.to_string()))?;
        Ok(Self {
            process,
            reader,
            compiler,
            control: CpuControl::default(),
            exclusive: ExclusiveMonitorState::default(),
            sample_remaining: crate::abi::SAMPLE_INTERVAL,
        })
    }

    pub fn control(&self) -> CpuControl {
        self.control.clone()
    }

    pub fn clear_local_exclusive_reservation(&mut self) {
        self.exclusive = ExclusiveMonitorState::default();
    }

    /// Acknowledge runtime notifications at a canonical boundary. Code
    /// invalidation itself is performed by the bound memory observer before
    /// mutation, not by replaying a runtime invalidation log here.
    pub fn synchronize_address_space(
        &mut self,
        binding: nixe_cpu::execution::MemoryBinding<'_>,
    ) -> Result<(), Error> {
        use nixe_cpu::memory::CpuMemory;
        if binding.address_space != self.process.cpu.address_space_id()
            || binding.memory.execution_gate_identity()
                != self.process.memory.execution_gate_identity()
        {
            return Err(Error::invalid(
                "LCQ execution memory differs from its immutable process binding",
            ));
        }
        self.control
            .acknowledge_invalidation(binding.invalidation_cursor.get());
        Ok(())
    }

    fn key(&self, pc: GuestVirtualAddress) -> Option<BlockKey> {
        BlockKey::new(self.process.cpu, pc, FpSpecialization::Dynamic)
    }

    /// One cold attempt, holding no invocation or memory lease. The slice loop
    /// handles stale admission, maintenance, capacity and stop requests; no
    /// output is relabelled and no failure is hidden in an internal retry loop.
    pub(crate) fn demand(&mut self, pc: GuestVirtualAddress) -> Result<Demand, PublishError> {
        let Some(key) = self.key(pc) else {
            return Ok(Demand::FetchFault(InstructionFetchFault::new(
                self.process.cpu.address_space_id(),
                pc,
                InstructionFetchFaultReason::Misaligned,
            )));
        };
        match self.reader.claim(key)? {
            Request::Ready => Ok(Demand::Ready),
            Request::Wait(waiter) => {
                waiter.wait()?;
                Ok(Demand::Retry)
            }
            Request::Owner(claim) => {
                let compilation = Compilation::capture(claim, &*self.process.memory)?;
                if compilation.fragment.image.words().is_empty() {
                    return compilation
                        .fragment
                        .image
                        .fault()
                        .cloned()
                        .map(Demand::FetchFault)
                        .ok_or_else(|| {
                            lifetime::Error::InvalidUnit("empty LCQ image has no fetch fault")
                                .into()
                        });
                }
                self.compiler.publish(
                    compilation,
                    &self.process.lifetime,
                    self.process.lifetime.executable_cache(),
                    &*self.process.memory,
                )?;
                Ok(Demand::Ready)
            }
        }
    }

    /// One native invocation (possibly a chain), returning owned output after
    /// native protections are gone, together with the reconciled budget.
    pub(crate) fn invoke(
        &mut self,
        returns: &mut crate::ReturnStack,
        worker: &mut NativeWorker,
        state: &mut A64State,
        budget: PollBudget,
        events: &VcpuEventState,
    ) -> Result<(Option<invocation::Exit>, PollBudget), invocation::Error> {
        if budget.slice_remaining <= 0 {
            return Err(invocation::Error::Native(
                crate::native::NativeReturnError::Budget(crate::abi::BudgetError::ExhaustedSlice),
            ));
        }
        let Some(key) = self.key(GuestVirtualAddress::new(state.pc())) else {
            return Ok((None, budget)); // Demand reports the precise fetch fault.
        };
        let faults = worker.faults().map_err(invocation::Error::Runtime)?;
        returns.prepare(key);
        let mut frame = NativeFrame::new(state, budget).with_return_stack(returns);
        // The process is bound by Reader::admit. These Arc-backed request words
        // remain alive throughout run; native cold polls only acquire-read them.
        frame.poll_requests[1] = self.control.pending_word_address() as *const _;
        frame.poll_requests[2] = events.pending_interrupts_address() as *const _;
        // SAFETY: this process fixes memory/platform/arena; this vCPU owns its
        // reader, compiler and monitor. Only this host's LCQ output and protected
        // static/PIC bridges are published. Prior FP restoration precedes entry.
        let result = unsafe {
            invocation::run(
                &mut self.reader,
                &mut frame,
                &self.process.memory,
                faults,
                &mut self.exclusive,
                key,
            )
        };
        result.map(|exit| (exit, frame.budget))
    }
}

#[cfg(test)]
mod tests;
