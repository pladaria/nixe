//! Real HCQ output through the candidate-bound lifetime transaction. Memory
//! capture/revalidation never holds JIT state, a guest lease or execution epoch.

use super::{
    backend::{Compiler, Failure},
    *,
};
use crate::{
    executable::Tier,
    lifetime::{
        self,
        background::Frozen,
        unit::{Input, UnitHandle},
    },
};
use nixe_cpu::memory::{ExecutableMemory, InstructionImage};
use nixe_memory::{MemoryInvalidationCursor, MemoryInvalidationSource};
use std::num::NonZeroU16;

struct Image {
    runs: Vec<InstructionImage>,
    cursor: MemoryInvalidationCursor,
}

impl Image {
    fn capture(
        frozen: &Frozen<'_, '_>,
        memory: &(impl ExecutableMemory + MemoryInvalidationSource),
    ) -> Result<Self, Failure> {
        frozen.check()?;
        let cursor = memory.invalidation_cursor();
        let mut runs = Vec::new();
        let mut words = frozen.graph().instructions.as_slice();
        while !words.is_empty() {
            let count = 1 + words
                .windows(2)
                .take_while(|pair| {
                    pair[0].instruction.key.block_key().pc.get().checked_add(4)
                        == Some(pair[1].instruction.key.block_key().pc.get())
                })
                .count();
            let (run, rest) = words.split_at(count);
            let key = run[0].instruction.key.block_key();
            let image = memory.capture_instructions(
                key.address_space,
                key.pc,
                NonZeroU16::new(u16::try_from(count).map_err(|_| {
                    Failure::Failed(Error::internal("HCQ capture exceeds instruction ceiling"))
                })?)
                .unwrap(),
                &|_, _| false,
            );
            if image.fault().is_some()
                || image.words().len() != run.len()
                || image
                    .words()
                    .iter()
                    .zip(run)
                    .any(|(a, b)| a.bits != b.instruction.bits)
                || image.dependencies().any(|dependency| {
                    frozen
                        .dependencies()
                        .binary_search_by_key(
                            &(dependency.page.get(), dependency.mapping_generation.get()),
                            |dep| (dep.page.get(), dep.mapping_generation.get()),
                        )
                        .is_err()
                })
            {
                return Err(Failure::Cancelled);
            }
            runs.push(image);
            words = rest;
        }
        // Capture can arm tracking and close/reopen execution. Only a change
        // to an actual captured LCQ input invalidates the candidate.
        frozen.check()?;
        let image = Self { runs, cursor };
        image.validate(memory)?;
        Ok(image)
    }

    fn validate(
        &self,
        memory: &(impl ExecutableMemory + MemoryInvalidationSource),
    ) -> Result<(), Failure> {
        if self
            .runs
            .iter()
            .any(|image| !memory.image_is_current(image))
        {
            return Err(Failure::Cancelled);
        }
        Ok(())
    }
}

impl Compiler {
    /// Compile and publish one frozen initial region. The caller must use the
    /// executable-memory authority bound to this candidate's Lifetime. Workers
    /// handle typed backend rejection separately from cancellation/pressure.
    pub(in crate::hcq) fn publish(
        &self,
        context: &mut Context,
        frontend: &mut FunctionBuilderContext,
        frozen: &Frozen<'_, '_>,
        memory: &(impl ExecutableMemory + MemoryInvalidationSource),
    ) -> Result<UnitHandle, Failure> {
        let image = Image::capture(frozen, memory)?;
        let process = frozen.lifetime();
        let identity = process.begin_unit(Tier::Hcq)?;
        let analysis = frozen.analyze()?;
        let body = self
            .emit(
                context,
                frontend,
                frozen.graph(),
                &analysis,
                frozen.entries(),
            )
            .map_err(Failure::Failed)?;
        let staged = self.finish(context, body, frozen.graph(), identity.version())?;
        frozen.check()?;
        image.validate(memory)?;
        let islands = staged
            .states
            .iter()
            .filter(|state| {
                state
                    .transfer
                    .as_ref()
                    .is_some_and(|transfer| transfer.static_target.is_some())
            })
            .count();
        let code = process
            .executable_cache()
            .install_with_islands(staged.output, Tier::Hcq, islands, |_| None)
            .map_err(lifetime::Error::from)?;
        let input = Input {
            identity,
            code,
            tier: Tier::Hcq,
            instructions: frozen
                .graph()
                .instructions
                .iter()
                .map(|word| word.instruction)
                .collect(),
            entries: staged.entries,
            dependencies: frozen.dependencies().into(),
            cursor: image.cursor,
            states: staged.states,
            faults: staged.faults,
        };
        let prepared = frozen.prepare(input, memory.invalidation_signal())?;
        // Allocation and directory construction may be long. Recheck memory
        // last, outside JIT state; coordinated changes after this check are
        // rejected by exact input/claim checks inside publish(). Each source
        // LCQ unit is invalidated by the bound coordinator before its memory
        // dependency can change, including changes while publication waits.
        image.validate(memory)?;
        frozen.check()?;
        prepared.publish().map_err(Into::into)
    }
}

#[cfg(test)]
mod tests;
