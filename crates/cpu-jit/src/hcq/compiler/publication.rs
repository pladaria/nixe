//! Real HCQ output through the candidate-bound lifetime transaction. Memory
//! capture/revalidation never holds JIT state, a guest lease or execution epoch.

use super::{
    backend::{Compiler, Failure, Limit},
    *,
};
use crate::{
    executable::Tier,
    lifetime::{
        self,
        background::Frozen,
        unit::{Input, Instruction, UnitHandle},
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
            let image = Self::capture_run(
                memory,
                run.iter().map(|word| word.instruction),
                |dependency| {
                    frozen
                        .dependencies()
                        .binary_search_by_key(
                            &(dependency.page.get(), dependency.mapping_generation.get()),
                            |dep| (dep.page.get(), dep.mapping_generation.get()),
                        )
                        .is_ok()
                },
            )?;
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

    fn capture_structural(
        result: &crate::hcq::discovery::Structural<'_, '_>,
        memory: &(impl ExecutableMemory + MemoryInvalidationSource),
    ) -> Result<Self, Failure> {
        let image = Self::capture_inputs(result.capture_inputs()?, memory)?;
        result.check()?;
        image.validate(memory)?;
        result.check()?;
        Ok(image)
    }

    fn capture_inputs(
        inputs: crate::executable::Accounted<Vec<crate::lifetime::unit::Snapshot>>,
        memory: &(impl ExecutableMemory + MemoryInvalidationSource),
    ) -> Result<Self, Failure> {
        let cursor = memory.invalidation_cursor();
        let mut runs = Vec::new();
        for input in inputs.iter() {
            // LCQ images are contiguous. Chunking also handles the largest
            // resident image without truncating its u16 capture length.
            let mut start = 0;
            while start < input.instructions.len() {
                let count = (input.instructions.len() - start).min(usize::from(u16::MAX));
                runs.push(Self::capture_run(
                    memory,
                    input.instructions.iter().skip(start).take(count),
                    |dependency| input.dependencies.contains(&dependency),
                )?);
                start += count;
            }
        }
        // Strong code pins and their vector charge end before final checks.
        Ok(Self { runs, cursor })
    }

    fn capture_run(
        memory: &impl ExecutableMemory,
        mut words: impl ExactSizeIterator<Item = Instruction>,
        mut dependency: impl FnMut(nixe_cpu::memory::CodePageDependency) -> bool,
    ) -> Result<InstructionImage, Failure> {
        let count = NonZeroU16::new(
            u16::try_from(words.len())
                .map_err(|_| Failure::Failed(Error::internal("HCQ capture exceeds run limit")))?,
        )
        .ok_or_else(|| Failure::Failed(Error::internal("empty HCQ capture run")))?;
        let first = words.next().unwrap();
        let key = first.key.block_key();
        let image = memory.capture_instructions(key.address_space, key.pc, count, &|_, _| false);
        if image.fault().is_some()
            || image.words().len() != usize::from(count.get())
            || image
                .words()
                .iter()
                .zip(std::iter::once(first).chain(words))
                .enumerate()
                .any(|(index, (actual, expected))| {
                    actual.bits != expected.bits
                        || key
                            .pc
                            .get()
                            .checked_add(index as u64 * 4)
                            .and_then(|pc| key.at(GuestVirtualAddress::new(pc)))
                            != Some(expected.key.block_key())
                })
            || image.dependencies().any(|page| !dependency(page))
        {
            return Err(Failure::Cancelled);
        }
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

/// No backend emission, unit identity or executable allocation for a validated
/// no-op. Reuse the positive publisher's memory capture and exact-image checks.
pub(in crate::hcq) fn record_unchanged(
    frozen: &Frozen<'_, '_>,
    memory: &(impl ExecutableMemory + MemoryInvalidationSource),
) -> Result<bool, Failure> {
    let image = Image::capture(frozen, memory)?;
    let prepared = frozen.prepare_unchanged(image.cursor)?;
    image.validate(memory)?;
    prepared.install().map_err(Into::into)
}

/// Rejected discovery still needs exact memory and selection proof, including
/// discarded inputs. No candidate claims or backend emission are required.
pub(in crate::hcq) fn record_structural(
    result: &crate::hcq::discovery::Structural<'_, '_>,
    memory: &(impl ExecutableMemory + MemoryInvalidationSource),
) -> Result<bool, Failure> {
    let image = Image::capture_structural(result, memory)?;
    let prepared = result.prepare(image.cursor)?;
    image.validate(memory)?;
    prepared.install().map_err(Into::into)
}

/// A typed optimization limit is not a compiler failure, a seed rejection or
/// cache pressure. Validate all inspected inputs, not just emitted membership.
pub(in crate::hcq) fn record_backend_rejection(
    frozen: &Frozen<'_, '_>,
    _limit: Limit,
    memory: &(impl ExecutableMemory + MemoryInvalidationSource),
) -> Result<bool, Failure> {
    let image = Image::capture_inputs(frozen.capture_backend_inputs()?, memory)?;
    image.validate(memory)?;
    let prepared = frozen.prepare_backend_negative(image.cursor)?;
    image.validate(memory)?;
    prepared.install().map_err(Into::into)
}

impl Compiler {
    /// Compile and publish one frozen initial or replacement region. The caller must use the
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
