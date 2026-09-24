//! The coordinator's narrowly scoped permission for executable mutation.
//! Patch/backlink ownership is the linker's responsibility, not this byte writer.

use super::*;
use crate::executable::{Allocation, Write};
use crate::lifetime::Transition;

/// Constructed only after validating the process-local registered owner while
/// Closed. The exclusive borrow prevents batch completion/reopening while the
/// cache uses this permit; the caller retains the allocation's strong owner.
pub(crate) struct ClosedCode<'c, 't, 'p> {
    allocation: &'c Allocation,
    transition: &'t mut Transition<'p>,
    finished: bool,
}
impl ClosedCode<'_, '_, '_> {
    pub(crate) fn allocation(&self) -> &Allocation {
        self.allocation
    }
}
impl Drop for ClosedCode<'_, '_, '_> {
    fn drop(&mut self) {
        if !self.finished {
            // Also covers unwinding after partially changing instructions. The
            // cache window has already closed/dropped, so locks never nest.
            let process = self.transition.process;
            process.fail(&mut process.lock(), Error::CacheFailed);
        }
    }
}

impl Transition<'_> {
    /// Apply cold, already-prepared writes while retaining the actual source.
    /// This is not a link installation: no roots/backlinks are registered here.
    ///
    /// # Safety
    /// Bytes must preserve the unit's ABI and immutable state/fault metadata.
    /// All newly callable targets must already have registered strong roots;
    /// retain old targets until their branches are removed and synchronization
    /// succeeds. No other process may execute this allocation. Caller buffers
    /// must not alias any executable span being written. Order island/bridge
    /// initialization before redirecting its source patch.
    pub(crate) unsafe fn patch_unit(
        &mut self,
        handle: UnitHandle,
        writes: &[Write<'_>],
    ) -> Result<(), Error> {
        let process = self.process;
        let code = {
            let state = process.lock();
            self.require_closed(&state)?;
            if handle.1 != process.identity {
                return Err(Error::StaleUnit);
            }
            let record = state.units.records.get(handle.0).ok_or(Error::StaleUnit)?;
            if matches!(
                record.lifecycle,
                Lifecycle::Unlinked | Lifecycle::Retired(_)
            ) {
                return Err(Error::StaleUnit);
            }
            Arc::clone(&record.code)
        };
        // No JIT-state lock during mprotect, instruction synchronization or
        // final owner release. The exact unit cannot be reclaimed under us.
        let mut permit = ClosedCode {
            allocation: &code.code.allocation,
            transition: self,
            finished: false,
        };
        process.cache.patch(&permit, writes)?;
        permit.finished = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
