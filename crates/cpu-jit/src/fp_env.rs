//! One host-FP owner for the current compiler and the tiered native boundary.
//! Host encodings and status translation are shared, not duplicated per tier.

use crate::abi::{HostFpState, NativeFrame};
use crate::fp_policy::native_fpcr_supported;

/// Translate either live or captured host sticky status with identical rules.
pub(crate) fn guest_status_from_host(abi: crate::abi::HostAbi, status: u64) -> u32 {
    let status = status as u32;
    match abi {
        crate::abi::HostAbi::Aarch64 => status & 0x0800_009f,
        crate::abi::HostAbi::X86_64 => {
            (status & 1)
                | ((status & (1 << 2)) >> 1)
                | ((status & (1 << 3)) >> 1)
                | ((status & (1 << 4)) >> 1)
                | ((status & (1 << 5)) >> 1)
                | ((status & (1 << 1)) << 6)
        }
    }
}

/// One encoding authority for the Rust FP owner and generated activation.
/// Call only after `native_fpcr_supported`; generated veneers embed the sixteen
/// supported encodings rather than reimplementing host control translation.
pub(crate) const fn native_control(abi: crate::abi::HostAbi, fpcr: u32) -> u32 {
    match abi {
        crate::abi::HostAbi::Aarch64 => fpcr & crate::fp_policy::NATIVE_FPCR_MASK,
        crate::abi::HostAbi::X86_64 => {
            let rounding = (fpcr >> 22) & 3;
            let rounding = ((rounding & 1) << 1) | ((rounding & 2) >> 1);
            0x1f80 | (rounding << 13) | if fpcr & (1 << 24) != 0 { 0x8040 } else { 0 }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedFpControl(pub u32);

impl std::fmt::Display for UnsupportedFpControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FPCR {:#010x} requires exact FP semantics, not native FP activation",
            self.0
        )
    }
}
impl std::error::Error for UnsupportedFpControl {}

/// A non-observing pause keeps hardware status separate from live software
/// FPSR. It borrows the existing owner, allocates nothing, and cannot cross OS
/// threads. Unlike canonical suspension, it must not clear guest sticky flags
/// on a successful continuation. There is deliberately no automatic resume on
/// drop: a failed/panicking observer must never resume guest execution.
#[must_use = "resume the observation or abort and merge status after canonical writeback"]
pub(crate) struct ObservationPause<'a> {
    owner: &'a mut HostFpState,
    guest: Option<(u64, u64)>,
    thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl ObservationPause<'_> {
    /// Restore the exact interrupted image, including accumulated sticky flags.
    /// An inactive segment stays inactive; observer FP effects are discarded.
    ///
    /// # Safety
    /// The observer succeeded and changed no guest architectural state. All
    /// locks/temporaries requiring general Rust work have been released. After
    /// this bounded leaf, run no general Rust until the next FP suspension.
    pub(crate) unsafe fn resume(self) {
        if let Some((control, status)) = self.guest {
            self.owner.active = 1;
            host::restore(control, status);
        } else {
            host::restore(self.owner.saved_control, self.owner.saved_status);
        }
    }

    /// Keep the caller environment and end the interrupted segment. The caller
    /// must retain this contribution until source canonical writeback completes,
    /// then OR it into FPSR exactly once. Merging before writeback would allow
    /// the still-live software FPSR to overwrite the hardware contribution.
    /// Never resume the guest after this failure path.
    #[must_use = "merge the interrupted guest status after canonical writeback"]
    pub(crate) fn abort(self) -> u32 {
        host::restore(self.owner.saved_control, self.owner.saved_status);
        self.owner.suspended = 0;
        let abi = if cfg!(target_arch = "x86_64") {
            crate::abi::HostAbi::X86_64
        } else {
            crate::abi::HostAbi::Aarch64
        };
        self.guest
            .map_or(0, |(_, status)| guest_status_from_host(abi, status))
    }
}

impl HostFpState {
    /// Pause a cold observation without touching canonical state or consuming
    /// host status. Read/restore uses the same host encoding authority as the
    /// normal FP owner; neither NativeFrame layout nor its ABI changes.
    ///
    /// # Safety
    /// `begin` ran on this OS thread and the invocation remains protected.
    /// Save all live caller-clobbered physical values before entering this
    /// System-ABI leaf. Do not change guest state while the pause is alive.
    /// Catch observer failure while retaining the pause, then consume it via
    /// `abort`; only a successful observer may consume it via `resume`.
    pub(crate) unsafe fn pause_observation(&mut self) -> ObservationPause<'_> {
        let guest = if self.active != 0 {
            Some(host::read())
        } else {
            None
        };
        host::restore(self.saved_control, self.saved_status);
        self.active = 0;
        // Any diagnostic runs only after restoration of the caller environment.
        assert_ne!(self.saved, 0, "FP observation before gateway save");
        ObservationPause {
            owner: self,
            guest,
            thread: std::marker::PhantomData,
        }
    }

    /// Save the caller environment once, before any guest segment.
    ///
    /// # Safety
    /// This owner must remain on this OS thread until `finish`. No other owner
    /// may manage this thread's FP environment during the invocation.
    pub unsafe fn begin(&mut self) {
        assert_eq!(self.active, 0, "guest FP segment is already active");
        assert_eq!(self.saved, 0, "caller FP environment was already saved");
        let (control, status) = host::read();
        self.saved_control = control;
        self.saved_status = status;
        self.saved = 1;
        self.suspended = 0;
    }

    /// Lazily activate supported native FP. Repeated activation on a compatible
    /// link does not clear accumulated guest status or replace the caller save.
    ///
    /// # Safety
    /// `begin` must have run on this thread. An active segment's FPCR must not
    /// change without `end`. No general Rust/helper work may execute while the
    /// guest environment is active; suspend first.
    pub unsafe fn ensure(&mut self, fpcr: u32) -> Result<(), UnsupportedFpControl> {
        assert_ne!(self.saved, 0, "guest FP activation before gateway save");
        if !native_fpcr_supported(fpcr) {
            return Err(UnsupportedFpControl(fpcr));
        }
        if self.active != 0 {
            return Ok(());
        }
        host::install_guest(fpcr);
        self.active = 1;
        self.suspended = 0;
        Ok(())
    }

    /// Restore the caller before a helper and return this segment's sticky
    /// guest FPSR contribution. The caller ORs it into authoritative software
    /// FPSR after any physical software-value writeback.
    ///
    /// # Safety
    /// Same invocation/thread as `begin`; all live native values are protected
    /// before this system-ABI operation.
    pub unsafe fn suspend(&mut self) -> u32 {
        if self.active == 0 {
            return 0;
        }
        let status = host::guest_status();
        host::restore(self.saved_control, self.saved_status);
        self.active = 0;
        self.suspended = 1;
        status
    }

    /// Resume only a successfully suspended segment, with cleared host status.
    ///
    /// # Safety
    /// Same invocation/thread, and the helper succeeded. FP mode replacement
    /// must finish the invocation, not resume a segment from the old mode.
    pub unsafe fn resume(&mut self, fpcr: u32) -> Result<(), UnsupportedFpControl> {
        if self.suspended == 0 {
            return Ok(());
        }
        unsafe { self.ensure(fpcr) }
    }

    /// End the invocation and restore the original caller environment, even
    /// when no native FP ran or a helper exited without resuming.
    ///
    /// # Safety
    /// Same invocation/thread as `begin`; canonical software state has been
    /// written. Commit the returned FPSR before announcing epoch quiescence.
    pub unsafe fn finish(&mut self) -> u32 {
        let was_active = self.active != 0;
        let status = unsafe { self.suspend() };
        if !was_active && self.saved != 0 {
            host::restore(self.saved_control, self.saved_status);
        }
        self.saved = 0;
        self.suspended = 0;
        status
    }
}

impl NativeFrame<'_> {
    /// Save the caller FP environment at gateway entry.
    ///
    /// # Safety
    /// Remain on this OS thread until `finish_fp`; no nested FP owner.
    pub unsafe fn begin_fp(&mut self) {
        unsafe { self.host_fp.begin() };
    }

    /// Activate the guest environment lazily, after the compiler's native-FP
    /// eligibility guard. Unsupported controls leave the owner unchanged.
    ///
    /// # Safety
    /// Same invocation/thread as `begin_fp`. Canonical FPCR is current and
    /// no general Rust work runs while active.
    #[cfg(test)]
    pub unsafe fn ensure_fp(&mut self) -> Result<(), UnsupportedFpControl> {
        unsafe { self.host_fp.ensure(*self.canonical.fpcr) }
    }

    /// Suspend before general Rust/helper work.
    ///
    /// # Safety
    /// Same invocation/thread as `begin_fp`; all observed software state,
    /// especially mapped FPSR, is canonical first.
    pub unsafe fn suspend_fp(&mut self) {
        let status = unsafe { self.host_fp.suspend() };
        unsafe { *self.canonical.fpsr |= status };
    }

    /// Resume the guest environment only on successful helper continuation.
    ///
    /// # Safety
    /// Same invocation/thread; canonical FPCR is current and unchanged in an
    /// existing segment. A mode replacement must finish the native invocation.
    pub unsafe fn resume_fp(&mut self) -> Result<(), UnsupportedFpControl> {
        unsafe { self.host_fp.resume(*self.canonical.fpcr) }
    }

    /// Complete canonical FP writeback and restore the original caller.
    /// Does not clear the execution epoch or reconcile the poll budget.
    ///
    /// # Safety
    /// Same invocation/thread as `begin_fp`; generated canonical data writeback
    /// has completed. No later write may overwrite this merged FPSR before
    /// announcing epoch quiescence.
    pub unsafe fn finish_fp(&mut self) {
        let status = unsafe { self.host_fp.finish() };
        unsafe { *self.canonical.fpsr |= status };
    }
}

#[cfg(target_arch = "x86_64")]
mod host {
    use core::arch::asm;

    const STATUS_MASK: u32 = 0x3f;

    pub(super) fn read() -> (u64, u64) {
        let mxcsr = read_mxcsr();
        (
            u64::from(mxcsr & !STATUS_MASK),
            u64::from(mxcsr & STATUS_MASK),
        )
    }

    pub(super) fn install_guest(fpcr: u32) {
        write_mxcsr(super::native_control(crate::abi::HostAbi::X86_64, fpcr));
    }

    pub(super) fn guest_status() -> u32 {
        super::guest_status_from_host(crate::abi::HostAbi::X86_64, u64::from(read_mxcsr()))
    }

    pub(super) fn restore(control: u64, status: u64) {
        write_mxcsr((control | status) as u32);
    }

    fn read_mxcsr() -> u32 {
        let mut value = 0_u32;
        unsafe { asm!("stmxcsr [{value}]", value = in(reg) &mut value, options(nostack)) };
        value
    }

    fn write_mxcsr(value: u32) {
        unsafe { asm!("ldmxcsr [{value}]", value = in(reg) &value, options(nostack)) };
    }
}

#[cfg(target_arch = "aarch64")]
mod host {
    use core::arch::asm;

    pub(super) fn read() -> (u64, u64) {
        let control: u64;
        let status: u64;
        unsafe {
            asm!("mrs {control}, fpcr", control = out(reg) control, options(nomem, nostack));
            asm!("mrs {status}, fpsr", status = out(reg) status, options(nomem, nostack));
        }
        (control, status)
    }

    pub(super) fn install_guest(fpcr: u32) {
        let guest = u64::from(super::native_control(crate::abi::HostAbi::Aarch64, fpcr));
        unsafe {
            asm!("msr fpcr, {guest}", guest = in(reg) guest, options(nomem, nostack));
            asm!("msr fpsr, xzr", options(nomem, nostack));
        }
    }

    pub(super) fn guest_status() -> u32 {
        let status: u64;
        unsafe { asm!("mrs {status}, fpsr", status = out(reg) status, options(nomem, nostack)) };
        super::guest_status_from_host(crate::abi::HostAbi::Aarch64, status)
    }

    pub(super) fn restore(control: u64, status: u64) {
        unsafe {
            asm!("msr fpcr, {control}", control = in(reg) control, options(nomem, nostack));
            asm!("msr fpsr, {status}", status = in(reg) status, options(nomem, nostack));
        }
    }
}
