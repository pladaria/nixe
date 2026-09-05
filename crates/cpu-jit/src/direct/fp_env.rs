//! Host floating-point state owned by one native guest invocation.
//!
//! Generated arithmetic runs with a known, trap-masked host environment. Slow
//! Rust calls suspend that environment at the central call boundary; they do
//! not each implement their own FP protocol.

use super::NativeContext;

pub(super) use crate::abi::HostFpState;

/// Resets the lazy environment owner for a new native invocation.
///
/// # Safety
/// `context` must be the live context of the current native invocation.
pub(super) unsafe fn begin(context: &mut NativeContext) {
    debug_assert_eq!(context.host_fp.active, 0);
    debug_assert_eq!(context.host_fp.saved, 0);
}

/// Commits the current native status and restores the caller environment.
///
/// # Safety
/// `context` must be the live context passed to [`begin`].
pub(super) unsafe fn finish(context: &mut NativeContext) {
    unsafe { suspend_inner(context) };
    context.host_fp.saved = 0;
    context.host_fp.suspended = 0;
}

/// Lazily starts the guest environment for native status-producing FP.
pub(super) unsafe extern "C" fn ensure(context: *mut NativeContext) {
    let Some(context) = (unsafe { context.as_mut() }) else {
        return;
    };
    if context.host_fp.active != 0 {
        return;
    }
    if context.host_fp.saved == 0 {
        let (control, status) = host::read();
        context.host_fp.saved_control = control;
        context.host_fp.saved_status = status;
        context.host_fp.saved = 1;
    }
    host::install_guest(context.guest_fpcr);
    context.host_fp.active = 1;
    context.host_fp.suspended = 0;
}

/// Central generated-code boundary before entering general Rust.
pub(super) unsafe extern "C" fn suspend(context: *mut NativeContext) {
    let Some(context) = (unsafe { context.as_mut() }) else {
        return;
    };
    unsafe { suspend_inner(context) };
}

/// Ends the current guest FP segment without making it resumable.
///
/// This is used after an architectural FPCR replacement. The caller's host
/// environment is restored immediately, and a later native segment must be
/// established from the new canonical FPCR rather than resuming the old one.
pub(super) unsafe extern "C" fn end(context: *mut NativeContext) {
    let Some(context) = (unsafe { context.as_mut() }) else {
        return;
    };
    unsafe { suspend_inner(context) };
    context.host_fp.suspended = 0;
}

/// Central generated-code boundary after returning from general Rust.
pub(super) unsafe extern "C" fn resume(context: *mut NativeContext) {
    let Some(context) = (unsafe { context.as_mut() }) else {
        return;
    };
    if context.host_fp.suspended == 0 {
        return;
    }
    if context.host_fp.saved == 0 {
        return;
    }
    host::install_guest(context.guest_fpcr);
    context.host_fp.active = 1;
    context.host_fp.suspended = 0;
}

unsafe fn suspend_inner(context: &mut NativeContext) {
    if context.host_fp.active == 0 {
        return;
    }
    let status = host::guest_status();
    unsafe { *context.fpsr |= status };
    host::restore(context.host_fp.saved_control, context.host_fp.saved_status);
    context.host_fp.active = 0;
    context.host_fp.suspended = 1;
}

#[cfg(target_arch = "x86_64")]
mod host {
    use core::arch::asm;

    const STATUS_MASK: u32 = 0x3f;
    const DENORMALS_ARE_ZERO: u32 = 1 << 6;
    const EXCEPTION_MASKS: u32 = 0x1f80;
    const ROUNDING_MASK: u32 = 3 << 13;
    const FLUSH_TO_ZERO: u32 = 1 << 15;

    pub(super) fn read() -> (u64, u64) {
        let mxcsr = read_mxcsr();
        (
            u64::from(mxcsr & !STATUS_MASK),
            u64::from(mxcsr & STATUS_MASK),
        )
    }

    pub(super) fn install_guest(fpcr: u32) {
        let current = read_mxcsr();
        let arm_rounding = (fpcr >> 22) & 3;
        let host_rounding = match arm_rounding {
            0 => 0,
            1 => 2,
            2 => 1,
            3 => 3,
            _ => unreachable!(),
        };
        let flush = if fpcr & (1 << 24) != 0 {
            DENORMALS_ARE_ZERO | FLUSH_TO_ZERO
        } else {
            0
        };
        let guest = (current & !(STATUS_MASK | DENORMALS_ARE_ZERO | ROUNDING_MASK | FLUSH_TO_ZERO))
            | EXCEPTION_MASKS
            | (host_rounding << 13)
            | flush;
        write_mxcsr(guest);
    }

    pub(super) fn guest_status() -> u32 {
        let status = read_mxcsr() & STATUS_MASK;
        // x86: IE, DE, ZE, OE, UE, PE. Arm: IOC, DZC, OFC, UFC, IXC, IDC.
        (status & 1)
            | ((status & (1 << 2)) >> 1)
            | ((status & (1 << 3)) >> 1)
            | ((status & (1 << 4)) >> 1)
            | ((status & (1 << 5)) >> 1)
            | ((status & (1 << 1)) << 6)
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

    const GUEST_STATUS_MASK: u64 = 0x0800_009f;

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
        let guest = u64::from(fpcr & super::super::NATIVE_FPCR_MASK);
        unsafe {
            asm!("msr fpcr, {guest}", guest = in(reg) guest, options(nomem, nostack));
            asm!("msr fpsr, xzr", options(nomem, nostack));
        }
    }

    pub(super) fn guest_status() -> u32 {
        let status: u64;
        unsafe { asm!("mrs {status}, fpsr", status = out(reg) status, options(nomem, nostack)) };
        (status & GUEST_STATUS_MASK) as u32
    }

    pub(super) fn restore(control: u64, status: u64) {
        unsafe {
            asm!("msr fpcr, {control}", control = in(reg) control, options(nomem, nostack));
            asm!("msr fpsr, {status}", status = in(reg) status, options(nomem, nostack));
        }
    }
}
