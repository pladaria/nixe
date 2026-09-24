//! Best-effort fatal diagnostics, including from the signal handler. Only
//! primitive integer formatting into a fixed stack buffer and one write(2);
//! never use Rust stderr locks, allocation, guest memory or code metadata here.

use core::fmt::{self, Write};

pub(super) enum Reason {
    RetryWithoutProgress,
    MissingDispatcher,
    RetryLimit,
    DispatcherRejected,
    UnattributedPc,
    DispatcherPanicked,
    NestedFault,
    #[cfg(target_arch = "aarch64")]
    MissingSignalFrame,
    #[cfg(target_arch = "x86_64")]
    ContextRestoreFailed,
    #[cfg(target_arch = "x86_64")]
    UnsupportedFpState,
}

impl Reason {
    fn name(&self) -> &'static str {
        match self {
            Self::RetryWithoutProgress => "retry-without-progress",
            Self::MissingDispatcher => "missing-dispatcher",
            Self::RetryLimit => "retry-limit",
            Self::DispatcherRejected => "dispatcher-rejected",
            Self::UnattributedPc => "unattributed-native-pc",
            Self::DispatcherPanicked => "dispatcher-panicked",
            Self::NestedFault => "nested-dispatch-fault",
            #[cfg(target_arch = "aarch64")]
            Self::MissingSignalFrame => "missing-signal-frame",
            #[cfg(target_arch = "x86_64")]
            Self::ContextRestoreFailed => "context-restore-failed",
            #[cfg(target_arch = "x86_64")]
            Self::UnsupportedFpState => "unsupported-signal-fp-state",
        }
    }
}

struct Line {
    bytes: [u8; 192],
    len: usize,
}
impl Write for Line {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        let end = self.len.checked_add(value.len()).ok_or(fmt::Error)?;
        let destination = self.bytes.get_mut(self.len..end).ok_or(fmt::Error)?;
        destination.copy_from_slice(value.as_bytes());
        self.len = end;
        Ok(())
    }
}

fn line(signal: i32, pc: usize, address: usize, reason: Reason) -> Line {
    let mut line = Line {
        bytes: [0; 192],
        len: 0,
    };
    let _ = writeln!(
        line,
        "nixe native fault: reason={} signal={} native_pc=0x{:016x} address=0x{:016x}",
        reason.name(),
        signal,
        pc,
        address
    );
    line
}

pub(super) fn report(signal: i32, pc: usize, address: usize, reason: Reason) {
    let line = line(signal, pc, address, reason);
    // All callers are terminal. Keep SIGPIPE blocked until termination so a
    // closed diagnostic pipe cannot replace the original fault signal.
    let mut blocked = unsafe { core::mem::zeroed::<libc::sigset_t>() };
    unsafe {
        libc::sigemptyset(&mut blocked);
        libc::sigaddset(&mut blocked, libc::SIGPIPE);
        if libc::sigprocmask(libc::SIG_BLOCK, &blocked, core::ptr::null_mut()) != 0 {
            return;
        }
    }
    // Do not retry failed/partial writes or replace the original fatal signal
    // with an I/O error. The diagnostic does not turn this into a recoverable exit.
    let _ = unsafe { libc::write(libc::STDERR_FILENO, line.bytes.as_ptr().cast(), line.len) };
}

pub(super) fn terminate(slot: &super::FaultSlot, reason: Reason) -> ! {
    use core::sync::atomic::Ordering;
    let signal = slot.signal.load(Ordering::Relaxed);
    report(
        signal,
        slot.native_pc.load(Ordering::Relaxed),
        slot.fault_address.load(Ordering::Relaxed),
        reason,
    );
    super::fatal_signal(signal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_is_bounded_and_keeps_full_addresses() {
        for reason in [
            Reason::RetryWithoutProgress,
            Reason::MissingDispatcher,
            Reason::RetryLimit,
            Reason::DispatcherRejected,
            Reason::UnattributedPc,
            Reason::DispatcherPanicked,
            Reason::NestedFault,
            #[cfg(target_arch = "aarch64")]
            Reason::MissingSignalFrame,
            #[cfg(target_arch = "x86_64")]
            Reason::ContextRestoreFailed,
            #[cfg(target_arch = "x86_64")]
            Reason::UnsupportedFpState,
        ] {
            let expected = format!(
                "nixe native fault: reason={} signal={} native_pc=0x{:016x} address=0x{:016x}\n",
                reason.name(),
                i32::MIN,
                usize::MAX,
                usize::MAX
            );
            let line = line(i32::MIN, usize::MAX, usize::MAX, reason);
            assert_eq!(&line.bytes[..line.len], expected.as_bytes());
            assert!(line.len < line.bytes.len());
        }
    }
}
