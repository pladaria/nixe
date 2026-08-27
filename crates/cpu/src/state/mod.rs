//! Canonical A64 guest thread state and diagnostic context.

pub mod a64;

use core::fmt;

use nixe_memory::GuestVirtualAddress;

pub use a64::{A64State as ThreadCpuState, Nzcv};

/// Bounded, pointer-free A64 context suitable for runtime diagnostics.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RegisterContext {
    pub x: [u64; a64::GENERAL_REGISTER_COUNT],
    pub sp: u64,
    pub pc: GuestVirtualAddress,
    pub nzcv: Nzcv,
}

impl fmt::Display for RegisterContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("state=A64")?;
        for (index, value) in self.x.iter().enumerate() {
            write!(formatter, " x{index}=0x{value:016x}")?;
        }
        write!(
            formatter,
            " sp=0x{:016x} pc={} nzcv=0x{:08x} flags=N{}Z{}C{}V{}",
            self.sp,
            self.pc,
            self.nzcv.bits(),
            u8::from(self.nzcv.negative()),
            u8::from(self.nzcv.zero()),
            u8::from(self.nzcv.carry()),
            u8::from(self.nzcv.overflow()),
        )
    }
}
