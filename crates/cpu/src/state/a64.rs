//! AArch64 architectural state.

use core::mem::offset_of;

/// Number of architecturally stored A64 general-purpose registers.
pub const GENERAL_REGISTER_COUNT: usize = 31;
/// Number of A64 SIMD/floating-point registers.
pub const VECTOR_REGISTER_COUNT: usize = 32;

/// A validated X0-X30 register index.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct A64GeneralRegister(u8);

impl A64GeneralRegister {
    #[must_use]
    pub const fn new(index: u8) -> Option<Self> {
        if (index as usize) < GENERAL_REGISTER_COUNT {
            Some(Self(index))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }
}

/// Register interpretation selected by an A64 instruction's operand rules.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum A64Register {
    General(A64GeneralRegister),
    StackPointer,
    Zero,
}

/// A64 N, Z, C, and V flags in their architectural bit positions.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct Nzcv(u32);

impl Nzcv {
    pub const MASK: u32 = 0xf000_0000;
    pub const N: u32 = 1 << 31;
    pub const Z: u32 = 1 << 30;
    pub const C: u32 = 1 << 29;
    pub const V: u32 = 1 << 28;

    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits & Self::MASK)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn negative(self) -> bool {
        self.0 & Self::N != 0
    }

    #[must_use]
    pub const fn zero(self) -> bool {
        self.0 & Self::Z != 0
    }

    #[must_use]
    pub const fn carry(self) -> bool {
        self.0 & Self::C != 0
    }

    #[must_use]
    pub const fn overflow(self) -> bool {
        self.0 & Self::V != 0
    }
}

/// Canonical A64 register state for one guest thread.
///
/// This C layout is an internal backend ABI only; it is not a save-state
/// format. Backend loads and stores must use the checked constants in
/// [`offsets`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C, align(16))]
pub struct A64State {
    x: [u64; GENERAL_REGISTER_COUNT],
    sp: u64,
    pc: u64,
    nzcv: Nzcv,
    vector: [u128; VECTOR_REGISTER_COUNT],
    fpcr: u32,
    fpsr: u32,
    tpidr_el0: u64,
    tpidrro_el0: u64,
}

impl Default for A64State {
    fn default() -> Self {
        Self {
            x: [0; GENERAL_REGISTER_COUNT],
            sp: 0,
            pc: 0,
            nzcv: Nzcv::default(),
            vector: [0; VECTOR_REGISTER_COUNT],
            fpcr: 0,
            fpsr: 0,
            tpidr_el0: 0,
            tpidrro_el0: 0,
        }
    }
}

impl A64State {
    /// Reads an X register. XZR reads as zero.
    #[must_use]
    pub fn read_x(&self, register: A64Register) -> u64 {
        match register {
            A64Register::General(register) => self.x[usize::from(register.index())],
            A64Register::StackPointer => self.sp,
            A64Register::Zero => 0,
        }
    }

    /// Writes an X register. Writes to XZR are discarded.
    pub fn write_x(&mut self, register: A64Register, value: u64) {
        match register {
            A64Register::General(register) => {
                self.x[usize::from(register.index())] = value;
            }
            A64Register::StackPointer => self.sp = value,
            A64Register::Zero => {}
        }
    }

    /// Reads the low 32 bits of an X register. WZR reads as zero.
    #[must_use]
    pub fn read_w(&self, register: A64Register) -> u32 {
        self.read_x(register) as u32
    }

    /// Writes a W register and zero-extends into its X register.
    /// Writes to WZR are discarded; WSP writes zero-extend into SP.
    pub fn write_w(&mut self, register: A64Register, value: u32) {
        self.write_x(register, u64::from(value));
    }

    #[must_use]
    pub const fn pc(&self) -> u64 {
        self.pc
    }

    pub const fn set_pc(&mut self, value: u64) {
        self.pc = value;
    }

    #[must_use]
    pub const fn nzcv(&self) -> Nzcv {
        self.nzcv
    }

    pub const fn set_nzcv(&mut self, value: Nzcv) {
        self.nzcv = value;
    }

    #[must_use]
    pub fn vector(&self, index: u8) -> Option<u128> {
        self.vector.get(usize::from(index)).copied()
    }

    pub fn set_vector(&mut self, index: u8, value: u128) -> bool {
        let Some(register) = self.vector.get_mut(usize::from(index)) else {
            return false;
        };
        *register = value;
        true
    }

    #[must_use]
    pub const fn fpcr(&self) -> u32 {
        self.fpcr
    }

    pub const fn set_fpcr(&mut self, value: u32) {
        self.fpcr = value;
    }

    #[must_use]
    pub const fn fpsr(&self) -> u32 {
        self.fpsr
    }

    pub const fn set_fpsr(&mut self, value: u32) {
        self.fpsr = value;
    }

    #[must_use]
    pub const fn tpidr_el0(&self) -> u64 {
        self.tpidr_el0
    }

    pub const fn set_tpidr_el0(&mut self, value: u64) {
        self.tpidr_el0 = value;
    }

    #[must_use]
    pub const fn tpidrro_el0(&self) -> u64 {
        self.tpidrro_el0
    }

    /// Sets the runtime-owned read-only thread-pointer register.
    pub const fn set_tpidrro_el0_from_runtime(&mut self, value: u64) {
        self.tpidrro_el0 = value;
    }
}

/// Stable offsets for the current internal machine-code backend ABI.
///
/// Changing one of these values is an intentional backend ABI change, not a
/// save-state migration.
pub mod offsets {
    use super::*;

    pub const X: usize = offset_of!(A64State, x);
    pub const SP: usize = offset_of!(A64State, sp);
    pub const PC: usize = offset_of!(A64State, pc);
    pub const NZCV: usize = offset_of!(A64State, nzcv);
    pub const VECTOR: usize = offset_of!(A64State, vector);
    pub const FPCR: usize = offset_of!(A64State, fpcr);
    pub const FPSR: usize = offset_of!(A64State, fpsr);
    pub const TPIDR_EL0: usize = offset_of!(A64State, tpidr_el0);
    pub const TPIDRRO_EL0: usize = offset_of!(A64State, tpidrro_el0);
}

#[cfg(test)]
mod tests {
    use core::mem::{align_of, size_of};

    use super::*;

    fn x(index: u8) -> A64Register {
        A64Register::General(A64GeneralRegister::new(index).unwrap())
    }

    #[test]
    fn x_and_w_aliases_obey_zero_extension_and_zero_register_rules() {
        let mut state = A64State::default();

        state.write_x(x(3), 0xffff_ffff_1234_5678);
        assert_eq!(state.read_w(x(3)), 0x1234_5678);
        state.write_w(x(3), 0x89ab_cdef);
        assert_eq!(state.read_x(x(3)), 0x0000_0000_89ab_cdef);

        state.write_x(A64Register::Zero, u64::MAX);
        state.write_w(A64Register::Zero, u32::MAX);
        assert_eq!(state.read_x(A64Register::Zero), 0);
        assert_eq!(state.read_w(A64Register::Zero), 0);

        state.write_w(A64Register::StackPointer, u32::MAX);
        assert_eq!(state.read_x(A64Register::StackPointer), u64::from(u32::MAX));
    }

    #[test]
    fn nzcv_keeps_only_architectural_flags() {
        let flags = Nzcv::from_bits(u32::MAX);
        assert_eq!(flags.bits(), Nzcv::MASK);
        assert!(flags.negative());
        assert!(flags.zero());
        assert!(flags.carry());
        assert!(flags.overflow());
    }

    #[test]
    fn vector_registers_preserve_all_128_bits() {
        let mut state = A64State::default();
        let value = 0xfedc_ba98_7654_3210_0123_4567_89ab_cdef;

        assert!(state.set_vector(31, value));
        assert_eq!(state.vector(31), Some(value));
        assert!(!state.set_vector(32, value));
        assert_eq!(state.vector(32), None);
    }

    #[test]
    fn backend_layout_offsets_are_intentional() {
        assert_eq!(align_of::<A64State>(), 16);
        assert_eq!(size_of::<A64State>(), 816);
        assert_eq!(offsets::X, 0);
        assert_eq!(offsets::SP, 248);
        assert_eq!(offsets::PC, 256);
        assert_eq!(offsets::NZCV, 264);
        assert_eq!(offsets::VECTOR, 272);
        assert_eq!(offsets::FPCR, 784);
        assert_eq!(offsets::FPSR, 788);
        assert_eq!(offsets::TPIDR_EL0, 792);
        assert_eq!(offsets::TPIDRRO_EL0, 800);
    }
}
