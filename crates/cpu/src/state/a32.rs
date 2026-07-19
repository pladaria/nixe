//! AArch32 architectural state shared by A32 and T32 execution.

use core::{fmt, mem::offset_of};

use crate::{location::ExecutionState, location::InstructionSize};

/// Number of ordinary AArch32 registers, excluding R15/PC.
pub const GENERAL_REGISTER_COUNT: usize = 15;
/// Number of underlying 64-bit VFP registers.
pub const DOUBLE_REGISTER_COUNT: usize = 32;

/// A validated R0-R14 register index. R15 is accessed through the PC API.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct A32GeneralRegister(u8);

impl A32GeneralRegister {
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

/// AArch32 CPSR, including condition flags and the Thumb state bit.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct Cpsr(u32);

impl Cpsr {
    pub const N: u32 = 1 << 31;
    pub const Z: u32 = 1 << 30;
    pub const C: u32 = 1 << 29;
    pub const V: u32 = 1 << 28;
    pub const Q: u32 = 1 << 27;
    pub const T: u32 = 1 << 5;
    /// User mode is the reset mode for frontend-created process threads.
    pub const USER_MODE: u32 = 0b1_0000;

    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn execution_state(self) -> ExecutionState {
        if self.0 & Self::T == 0 {
            ExecutionState::A32
        } else {
            ExecutionState::T32
        }
    }

    #[must_use]
    pub const fn with_execution_state(self, state: ExecutionState) -> Option<Self> {
        match state {
            ExecutionState::A64 => None,
            ExecutionState::A32 => Some(Self(self.0 & !Self::T)),
            ExecutionState::T32 => Some(Self(self.0 | Self::T)),
        }
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

impl Default for Cpsr {
    fn default() -> Self {
        Self(Self::USER_MODE)
    }
}

/// Why an architectural branch target cannot be installed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InvalidBranchTarget {
    pub target: u32,
    pub destination_state: ExecutionState,
}

impl fmt::Display for InvalidBranchTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "misaligned branch target 0x{:08x} for {}",
            self.target, self.destination_state
        )
    }
}

impl std::error::Error for InvalidBranchTarget {}

/// Canonical AArch32 register state for one guest thread.
///
/// R15 is stored as the address of the current instruction. Architectural PC
/// operand reads apply the A32/T32 pipeline offset through [`Self::read_pc`].
/// This is the user-mode process view; privileged banked registers and
/// exception entry state belong to the runtime's exception model.
/// The C layout is an internal backend ABI and is not a save-state format.
#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C, align(8))]
pub struct A32State {
    r: [u32; GENERAL_REGISTER_COUNT],
    pc: u32,
    cpsr: Cpsr,
    d: [u64; DOUBLE_REGISTER_COUNT],
    fpscr: u32,
    tpidrurw: u32,
    tpidruro: u32,
}

impl Default for A32State {
    fn default() -> Self {
        Self::a32()
    }
}

impl A32State {
    /// Creates zeroed state in A32 execution state and user mode.
    #[must_use]
    pub const fn a32() -> Self {
        Self::new(false)
    }

    /// Creates zeroed state in T32 execution state and user mode.
    #[must_use]
    pub const fn t32() -> Self {
        Self::new(true)
    }

    const fn new(thumb: bool) -> Self {
        Self {
            r: [0; GENERAL_REGISTER_COUNT],
            pc: 0,
            cpsr: Cpsr::from_bits(Cpsr::USER_MODE | if thumb { Cpsr::T } else { 0 }),
            d: [0; DOUBLE_REGISTER_COUNT],
            fpscr: 0,
            tpidrurw: 0,
            tpidruro: 0,
        }
    }

    #[must_use]
    pub fn read_r(&self, register: A32GeneralRegister) -> u32 {
        self.r[usize::from(register.index())]
    }

    pub fn write_r(&mut self, register: A32GeneralRegister, value: u32) {
        self.r[usize::from(register.index())] = value;
    }

    /// Returns the stored address of the current instruction, without a
    /// pipeline offset. Frontend control flow and instruction fetch use this.
    #[must_use]
    pub const fn instruction_address(&self) -> u32 {
        self.pc
    }

    /// Reads architectural R15 as an instruction operand.
    ///
    /// A32 observes current instruction + 8, while T32 observes current
    /// instruction + 4. Operations that require word alignment must apply that
    /// operation-specific rule after this canonical read.
    #[must_use]
    pub const fn read_pc(&self) -> u32 {
        let pipeline_offset = match self.execution_state() {
            ExecutionState::A32 => 8,
            ExecutionState::T32 => 4,
            ExecutionState::A64 => unreachable!(),
        };
        self.pc.wrapping_add(pipeline_offset)
    }

    /// Installs an instruction address without changing A32/T32 state.
    pub fn set_instruction_address(&mut self, target: u32) -> Result<(), InvalidBranchTarget> {
        self.install_branch_target(target, self.execution_state())
    }

    #[must_use]
    pub const fn cpsr(&self) -> Cpsr {
        self.cpsr
    }

    pub const fn set_cpsr(&mut self, value: Cpsr) {
        self.cpsr = value;
    }

    #[must_use]
    pub const fn execution_state(&self) -> ExecutionState {
        self.cpsr.execution_state()
    }

    /// Implements BX: address bit 0 selects T32; an A32 destination must be
    /// word-aligned. The selector bit is never part of the installed PC.
    pub fn branch_exchange(&mut self, target: u32) -> Result<(), InvalidBranchTarget> {
        let destination_state = if target & 1 == 0 {
            ExecutionState::A32
        } else {
            ExecutionState::T32
        };
        self.install_branch_target(target & !1, destination_state)
    }

    /// Implements register-form BLX and records the architectural return
    /// address in LR before exchanging state.
    pub fn branch_link_exchange(
        &mut self,
        target: u32,
        instruction_size: InstructionSize,
    ) -> Result<(), InvalidBranchTarget> {
        let link = self.link_value(instruction_size);
        self.branch_exchange(target)?;
        self.r[14] = link;
        Ok(())
    }

    /// Implements immediate-form BLX, whose destination state is the opposite
    /// of the current A32/T32 state.
    pub fn branch_link_exchange_immediate(
        &mut self,
        target: u32,
        instruction_size: InstructionSize,
    ) -> Result<(), InvalidBranchTarget> {
        let destination_state = match self.execution_state() {
            ExecutionState::A32 => ExecutionState::T32,
            ExecutionState::T32 => ExecutionState::A32,
            ExecutionState::A64 => unreachable!(),
        };
        let link = self.link_value(instruction_size);
        self.install_branch_target(target, destination_state)?;
        self.r[14] = link;
        Ok(())
    }

    const fn link_value(&self, instruction_size: InstructionSize) -> u32 {
        let next = self.pc.wrapping_add(instruction_size.bytes() as u32);
        match self.execution_state() {
            ExecutionState::A32 => next & !1,
            ExecutionState::T32 => next | 1,
            ExecutionState::A64 => unreachable!(),
        }
    }

    fn install_branch_target(
        &mut self,
        target: u32,
        destination_state: ExecutionState,
    ) -> Result<(), InvalidBranchTarget> {
        let alignment = match destination_state {
            ExecutionState::A32 => 4,
            ExecutionState::T32 => 2,
            ExecutionState::A64 => unreachable!(),
        };
        if target & (alignment - 1) != 0 {
            return Err(InvalidBranchTarget {
                target,
                destination_state,
            });
        }
        self.pc = target;
        self.cpsr = self
            .cpsr
            .with_execution_state(destination_state)
            .expect("AArch32 destination state was already validated");
        Ok(())
    }

    /// Reads D0-D31, the canonical VFP/NEON backing storage.
    #[must_use]
    pub fn read_d(&self, index: u8) -> Option<u64> {
        self.d.get(usize::from(index)).copied()
    }

    pub fn write_d(&mut self, index: u8, value: u64) -> bool {
        let Some(register) = self.d.get_mut(usize::from(index)) else {
            return false;
        };
        *register = value;
        true
    }

    /// Reads S0-S31 as aliases of the two 32-bit halves of D0-D15.
    #[must_use]
    pub fn read_s(&self, index: u8) -> Option<u32> {
        if index >= 32 {
            return None;
        }
        let bits = self.d[usize::from(index / 2)];
        Some(if index & 1 == 0 {
            bits as u32
        } else {
            (bits >> 32) as u32
        })
    }

    /// Writes an S alias while preserving the other half of its D register.
    pub fn write_s(&mut self, index: u8, value: u32) -> bool {
        if index >= 32 {
            return false;
        }
        let register = &mut self.d[usize::from(index / 2)];
        if index & 1 == 0 {
            *register = (*register & 0xffff_ffff_0000_0000) | u64::from(value);
        } else {
            *register = (*register & 0x0000_0000_ffff_ffff) | (u64::from(value) << 32);
        }
        true
    }

    /// Reads Q0-Q15 as aliases of consecutive D-register pairs.
    #[must_use]
    pub fn read_q(&self, index: u8) -> Option<u128> {
        if index >= 16 {
            return None;
        }
        let low_index = usize::from(index) * 2;
        Some(u128::from(self.d[low_index]) | (u128::from(self.d[low_index + 1]) << 64))
    }

    pub fn write_q(&mut self, index: u8, value: u128) -> bool {
        if index >= 16 {
            return false;
        }
        let low_index = usize::from(index) * 2;
        self.d[low_index] = value as u64;
        self.d[low_index + 1] = (value >> 64) as u64;
        true
    }

    #[must_use]
    pub const fn fpscr(&self) -> u32 {
        self.fpscr
    }

    pub const fn set_fpscr(&mut self, value: u32) {
        self.fpscr = value;
    }

    #[must_use]
    pub const fn tpidrurw(&self) -> u32 {
        self.tpidrurw
    }

    pub const fn set_tpidrurw(&mut self, value: u32) {
        self.tpidrurw = value;
    }

    #[must_use]
    pub const fn tpidruro(&self) -> u32 {
        self.tpidruro
    }

    /// Sets the runtime-owned read-only user thread ID register.
    pub const fn set_tpidruro_from_runtime(&mut self, value: u32) {
        self.tpidruro = value;
    }
}

/// Stable offsets for the current internal machine-code backend ABI.
pub mod offsets {
    use super::*;

    pub const R: usize = offset_of!(A32State, r);
    pub const PC: usize = offset_of!(A32State, pc);
    pub const CPSR: usize = offset_of!(A32State, cpsr);
    pub const D: usize = offset_of!(A32State, d);
    pub const FPSCR: usize = offset_of!(A32State, fpscr);
    pub const TPIDRURW: usize = offset_of!(A32State, tpidrurw);
    pub const TPIDRURO: usize = offset_of!(A32State, tpidruro);
}

#[cfg(test)]
mod tests {
    use core::mem::{align_of, size_of};

    use super::*;

    fn r(index: u8) -> A32GeneralRegister {
        A32GeneralRegister::new(index).unwrap()
    }

    #[test]
    fn pc_reads_are_distinct_from_ordinary_gpr_access() {
        let mut a32 = A32State::a32();
        a32.set_instruction_address(0x1000).unwrap();
        a32.write_r(r(14), 0xfeed_beef);
        assert_eq!(a32.instruction_address(), 0x1000);
        assert_eq!(a32.read_pc(), 0x1008);
        assert_eq!(a32.read_r(r(14)), 0xfeed_beef);
        assert!(A32GeneralRegister::new(15).is_none());

        let mut t32 = A32State::t32();
        t32.set_instruction_address(0x1002).unwrap();
        assert_eq!(t32.read_pc(), 0x1006);
    }

    #[test]
    fn cpsr_preserves_flags_and_selects_execution_state() {
        let bits = Cpsr::N | Cpsr::Z | Cpsr::C | Cpsr::V | Cpsr::Q | Cpsr::USER_MODE;
        let mut state = A32State::a32();
        state.set_cpsr(Cpsr::from_bits(bits));

        assert!(state.cpsr().negative());
        assert!(state.cpsr().zero());
        assert!(state.cpsr().carry());
        assert!(state.cpsr().overflow());
        assert_eq!(state.execution_state(), ExecutionState::A32);

        state.set_cpsr(
            state
                .cpsr()
                .with_execution_state(ExecutionState::T32)
                .unwrap(),
        );
        assert_eq!(state.execution_state(), ExecutionState::T32);
        assert_eq!(state.cpsr().bits() & !Cpsr::T, bits);
    }

    #[test]
    fn s_d_and_q_views_alias_the_same_storage() {
        let mut state = A32State::a32();
        assert!(state.write_q(0, 0xdddd_dddd_dddd_dddd_ffff_ffff_eeee_eeee));
        assert_eq!(state.read_d(0), Some(0xffff_ffff_eeee_eeee));
        assert_eq!(state.read_d(1), Some(0xdddd_dddd_dddd_dddd));
        assert_eq!(state.read_s(0), Some(0xeeee_eeee));
        assert_eq!(state.read_s(1), Some(0xffff_ffff));

        assert!(state.write_s(1, 0x1234_5678));
        assert_eq!(state.read_d(0), Some(0x1234_5678_eeee_eeee));
        assert_eq!(
            state.read_q(0),
            Some(0xdddd_dddd_dddd_dddd_1234_5678_eeee_eeee)
        );
        assert_eq!(state.read_q(16), None);
    }

    #[test]
    fn bx_selects_state_from_bit_zero_and_validates_alignment() {
        let mut state = A32State::a32();
        state.branch_exchange(0x2001).unwrap();
        assert_eq!(state.execution_state(), ExecutionState::T32);
        assert_eq!(state.instruction_address(), 0x2000);

        state.branch_exchange(0x3000).unwrap();
        assert_eq!(state.execution_state(), ExecutionState::A32);
        assert_eq!(state.instruction_address(), 0x3000);

        assert_eq!(
            state.branch_exchange(0x3002).unwrap_err(),
            InvalidBranchTarget {
                target: 0x3002,
                destination_state: ExecutionState::A32,
            }
        );
    }

    #[test]
    fn blx_updates_link_register_and_interworks_in_both_directions() {
        let mut a32 = A32State::a32();
        a32.set_instruction_address(0x1000).unwrap();
        a32.branch_link_exchange(0x2001, InstructionSize::Bits32)
            .unwrap();
        assert_eq!(a32.execution_state(), ExecutionState::T32);
        assert_eq!(a32.read_r(r(14)), 0x1004);

        let mut t32 = A32State::t32();
        t32.set_instruction_address(0x4000).unwrap();
        t32.branch_link_exchange_immediate(0x5000, InstructionSize::Bits32)
            .unwrap();
        assert_eq!(t32.execution_state(), ExecutionState::A32);
        assert_eq!(t32.read_r(r(14)), 0x4005);
    }

    #[test]
    fn failed_link_exchange_is_atomic() {
        let mut state = A32State::a32();
        state.set_instruction_address(0x1000).unwrap();
        state.write_r(r(14), 0xaaaa_aaaa);

        assert!(
            state
                .branch_link_exchange(0x2002, InstructionSize::Bits32)
                .is_err()
        );
        assert_eq!(state.instruction_address(), 0x1000);
        assert_eq!(state.read_r(r(14)), 0xaaaa_aaaa);
        assert_eq!(state.execution_state(), ExecutionState::A32);
    }

    #[test]
    fn backend_layout_offsets_are_intentional() {
        assert_eq!(align_of::<A32State>(), 8);
        assert_eq!(size_of::<A32State>(), 344);
        assert_eq!(offsets::R, 0);
        assert_eq!(offsets::PC, 60);
        assert_eq!(offsets::CPSR, 64);
        assert_eq!(offsets::D, 72);
        assert_eq!(offsets::FPSCR, 328);
        assert_eq!(offsets::TPIDRURW, 332);
        assert_eq!(offsets::TPIDRURO, 336);
    }
}
