//! Normalized instruction operands shared by the A32 and T32 frontends.
//!
//! Encoding-specific decoders construct these values once. Lifters and the
//! reference interpreter must not recover operands from raw instruction bits.

use crate::semantics::shifts::A32ShiftKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataOperation {
    And,
    ExclusiveOr,
    Subtract,
    ReverseSubtract,
    Add,
    AddCarry,
    SubtractCarry,
    ReverseSubtractCarry,
    Test,
    TestExclusiveOr,
    Compare,
    CompareNegative,
    Or,
    Move,
    BitClear,
    MoveNot,
}

impl DataOperation {
    #[must_use]
    pub const fn from_a32_opcode(opcode: u8) -> Self {
        match opcode {
            0 => Self::And,
            1 => Self::ExclusiveOr,
            2 => Self::Subtract,
            3 => Self::ReverseSubtract,
            4 => Self::Add,
            5 => Self::AddCarry,
            6 => Self::SubtractCarry,
            7 => Self::ReverseSubtractCarry,
            8 => Self::Test,
            9 => Self::TestExclusiveOr,
            10 => Self::Compare,
            11 => Self::CompareNegative,
            12 => Self::Or,
            13 => Self::Move,
            14 => Self::BitClear,
            15 => Self::MoveNot,
            _ => unreachable!(),
        }
    }

    #[must_use]
    pub const fn is_test(self) -> bool {
        matches!(
            self,
            Self::Test | Self::TestExclusiveOr | Self::Compare | Self::CompareNegative
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShiftAmount {
    Immediate(u8),
    Register(u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shift {
    pub kind: A32ShiftKind,
    pub amount: ShiftAmount,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShifterOperand {
    Immediate { value: u32, rotation: u8 },
    Register { rm: u8, shift: Shift },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DataProcessing {
    pub operation: DataOperation,
    pub set_flags: bool,
    pub rn: u8,
    pub rd: u8,
    pub operand2: ShifterOperand,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Multiply {
    pub rd: u8,
    pub rn: u8,
    pub rs: u8,
    pub rm: u8,
    pub accumulate: bool,
    pub set_flags: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemorySize {
    Byte,
    Halfword,
    Word,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryOffset {
    Immediate(u32),
    Register { rm: u8, shift: Shift },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SingleTransfer {
    pub load: bool,
    pub signed: bool,
    pub size: MemorySize,
    pub rn: u8,
    pub rt: u8,
    pub offset: MemoryOffset,
    pub add: bool,
    pub pre_index: bool,
    pub writeback: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultipleTransfer {
    pub load: bool,
    pub rn: u8,
    pub registers: u16,
    pub increment: bool,
    pub before: bool,
    pub writeback: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExclusiveTransfer {
    pub load: bool,
    pub size: MemorySize,
    pub rn: u8,
    pub rt: u8,
    pub status: Option<u8>,
    pub acquire: bool,
    pub release: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorSize {
    D,
    Q,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorOperation {
    Move,
    And,
    BitClear,
    Or,
    ExclusiveOr,
    AddInteger { lane_bits: u8 },
    SubtractInteger { lane_bits: u8 },
    AddF32,
    SubtractF32,
    MultiplyF32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VectorDataProcessing {
    pub operation: VectorOperation,
    pub size: VectorSize,
    pub vd: u8,
    pub vn: u8,
    pub vm: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VectorTransfer {
    pub load: bool,
    pub rn: u8,
    pub vd: u8,
    pub register_count: u8,
    pub writeback_rm: Option<u8>,
}

#[must_use]
pub const fn decode_immediate_shift(type_bits: u8, immediate: u8) -> Shift {
    use A32ShiftKind::{
        ArithmeticRight, LogicalLeft, LogicalRight, RotateRight, RotateRightExtended,
    };
    let (kind, amount) = match (type_bits, immediate) {
        (0, value) => (LogicalLeft, value),
        (1, 0) => (LogicalRight, 32),
        (1, value) => (LogicalRight, value),
        (2, 0) => (ArithmeticRight, 32),
        (2, value) => (ArithmeticRight, value),
        (3, 0) => (RotateRightExtended, 1),
        (3, value) => (RotateRight, value),
        _ => unreachable!(),
    };
    Shift {
        kind,
        amount: ShiftAmount::Immediate(amount),
    }
}
