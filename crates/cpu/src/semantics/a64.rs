//! Pure architectural decisions shared by A64 execution engines.

use crate::{memory::MemoryAccessSize, semantics::shifts::ShiftKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadSpec {
    pub signed: bool,
    pub destination_bits: u8,
}

impl LoadSpec {
    #[must_use]
    pub const fn unsigned(size: MemoryAccessSize) -> Self {
        Self {
            signed: false,
            destination_bits: if matches!(size, MemoryAccessSize::Doubleword) {
                64
            } else {
                32
            },
        }
    }

    const fn signed(destination_bits: u8) -> Self {
        Self {
            signed: true,
            destination_bits,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarTransfer {
    Store,
    Load(LoadSpec),
}

#[must_use]
pub const fn memory_size(size_bits: u8) -> MemoryAccessSize {
    match size_bits {
        0 => MemoryAccessSize::Byte,
        1 => MemoryAccessSize::Halfword,
        2 => MemoryAccessSize::Word,
        3 => MemoryAccessSize::Doubleword,
        _ => panic!("A64 memory size is a two-bit field"),
    }
}

#[must_use]
pub const fn scalar_transfer(opc: u8, size: MemoryAccessSize) -> Option<ScalarTransfer> {
    match (opc, size) {
        (0, _) => Some(ScalarTransfer::Store),
        (1, _) => Some(ScalarTransfer::Load(LoadSpec::unsigned(size))),
        (2, MemoryAccessSize::Byte | MemoryAccessSize::Halfword | MemoryAccessSize::Word) => {
            Some(ScalarTransfer::Load(LoadSpec::signed(64)))
        }
        (3, MemoryAccessSize::Byte | MemoryAccessSize::Halfword) => {
            Some(ScalarTransfer::Load(LoadSpec::signed(32)))
        }
        _ => None,
    }
}

#[must_use]
pub const fn literal_load(opc: u8) -> Option<(MemoryAccessSize, LoadSpec)> {
    match opc {
        0 => Some((
            MemoryAccessSize::Word,
            LoadSpec::unsigned(MemoryAccessSize::Word),
        )),
        1 => Some((
            MemoryAccessSize::Doubleword,
            LoadSpec::unsigned(MemoryAccessSize::Doubleword),
        )),
        2 => Some((MemoryAccessSize::Word, LoadSpec::signed(64))),
        _ => None,
    }
}

#[must_use]
pub const fn pair_transfer(size_bits: u8, load: bool) -> Option<(MemoryAccessSize, LoadSpec)> {
    match (size_bits, load) {
        (0, _) => Some((
            MemoryAccessSize::Word,
            LoadSpec::unsigned(MemoryAccessSize::Word),
        )),
        (1, true) => Some((MemoryAccessSize::Word, LoadSpec::signed(64))),
        (2, _) => Some((
            MemoryAccessSize::Doubleword,
            LoadSpec::unsigned(MemoryAccessSize::Doubleword),
        )),
        _ => None,
    }
}

#[must_use]
pub const fn shift_kind(encoded: u8, allow_rotate: bool) -> Option<ShiftKind> {
    match encoded {
        0 => Some(ShiftKind::LogicalLeft),
        1 => Some(ShiftKind::LogicalRight),
        2 => Some(ShiftKind::ArithmeticRight),
        3 if allow_rotate => Some(ShiftKind::RotateRight),
        _ => None,
    }
}

#[must_use]
pub const fn signed_immediate(value: u64, bits: u8) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_transfer_table_covers_legal_boundaries() {
        use MemoryAccessSize::{Doubleword, Halfword, Word};
        use ScalarTransfer::{Load, Store};
        let unsigned32 = Load(LoadSpec::unsigned(Word));
        let cases = [
            (0, Doubleword, Some(Store)),
            (1, Word, Some(unsigned32)),
            (2, Word, Some(Load(LoadSpec::signed(64)))),
            (2, Doubleword, None),
            (3, Halfword, Some(Load(LoadSpec::signed(32)))),
            (3, Word, None),
        ];
        for (opc, size, expected) in cases {
            assert_eq!(scalar_transfer(opc, size), expected);
        }
    }

    #[test]
    fn signed_immediates_and_shift_kinds_are_width_aware() {
        assert_eq!(signed_immediate(0x1ff, 9), -1);
        assert_eq!(signed_immediate(0x100, 9), -256);
        assert_eq!(shift_kind(3, false), None);
        assert_eq!(shift_kind(3, true), Some(ShiftKind::RotateRight));
    }
}
