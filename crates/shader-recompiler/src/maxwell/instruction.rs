use super::bits::{extract, extract_signed};
use crate::{
    CompileError, Result,
    diagnostics::FailureCategory,
    ir::{Condition, FlowTest, Pred, Reg},
    types::{ByteOffset, CbufIndex},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instruction(pub u64);

impl Instruction {
    pub const fn from_le_bytes(bytes: [u8; 8]) -> Self {
        Self(u64::from_le_bytes(bytes))
    }
    pub fn predicate(self) -> (Pred, bool) {
        (
            Pred::new(extract::<16, 3>(self.0) as u8).unwrap(),
            extract::<19, 1>(self.0) != 0,
        )
    }
    pub fn condition(self) -> Condition {
        let (predicate, negated) = self.predicate();
        Condition {
            flow_test: self.flow_test(),
            predicate,
            negated,
        }
    }
    pub fn flow_test(self) -> FlowTest {
        FlowTest::try_from(extract::<0, 5>(self.0) as u8).unwrap()
    }
    pub const fn branch_is_cbuf(self) -> bool {
        extract::<5, 1>(self.0) != 0
    }
    pub const fn branch_absolute(self) -> u32 {
        extract::<20, 32>(self.0) as u32
    }
    pub const fn branch_offset(self) -> i32 {
        extract_signed::<20, 24>(self.0) as i32
    }
    pub const fn reg0(self) -> Reg {
        Reg(extract::<0, 8>(self.0) as u8)
    }
    pub const fn reg8(self) -> Reg {
        Reg(extract::<8, 8>(self.0) as u8)
    }
    pub const fn reg20(self) -> Reg {
        Reg(extract::<20, 8>(self.0) as u8)
    }
    pub const fn reg39(self) -> Reg {
        Reg(extract::<39, 8>(self.0) as u8)
    }
    pub const fn special_register_raw(self) -> u8 {
        extract::<20, 8>(self.0) as u8
    }

    pub fn special_register(self) -> Result<super::special_register::SpecialRegister> {
        super::special_register::SpecialRegister::try_from(self.special_register_raw())
    }

    pub fn cbuf_address(self) -> Result<(CbufIndex, ByteOffset)> {
        let bank = extract::<34, 5>(self.0) as u32;
        let word_offset = extract::<20, 14>(self.0) as u32;
        if bank >= 18 {
            return Err(CompileError::new(
                FailureCategory::Unsupported,
                format!("Constant buffer bank {bank} exceeds 17"),
            ));
        }
        Ok((CbufIndex(bank), ByteOffset(word_offset * 4)))
    }

    pub const fn cbuf_is_unaligned(self) -> bool {
        extract::<20, 1>(self.0) != 0
    }
    pub const fn imm20(self) -> u32 {
        let bits = extract::<20, 19>(self.0) as u32;
        if extract::<56, 1>(self.0) != 0 {
            bits | 0xfff8_0000
        } else {
            bits
        }
    }
    pub const fn float_imm20_bits(self) -> u32 {
        (extract::<20, 19>(self.0) as u32) << 12 | (extract::<56, 1>(self.0) as u32) << 31
    }
    pub const fn double_imm20_bits(self) -> u64 {
        extract::<20, 19>(self.0) << 44 | extract::<56, 1>(self.0) << 63
    }
    pub const fn packed_imm20(self) -> u64 {
        (self.imm20() as u64) << 32
    }
    pub const fn imm32(self) -> u32 {
        extract::<20, 32>(self.0) as u32
    }
}
