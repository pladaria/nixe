//! Declarative A64 instruction decoding through one platform-bound table.

pub mod a64;
mod allocation;
pub mod table;

use core::fmt;

use crate::{
    location::{DecodedInstruction, InstructionEncoding, LocationDescriptor},
    platform::PlatformDecoder,
};

pub use table::{
    DecodeSupport, DecodedOpcode, DecodedOperands, InstructionPattern, OperandField, OperandId,
    OperandKind, OperandValue, RegisterClass,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodeResult {
    Decoded(DecodedInstruction<DecodedOpcode>),
    Unallocated {
        instruction: crate::error::InstructionDiagnostic,
        reason: &'static str,
    },
    Reserved {
        instruction: crate::error::InstructionDiagnostic,
        name: &'static str,
        reason: &'static str,
    },
    RecognizedUnimplemented(DecodedInstruction<DecodedOpcode>),
}

#[must_use]
pub fn decode(
    decoder: impl Into<PlatformDecoder>,
    location: LocationDescriptor,
    encoding: InstructionEncoding,
) -> DecodeResult {
    a64::decode(decoder, location, encoding)
}

pub struct Disassembly<'a>(&'a DecodedOpcode);

impl fmt::Display for Disassembly<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_disassembly(formatter)
    }
}

#[must_use]
pub const fn disassemble(opcode: &DecodedOpcode) -> Disassembly<'_> {
    Disassembly(opcode)
}
