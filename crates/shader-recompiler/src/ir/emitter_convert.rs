use super::*;
use crate::{CompileError, Result, diagnostics::FailureCategory};

impl Emitter<'_> {
    pub fn convert_f_to_i(&mut self, bits: u32, signed: bool, value: Value) -> Result<Value> {
        let opcode = match (bits, signed, self.ir.value_type(value)?) {
            (16, true, Type::F16) => Opcode::ConvertS16F16,
            (16, true, Type::F32) => Opcode::ConvertS16F32,
            (16, true, Type::F64) => Opcode::ConvertS16F64,
            (32, true, Type::F16) => Opcode::ConvertS32F16,
            (32, true, Type::F32) => Opcode::ConvertS32F32,
            (32, true, Type::F64) => Opcode::ConvertS32F64,
            (64, true, Type::F16) => Opcode::ConvertS64F16,
            (64, true, Type::F32) => Opcode::ConvertS64F32,
            (64, true, Type::F64) => Opcode::ConvertS64F64,
            (16, false, Type::F16) => Opcode::ConvertU16F16,
            (16, false, Type::F32) => Opcode::ConvertU16F32,
            (16, false, Type::F64) => Opcode::ConvertU16F64,
            (32, false, Type::F16) => Opcode::ConvertU32F16,
            (32, false, Type::F32) => Opcode::ConvertU32F32,
            (32, false, Type::F64) => Opcode::ConvertU32F64,
            (64, false, Type::F16) => Opcode::ConvertU64F16,
            (64, false, Type::F32) => Opcode::ConvertU64F32,
            (64, false, Type::F64) => Opcode::ConvertU64F64,
            _ => return Err(Self::invalid("Invalid float-to-integer conversion widths")),
        };
        self.emit(opcode, &[value], 0)
    }
    pub fn convert_i_to_f(
        &mut self,
        destination: u32,
        source: u32,
        signed: bool,
        value: Value,
        control: FpControl,
    ) -> Result<Value> {
        let opcode = match (destination, source, signed) {
            (16, 8, true) => Opcode::ConvertF16S8,
            (16, 16, true) => Opcode::ConvertF16S16,
            (16, 32, true) => Opcode::ConvertF16S32,
            (16, 64, true) => Opcode::ConvertF16S64,
            (32, 8, true) => Opcode::ConvertF32S8,
            (32, 16, true) => Opcode::ConvertF32S16,
            (32, 32, true) => Opcode::ConvertF32S32,
            (32, 64, true) => Opcode::ConvertF32S64,
            (64, 8, true) => Opcode::ConvertF64S8,
            (64, 16, true) => Opcode::ConvertF64S16,
            (64, 32, true) => Opcode::ConvertF64S32,
            (64, 64, true) => Opcode::ConvertF64S64,
            (16, 8, false) => Opcode::ConvertF16U8,
            (16, 16, false) => Opcode::ConvertF16U16,
            (16, 32, false) => Opcode::ConvertF16U32,
            (16, 64, false) => Opcode::ConvertF16U64,
            (32, 8, false) => Opcode::ConvertF32U8,
            (32, 16, false) => Opcode::ConvertF32U16,
            (32, 32, false) => Opcode::ConvertF32U32,
            (32, 64, false) => Opcode::ConvertF32U64,
            (64, 8, false) => Opcode::ConvertF64U8,
            (64, 16, false) => Opcode::ConvertF64U16,
            (64, 32, false) => Opcode::ConvertF64U32,
            (64, 64, false) => Opcode::ConvertF64U64,
            _ => return Err(Self::invalid("Invalid integer-to-float conversion widths")),
        };
        self.emit(opcode, &[value], control.bits())
    }
    pub fn u_convert(&mut self, bits: u32, value: Value) -> Result<Value> {
        let opcode = match (bits, self.ir.value_type(value)?) {
            (8, Type::U8) | (16, Type::U16) | (32, Type::U32) | (64, Type::U64) => {
                return Ok(value);
            }
            (8, Type::U32) => Opcode::ConvertU8U32,
            (16, Type::U32) => Opcode::ConvertU16U32,
            (32, Type::U8) => Opcode::ConvertU32U8,
            (32, Type::U16) => Opcode::ConvertU32U16,
            (32, Type::U64) => Opcode::ConvertU32U64,
            (64, Type::U32) => Opcode::ConvertU64U32,
            _ => {
                return Err(CompileError::new(
                    FailureCategory::Unsupported,
                    "Unsupported unsigned conversion",
                ));
            }
        };
        self.emit(opcode, &[value], 0)
    }
    pub fn s_convert(&mut self, bits: u32, value: Value) -> Result<Value> {
        let opcode = match (bits, self.ir.value_type(value)?) {
            (32, Type::U8) => Opcode::ConvertS32S8,
            (32, Type::U16) => Opcode::ConvertS32S16,
            _ => {
                return Err(CompileError::new(
                    FailureCategory::Unsupported,
                    "Unsupported signed conversion",
                ));
            }
        };
        self.emit(opcode, &[value], 0)
    }
    pub fn fp_convert(&mut self, bits: u32, value: Value, control: FpControl) -> Result<Value> {
        let opcode = match (bits, self.ir.value_type(value)?) {
            (16, Type::F16) | (32, Type::F32) | (64, Type::F64) => return Ok(value),
            (16, Type::F32) => Opcode::ConvertF16F32,
            (32, Type::F16) => Opcode::ConvertF32F16,
            (32, Type::F64) => Opcode::ConvertF32F64,
            (64, Type::F32) => Opcode::ConvertF64F32,
            (16, Type::F64) | (64, Type::F16) => {
                return Err(CompileError::new(
                    FailureCategory::Internal,
                    "Illegal F16/F64 conversion",
                ));
            }
            _ => {
                return Err(CompileError::new(
                    FailureCategory::Unsupported,
                    "Unsupported floating-point conversion",
                ));
            }
        };
        self.emit(opcode, &[value], control.bits())
    }
    pub fn get_cbuf(
        &mut self,
        binding: Value,
        offset: Value,
        bits: u32,
        signed: bool,
    ) -> Result<Value> {
        let opcode = match (bits, signed) {
            (8, false) => Opcode::GetCbufU8,
            (8, true) => Opcode::GetCbufS8,
            (16, false) => Opcode::GetCbufU16,
            (16, true) => Opcode::GetCbufS16,
            (32, _) => Opcode::GetCbufU32,
            (64, _) => Opcode::GetCbufU32x2,
            _ => return Err(Self::invalid("Invalid constant buffer access width")),
        };
        self.emit(opcode, &[binding, offset], 0)
    }
    pub fn load_shared(&mut self, bits: u32, signed: bool, offset: Value) -> Result<Value> {
        let opcode = match (bits, signed) {
            (8, false) => Opcode::LoadSharedU8,
            (8, true) => Opcode::LoadSharedS8,
            (16, false) => Opcode::LoadSharedU16,
            (16, true) => Opcode::LoadSharedS16,
            (32, _) => Opcode::LoadSharedU32,
            (64, _) => Opcode::LoadSharedU64,
            (128, _) => Opcode::LoadSharedU128,
            _ => return Err(Self::invalid("Invalid shared memory load width")),
        };
        self.emit(opcode, &[offset], 0)
    }
    pub fn write_shared(&mut self, bits: u32, offset: Value, value: Value) -> Result<Value> {
        let opcode = match bits {
            8 => Opcode::WriteSharedU8,
            16 => Opcode::WriteSharedU16,
            32 => Opcode::WriteSharedU32,
            64 => Opcode::WriteSharedU64,
            128 => Opcode::WriteSharedU128,
            _ => return Err(Self::invalid("Invalid shared memory store width")),
        };
        self.emit(opcode, &[offset, value], 0)
    }
}
