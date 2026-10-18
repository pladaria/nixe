use super::instruction::Instruction;
use crate::{
    CompileError, Result,
    diagnostics::FailureCategory,
    ir::{Emitter, Reg, Type, Value},
};

/// Common operand assembly shared by instruction visitors. Guest addresses and
/// texture handles remain symbolic IR values at this stage.
pub struct Operands<'a, 'ir> {
    pub emit: &'a mut Emitter<'ir>,
    pub instruction: Instruction,
}

impl Operands<'_, '_> {
    pub fn read_u32(&mut self, reg: Reg) -> Result<Value> {
        self.emit.get_reg(reg)
    }
    pub fn read_f32(&mut self, reg: Reg) -> Result<Value> {
        let bits = self.read_u32(reg)?;
        self.emit.bit_cast(Type::F32, bits)
    }
    fn pair(&mut self, reg: Reg) -> Result<Value> {
        if !reg.is_aligned(2) {
            return Err(CompileError::new(
                FailureCategory::Unsupported,
                format!("Unaligned source register R{}", reg.0),
            ));
        }
        let low = self.read_u32(reg)?;
        let high = self.read_u32(reg.offset(1)?)?;
        self.emit.composite_construct(&[low, high])
    }
    pub fn read_u64(&mut self, reg: Reg) -> Result<Value> {
        let pair = self.pair(reg)?;
        self.emit.pack_uint2x32(pair)
    }
    pub fn read_f64(&mut self, reg: Reg) -> Result<Value> {
        let pair = self.pair(reg)?;
        self.emit.pack_double2x32(pair)
    }
    pub fn write_u32(&mut self, reg: Reg, value: Value) -> Result<()> {
        self.emit.set_reg(reg, value)?;
        Ok(())
    }
    pub fn write_f32(&mut self, reg: Reg, value: Value) -> Result<()> {
        let bits = self.emit.bit_cast(Type::U32, value)?;
        self.write_u32(reg, bits)
    }
    pub fn write_pair(&mut self, reg: Reg, value: Value) -> Result<()> {
        if !reg.is_aligned(2) {
            return Err(CompileError::new(
                FailureCategory::Unsupported,
                format!("Unaligned destination register R{}", reg.0),
            ));
        }
        let pair = match self.emit.ir.value_type(value)? {
            Type::U64 => self.emit.unpack_uint2x32(value)?,
            Type::F64 => self.emit.unpack_double2x32(value)?,
            _ => {
                return Err(CompileError::new(
                    FailureCategory::InvalidArgument,
                    "Paired destination requires U64 or F64",
                ));
            }
        };
        for index in 0..2 {
            let part = self.emit.composite_extract(pair, index)?;
            self.write_u32(reg.offset(index as i32)?, part)?;
        }
        Ok(())
    }

    pub fn reg8(&mut self) -> Result<Value> {
        self.read_u32(self.instruction.reg8())
    }
    pub fn reg20(&mut self) -> Result<Value> {
        self.read_u32(self.instruction.reg20())
    }
    pub fn reg39(&mut self) -> Result<Value> {
        self.read_u32(self.instruction.reg39())
    }
    pub fn float_reg8(&mut self) -> Result<Value> {
        self.read_f32(self.instruction.reg8())
    }
    pub fn float_reg20(&mut self) -> Result<Value> {
        self.read_f32(self.instruction.reg20())
    }
    pub fn float_reg39(&mut self) -> Result<Value> {
        self.read_f32(self.instruction.reg39())
    }
    pub fn double_reg20(&mut self) -> Result<Value> {
        self.read_f64(self.instruction.reg20())
    }
    pub fn double_reg39(&mut self) -> Result<Value> {
        self.read_f64(self.instruction.reg39())
    }

    pub fn cbuf(&mut self) -> Result<Value> {
        let (bank, offset) = self.instruction.cbuf_address()?;
        self.emit
            .get_cbuf(Value::U32(bank.0), Value::U32(offset.0), 32, false)
    }
    pub fn float_cbuf(&mut self) -> Result<Value> {
        let (bank, offset) = self.instruction.cbuf_address()?;
        self.emit
            .get_float_cbuf(Value::U32(bank.0), Value::U32(offset.0))
    }
    pub fn double_cbuf(&mut self) -> Result<Value> {
        let (bank, offset) = self.instruction.cbuf_address()?;
        let unaligned = self.instruction.cbuf_is_unaligned();
        let high_offset = if unaligned {
            offset.0 | 4
        } else {
            (offset.0 & !7) | 4
        };
        let high = self
            .emit
            .get_cbuf(Value::U32(bank.0), Value::U32(high_offset), 32, false)?;
        let low = if unaligned {
            Value::U32(0)
        } else {
            self.emit
                .get_cbuf(Value::U32(bank.0), Value::U32(offset.0), 32, false)?
        };
        let pair = self.emit.composite_construct(&[low, high])?;
        self.emit.pack_double2x32(pair)
    }
    pub fn packed_cbuf(&mut self) -> Result<Value> {
        if self.instruction.cbuf_is_unaligned() {
            return Err(CompileError::new(
                FailureCategory::Unsupported,
                "Unaligned packed constant buffer read",
            ));
        }
        let (bank, offset) = self.instruction.cbuf_address()?;
        let low = self
            .emit
            .get_cbuf(Value::U32(bank.0), Value::U32(offset.0), 32, false)?;
        let high = self
            .emit
            .get_cbuf(Value::U32(bank.0), Value::U32(offset.0 + 4), 32, false)?;
        let pair = self.emit.composite_construct(&[low, high])?;
        self.emit.pack_uint2x32(pair)
    }

    pub fn imm20(&self) -> Value {
        Value::U32(self.instruction.imm20())
    }
    pub fn float_imm20(&self) -> Value {
        Value::F32(self.instruction.float_imm20_bits())
    }
    pub fn double_imm20(&self) -> Value {
        Value::F64(self.instruction.double_imm20_bits())
    }
    pub fn packed_imm20(&self) -> Value {
        Value::U64(self.instruction.packed_imm20())
    }
    pub fn imm32(&self) -> Value {
        Value::U32(self.instruction.imm32())
    }
    pub fn float_imm32(&self) -> Value {
        Value::F32(self.instruction.imm32())
    }
}
