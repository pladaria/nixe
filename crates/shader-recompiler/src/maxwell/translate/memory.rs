use super::{Operands, Result, Value, Visitor, unsupported};
use crate::{ir::Reg, maxwell::bits::extract_signed};

pub(super) fn memory_size(size: u32) -> Result<(u32, bool)> {
    match size {
        0 => Ok((8, false)),
        1 => Ok((8, true)),
        2 => Ok((16, false)),
        3 => Ok((16, true)),
        4 => Ok((32, false)),
        5 => Ok((64, false)),
        6 => Ok((128, false)),
        _ => Err(unsupported("Invalid memory access size")),
    }
}
impl Visitor<'_> {
    fn local_offset(&mut self) -> Result<Value> {
        if self.instruction.reg8() == Reg::RZ {
            Ok(Value::U32(self.field::<20, 24>()))
        } else {
            let base = self.a()?;
            self.emit.i_add(
                base,
                Value::U32(extract_signed::<20, 24>(self.instruction.0) as u32),
            )
        }
    }
    pub(super) fn global_address(&mut self, atomic: bool) -> Result<Value> {
        let extended = if atomic {
            self.bit::<48>()
        } else {
            self.bit::<45>()
        };
        let address = if extended {
            Operands {
                emit: &mut self.emit,
                instruction: self.instruction,
            }
            .read_u64(self.instruction.reg8())?
        } else {
            let base = self.a()?;
            self.emit.u_convert(64, base)?
        };
        let absolute = self.instruction.reg8() == Reg::RZ;
        let offset = match (atomic, absolute) {
            (true, true) => self.field::<28, 20>() as u64,
            (false, true) => self.field::<20, 24>() as u64,
            (true, false) => extract_signed::<28, 20>(self.instruction.0) as u64,
            (false, false) => extract_signed::<20, 24>(self.instruction.0) as u64,
        };
        self.emit.i_add(address, Value::U64(offset))
    }
    fn local_word(&mut self, offset: Value) -> Result<Value> {
        if let Value::U32(offset) = offset {
            Ok(Value::U32(offset / 4))
        } else {
            self.emit.shift_right_arithmetic(offset, Value::U32(2))
        }
    }
    fn local_bit(&mut self, offset: Value, bits: u32) -> Result<Value> {
        let shift = self.emit.shift_left_logical(offset, Value::U32(3))?;
        self.emit
            .bitwise_and(shift, Value::U32(if bits == 8 { 24 } else { 16 }))
    }
    fn load_local_checked(&mut self, word: Value, offset: Value) -> Result<Value> {
        let condition =
            self.emit
                .i_less_than(offset, Value::U32(self.metadata.local_memory_size), false)?;
        let value = self.emit.load_local(word)?;
        self.emit.select(condition, value, Value::U32(0))
    }
    fn check_register_alignment(&self, reg: Reg, words: u32) -> Result<()> {
        if !reg.is_aligned(words as u8) {
            return Err(unsupported(format!(
                "Unaligned data register R{} for {} words",
                reg.0, words
            )));
        }
        Ok(())
    }
    fn read_vector(&mut self, reg: Reg, words: u32) -> Result<Value> {
        let mut values = [Value::None; 4];
        for (index, value) in values.iter_mut().enumerate().take(words as usize) {
            *value = self.emit.get_reg(reg.offset(index as i32)?)?;
        }
        self.emit.composite_construct(&values[..words as usize])
    }
    fn write_vector(&mut self, reg: Reg, vector: Value, words: u32) -> Result<()> {
        for index in 0..words {
            let value = self.emit.composite_extract(vector, index as usize)?;
            self.emit.set_reg(reg.offset(index as i32)?, value)?;
        }
        Ok(())
    }
    pub(super) fn ldc(&mut self) -> Result<()> {
        let mut index = Value::U32(self.field::<36, 5>());
        let reg = self.a()?;
        let imm = Value::U32(extract_signed::<20, 16>(self.instruction.0) as u32);
        let mut offset = self.emit.i_add(reg, imm)?;
        match self.field::<44, 2>() {
            0 => (),
            2 => {
                let segment =
                    self.emit
                        .bit_field_extract(offset, Value::U32(16), Value::U32(16), false)?;
                offset =
                    self.emit
                        .bit_field_extract(offset, Value::U32(0), Value::U32(16), false)?;
                index = self.emit.i_add(segment, index)?;
            }
            _ => return Err(unsupported("LDC address mode")),
        }
        let (bits, signed) = memory_size(self.field::<48, 3>())?;
        if bits == 128 {
            return Err(unsupported("LDC 128-bit access"));
        }
        if bits == 64 {
            self.check_register_alignment(self.instruction.reg0(), 2)?;
        }
        let value = self.emit.get_cbuf(index, offset, bits, signed)?;
        if bits == 64 {
            self.write_vector(self.instruction.reg0(), value, 2)
        } else {
            self.write(value)
        }
    }
    pub(super) fn local_memory(&mut self, store: bool) -> Result<()> {
        let offset = self.local_offset()?;
        let word = self.local_word(offset)?;
        if store && matches!(offset,Value::U32(offset) if offset>=self.metadata.local_memory_size) {
            // Immediate stores beyond the local allocation have no effect.
            return Ok(());
        }
        let reg = self.instruction.reg0();
        let first = if store {
            self.emit.get_reg(reg)?
        } else {
            self.load_local_checked(word, offset)?
        };
        let (bits, signed) = memory_size(self.field::<48, 3>())?;
        if bits < 32 {
            let bit = self.local_bit(offset, bits)?;
            if store {
                let old = self.emit.load_local(word)?;
                let value = self
                    .emit
                    .bit_field_insert(old, first, bit, Value::U32(bits))?;
                self.emit.write_local(word, value)?;
            } else {
                let value = self
                    .emit
                    .bit_field_extract(first, bit, Value::U32(bits), signed)?;
                self.write(value)?;
            }
        } else {
            self.check_register_alignment(reg, bits / 32)?;
            if store {
                self.emit.write_local(word, first)?;
            } else {
                self.write(first)?;
            }
            for index in 1..bits / 32 {
                let subword = self.emit.i_add(word, Value::U32(index))?;
                if store {
                    let value = self.emit.get_reg(reg.offset(index as i32)?)?;
                    self.emit.write_local(subword, value)?;
                } else {
                    let suboffset = self.emit.i_add(offset, Value::U32(index * 4))?;
                    let value = self.load_local_checked(subword, suboffset)?;
                    self.emit.set_reg(reg.offset(index as i32)?, value)?;
                }
            }
        }
        Ok(())
    }
    pub(super) fn shared_memory(&mut self, store: bool) -> Result<()> {
        let offset = self.local_offset()?;
        let reg = self.instruction.reg0();
        let (bits, signed) = memory_size(self.field::<48, 3>())?;
        if store {
            let value = if bits <= 32 {
                self.emit.get_reg(reg)?
            } else {
                // Four-word shared stores require only pair alignment.
                self.check_register_alignment(reg, 2)?;
                self.read_vector(reg, bits / 32)?
            };
            self.emit.write_shared(bits, offset, value)?;
        } else {
            let value = self.emit.load_shared(bits, signed, offset)?;
            if bits <= 32 {
                self.write(value)?;
            } else {
                self.check_register_alignment(reg, bits / 32)?;
                self.write_vector(reg, value, bits / 32)?;
            }
        }
        Ok(())
    }
    pub(super) fn global_memory(&mut self, store: bool) -> Result<()> {
        let address = self.global_address(false)?;
        let reg = self.instruction.reg0();
        let size = self.field::<48, 3>();
        if store && size == 7 {
            // The reserved store-size encoding has no memory operation.
            return Ok(());
        }
        let (bits, _) = if !store && size == 7 {
            (128, false)
        } else {
            memory_size(size)?
        };
        if bits > 32 {
            self.check_register_alignment(reg, bits / 32)?;
        }
        if store {
            let value = if bits <= 32 {
                self.emit.get_reg(reg)?
            } else {
                self.read_vector(reg, bits / 32)?
            };
            match size {
                0 => self.emit.write_global_u8(address, value)?,
                1 => self.emit.write_global_s8(address, value)?,
                2 => self.emit.write_global_u16(address, value)?,
                3 => self.emit.write_global_s16(address, value)?,
                4 => self.emit.write_global32(address, value)?,
                5 => self.emit.write_global64(address, value)?,
                6 => self.emit.write_global128(address, value)?,
                _ => unreachable!(),
            };
        } else {
            let value = match size {
                0 => self.emit.load_global_u8(address)?,
                1 => self.emit.load_global_s8(address)?,
                2 => self.emit.load_global_u16(address)?,
                3 => self.emit.load_global_s16(address)?,
                4 => self.emit.load_global32(address)?,
                5 => self.emit.load_global64(address)?,
                6 | 7 => self.emit.load_global128(address)?,
                _ => unreachable!(),
            };
            if bits <= 32 {
                self.write(value)?;
            } else {
                self.write_vector(reg, value, bits / 32)?;
            }
        }
        Ok(())
    }
}
