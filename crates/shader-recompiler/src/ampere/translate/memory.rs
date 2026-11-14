use super::*;

impl Visitor<'_> {
    pub(super) fn shared_memory(&mut self) -> Result<()> {
        let i = self.instruction;
        let kind = i.bits(73, 3);
        let stride = i.bits(78, 2);
        let store = i.opcode() != 0x984;
        let uniform = i.opcode() != 0x388 && i.bits(91, 1) != 0;
        let variable = if uniform {
            if store { 0x080000ff } else { 0x08000000 }
        } else {
            0
        };
        if kind > 6
            || i.bits(64, 32) & !variable != (kind << 9 | stride << 14)
            || (!uniform && (!store && i.bits(32, 8) != 0 || i.opcode() == 0x988))
        {
            return Err(unsupported("Native shared memory addressing mode"));
        }
        let (bits, signed) = match kind {
            0 => (8, false),
            1 => (8, true),
            2 => (16, false),
            3 => (16, true),
            _ => (32 << (kind - 4), false),
        };
        let words = (bits / 32).max(1);
        let data = i.bits(if store { 32 } else { 16 }, 8);
        let alignment = if store { words.min(2) } else { words };
        if data != 255 && (!data.is_multiple_of(alignment) || data + words > 255) {
            return Err(unsupported("Native shared memory register alignment"));
        }
        let mut address = self.reg(i.bits(24, 8))?;
        if stride != 0 {
            address = self
                .emit
                .shift_left_logical(address, Value::U32(stride + 1))?;
        }
        if uniform {
            let base = self.ureg(i.bits(if store { 64 } else { 32 }, 8))?;
            address = self.emit.i_add(address, base)?;
        }
        let offset = ((i.bits(40, 24) as i32) << 8 >> 8) as u32;
        address = self.emit.i_add(address, Value::U32(offset))?;
        if store {
            let mut values = [Value::None; 4];
            for (word, value) in values.iter_mut().enumerate().take(words as usize) {
                *value = self.reg(if data == 255 {
                    data
                } else {
                    data + word as u32
                })?;
            }
            let value = if words == 1 {
                values[0]
            } else {
                self.emit.composite_construct(&values[..words as usize])?
            };
            self.emit.write_shared(bits, address, value)?;
        } else {
            let value = self.emit.load_shared(bits, signed, address)?;
            for word in 0..words {
                let component = if words == 1 {
                    value
                } else {
                    self.emit.composite_extract(value, word as usize)?
                };
                self.emit.set_reg(
                    Reg(if data == 255 {
                        255
                    } else {
                        (data + word) as u8
                    }),
                    component,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn global_atomic(&mut self) -> Result<()> {
        let i = self.instruction;
        let uniform = i.opcode() == 0x9a8;
        let operation = i.bits(87, 4);
        if if uniform {
            i.bits(24, 8) != 255 || i.bits(64, 32) & !0x3f != 0x081ee140
        } else {
            i.bits(64, 32) & !0x07800000 != 0x001ee100 || operation > 8
        } {
            return Err(unsupported("Native global atomic mode"));
        }
        let register = if uniform {
            i.bits(64, 6)
        } else {
            i.bits(24, 8)
        };
        let base = self.address_pair(register, uniform)?;
        let offset = ((i.bits(40, 24) as i32) << 8 >> 8) as u64;
        let address = self.emit.i_add(base, Value::U64(offset))?;
        let value = self.reg(i.bits(32, 8))?;
        let result = self.integer_atomic(address, value, operation, false)?;
        self.write(result)
    }

    fn integer_atomic(
        &mut self,
        address: Value,
        value: Value,
        operation: u32,
        signed: bool,
    ) -> Result<Value> {
        match operation {
            0 => self.emit.global_atomic_i_add(address, value),
            1 => self.emit.global_atomic_i_min(address, value, signed),
            2 => self.emit.global_atomic_i_max(address, value, signed),
            3 => self.emit.global_atomic_inc(address, value),
            4 => self.emit.global_atomic_dec(address, value),
            5 => self.emit.global_atomic_and(address, value),
            6 => self.emit.global_atomic_or(address, value),
            7 => self.emit.global_atomic_xor(address, value),
            8 => self.emit.global_atomic_exchange(address, value),
            _ => Err(unsupported("Native integer atomic operation")),
        }
    }
    pub(super) fn global_reduction(&mut self) -> Result<()> {
        let i = self.instruction;
        let kind = i.bits(73, 3);
        let operation = i.bits(87, 3);
        let uniform = i.bits(24, 8) == 255;
        let variable = 0x03800e00 | if uniform { 0x3f } else { 0 };
        let mode = if uniform { 0x0c10e100 } else { 0x0010e100 };
        // RED.E.STRONG.GPU carries no result register. Its global operation
        // remains atomic when lowered to a physical storage-buffer binding.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L1327-L1349
        if i.bits(64, 32) & !variable != mode
            || !matches!(kind, 0 | 1 | 3)
            || (kind == 3 && operation != 0)
            || matches!(operation, 3 | 4)
        {
            return Err(unsupported("Native global reduction format/operation"));
        }
        let base = self.address_pair(
            if uniform {
                i.bits(64, 6)
            } else {
                i.bits(24, 8)
            },
            uniform,
        )?;
        let offset = ((i.bits(40, 24) as i32) << 8 >> 8) as u64;
        let address = self.emit.i_add(base, Value::U64(offset))?;
        let value = self.reg(i.bits(32, 8))?;
        if kind == 3 {
            let value = self.emit.bit_cast(Type::F32, value)?;
            self.emit.global_atomic_f32_add(
                address,
                value,
                FpControl {
                    no_contraction: false,
                    rounding: FpRounding::RN,
                    fmz_mode: FmzMode::FTZ,
                },
            )?;
        } else {
            self.integer_atomic(address, value, operation, kind == 1)?;
        }
        Ok(())
    }

    fn address_words(&mut self, register: u32, uniform: bool) -> Result<[Value; 2]> {
        if register == if uniform { 63 } else { 255 } {
            return Ok([Value::U32(0); 2]);
        }
        if register & 1 != 0 || register >= if uniform { 62 } else { 254 } {
            return Err(unsupported("Native memory address register pair"));
        }
        let low = self.register(register, uniform)?;
        let high = self.register(register + 1, uniform)?;
        Ok([low, high])
    }
    fn address_pair(&mut self, register: u32, uniform: bool) -> Result<Value> {
        let words = self.address_words(register, uniform)?;
        let pair = self.emit.composite_construct(&words)?;
        self.emit.pack_uint2x32(pair)
    }

    fn uniform_address(&mut self, base: u32, offset: u32, wide: bool) -> Result<Value> {
        let base = self.address_words(base, true)?;
        let offset = if wide {
            self.address_words(offset, false)?
        } else {
            [self.reg(offset)?, Value::U32(0)]
        };
        // Keep both halves and the carry explicit. This also preserves pointer
        // provenance for storage-buffer rebasing without host 64-bit arithmetic.
        let low = self.emit.i_add(base[0], offset[0])?;
        let carry = self.emit.get_carry_from_op(low)?;
        let carry = self.emit.select(carry, Value::U32(1), Value::U32(0))?;
        let high = self.emit.i_add(base[1], offset[1])?;
        let high = self.emit.i_add(high, carry)?;
        let pair = self.emit.composite_construct(&[low, high])?;
        self.emit.pack_uint2x32(pair)
    }
    pub(super) fn global_store(&mut self) -> Result<()> {
        let i = self.instruction;
        let size = i.bits(73, 3);
        let words = if size <= 4 { 1 } else { 1 << (size - 4) };
        let data = i.bits(32, 8);
        let uniform = i.opcode() == 0x986;
        if size > 6
            || (data != 255 && (!data.is_multiple_of(words) || data + words > 255))
            || i.bits(64, 32) & !(if uniform { 0x04000e3f } else { 0xe00 })
                != if uniform { 0x0810a100 } else { 0x0010a100 }
        {
            return Err(unsupported("Native global store size/addressing mode"));
        }
        let base = if uniform {
            self.uniform_address(i.bits(64, 6), i.bits(24, 8), i.bits(90, 1) != 0)?
        } else {
            self.address_pair(i.bits(24, 8), false)?
        };
        let immediate = ((i.bits(40, 24) as i32) << 8 >> 8) as u64;
        let address = self.emit.i_add(base, Value::U64(immediate))?;
        let mut values = [Value::U32(0); 4];
        for (word, value) in values[..words as usize].iter_mut().enumerate() {
            *value = self.reg(if data == 255 { 255 } else { data + word as u32 })?;
        }
        let value = if words == 1 {
            values[0]
        } else {
            self.emit.composite_construct(&values[..words as usize])?
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
        Ok(())
    }
    pub(super) fn global_load(&mut self) -> Result<()> {
        let i = self.instruction;
        let size = i.bits(73, 3);
        let address_reg = Reg(i.bits(24, 8) as u8);
        let dest = Reg(i.bits(16, 8) as u8);
        let words = if size <= 4 { 1 } else { 1 << (size - 4) };
        let uniform = i.opcode() == 0x981;
        if size > 6
            || (!uniform && (i.bits(32, 8) != 0 || i.bits(64, 32) & !0x2e00 != 0x001e8100))
            || (uniform && (i.bits(38, 2) != 0 || i.bits(64, 32) & !0x04002e00 != 0x081e8100))
            || (dest != Reg::RZ
                && (u32::from(dest.0) % words != 0 || u32::from(dest.0) + words > 255))
        {
            return Err(unsupported(
                "Native global load format/address/predicate mode",
            ));
        }
        let base = if uniform {
            // The uniform pair is the base; Ra adds a 32- or 64-bit offset.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L2018-L2058
            self.uniform_address(i.bits(32, 6), u32::from(address_reg.0), i.bits(90, 1) != 0)?
        } else {
            self.address_pair(u32::from(address_reg.0), false)?
        };
        let offset = ((i.bits(40, 24) as i32) << 8 >> 8) as u64;
        let address = self.emit.i_add(base, Value::U64(offset))?;
        let value = match size {
            0 => self.emit.load_global_u8(address)?,
            1 => self.emit.load_global_s8(address)?,
            2 => self.emit.load_global_u16(address)?,
            3 => self.emit.load_global_s16(address)?,
            4 => self.emit.load_global32(address)?,
            5 => self.emit.load_global64(address)?,
            6 => self.emit.load_global128(address)?,
            _ => unreachable!(),
        };
        if words == 1 {
            return self.write(value);
        }
        for word in 0..words {
            let component = self.emit.composite_extract(value, word as usize)?;
            let register = if dest == Reg::RZ {
                dest
            } else {
                dest.offset(word as i32)?
            };
            self.emit.set_reg(register, component)?;
        }
        Ok(())
    }
}
