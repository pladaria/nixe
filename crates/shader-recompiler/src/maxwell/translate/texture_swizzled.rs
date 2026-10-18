use super::{Result, Value, Visitor, unsupported};
use crate::{
    ir::{FpControl, Reg, TextureInstInfo, Type},
    types::TextureType,
};

impl Visitor<'_> {
    fn texture_pair_alignment(&self, reg: Reg) -> Result<()> {
        if !reg.is_aligned(2) {
            return Err(unsupported("Unaligned texture register pair"));
        }
        Ok(())
    }
    fn texture_register_vector(&mut self, regs: &[Reg], float: bool) -> Result<Value> {
        let mut values = [Value::None; 4];
        for (i, &reg) in regs.iter().enumerate() {
            values[i] = if float {
                self.read_float_reg(reg)?
            } else {
                self.emit.get_reg(reg)?
            };
        }
        self.vector_or_scalar(&values[..regs.len()])
    }
    fn texture_layer(&mut self, reg: Reg, float: bool) -> Result<Value> {
        let value = self.emit.get_reg(reg)?;
        let value = self
            .emit
            .bit_field_extract(value, Value::U32(0), Value::U32(16), false)?;
        if float {
            self.emit
                .convert_i_to_f(32, 16, false, value, FpControl::default())
        } else {
            Ok(value)
        }
    }
    fn swizzled_result(&mut self, sample: Value, half: bool, gather: bool) -> Result<()> {
        let a = self.instruction.reg0();
        let b = Reg(self.field::<28, 8>() as u8);
        let swizzle = self.field::<50, 3>() as usize;
        let mask = if gather {
            15u32
        } else if b == Reg::RZ {
            [1, 2, 4, 8, 3, 9, 10, 12][swizzle]
        } else {
            *[7, 11, 13, 14, 15]
                .get(swizzle)
                .ok_or_else(|| unsupported("Dual-destination texture swizzle"))?
        };
        let shadow = self.emit.ir.value_type(sample)? == Type::F32;
        let mut values = [Value::F32(0); 4];
        let mut index = 0;
        for component in 0..4 {
            if mask & (1 << component) == 0 {
                continue;
            }
            if !half && index % 2 == 1 {
                self.texture_pair_alignment(if index < 2 { a } else { b })?;
            }
            let value = if shadow {
                if component == 3 {
                    Value::F32(1.0f32.to_bits())
                } else {
                    sample
                }
            } else {
                self.emit.composite_extract(sample, component)?
            };
            if half {
                values[index] = value;
            } else {
                let dest = if index < 2 {
                    a.offset(index as i32)?
                } else {
                    b.offset((index - 2) as i32)?
                };
                self.write_float_reg(dest, value)?;
            }
            index += 1;
        }
        if half {
            let pair = self.emit.composite_construct(&values[..2])?;
            let packed = self.emit.pack_half2x16(pair)?;
            self.emit.set_reg(a, packed)?;
            if index > 2 {
                let pair = self.emit.composite_construct(&values[2..])?;
                let packed = self.emit.pack_half2x16(pair)?;
                self.emit.set_reg(b, packed)?;
            }
        }
        Ok(())
    }
    pub(super) fn texs(&mut self) -> Result<()> {
        let encoding = self.field::<53, 4>();
        if encoding > 13 {
            return Err(unsupported("Swizzled texture sample encoding"));
        }
        let a = self.instruction.reg8();
        let b = self.instruction.reg20();
        if encoding >= 3 {
            self.texture_pair_alignment(a)?;
        }
        if matches!(encoding, 5 | 9 | 13) {
            self.texture_pair_alignment(b)?;
        }
        let kind = match encoding {
            0 => TextureType::Color1D,
            1..=6 => TextureType::Color2D,
            7..=9 => TextureType::ColorArray2D,
            10 | 11 => TextureType::Color3D,
            _ => TextureType::ColorCube,
        };
        let coords = match encoding {
            0 => self.read_float_reg(a)?,
            1 | 2 => self.texture_register_vector(&[a, b], true)?,
            3..=6 => self.texture_register_vector(&[a, a.offset(1)?], true)?,
            7..=9 => {
                let x = self.read_float_reg(a.offset(1)?)?;
                let y = self.read_float_reg(b)?;
                let layer = self.texture_layer(a, true)?;
                self.emit.composite_construct(&[x, y, layer])?
            }
            _ => self.texture_register_vector(&[a, a.offset(1)?, b], true)?,
        };
        let depth = matches!(encoding, 4..=6 | 9);
        let explicit = matches!(encoding, 0 | 2 | 3 | 5 | 6 | 8 | 9 | 11 | 13);
        let dref = if depth {
            self.read_float_reg(if matches!(encoding, 5 | 9) {
                b.offset(1)?
            } else {
                b
            })?
        } else {
            Value::None
        };
        let lod = match encoding {
            3 | 5 => self.read_float_reg(b)?,
            13 => self.read_float_reg(b.offset(1)?)?,
            _ => Value::F32(0),
        };
        let half = !self.bit::<59>();
        let mut info = TextureInstInfo::default();
        info.set_texture_type(kind);
        info.set_is_depth(depth);
        info.set_relaxed_precision(half);
        let handle = Value::U32(self.field::<36, 13>() * 4);
        let sample = match (depth, explicit) {
            (false, false) => self.emit.image_sample_implicit_lod(
                handle,
                coords,
                Value::None,
                Value::None,
                Value::None,
                info,
            )?,
            (false, true) => {
                self.emit
                    .image_sample_explicit_lod(handle, coords, lod, Value::None, info)?
            }
            (true, false) => self.emit.image_sample_dref_implicit_lod(
                handle,
                coords,
                dref,
                Value::None,
                Value::None,
                Value::None,
                info,
            )?,
            (true, true) => self.emit.image_sample_dref_explicit_lod(
                handle,
                coords,
                dref,
                lod,
                Value::None,
                info,
            )?,
        };
        self.swizzled_result(sample, half, false)
    }
    pub(super) fn tlds(&mut self) -> Result<()> {
        let encoding = self.field::<53, 4>();
        let a = self.instruction.reg8();
        let b = self.instruction.reg20();
        let kind = match encoding {
            0 | 1 => TextureType::Color1D,
            2 | 4 | 5 | 6 | 12 => TextureType::Color2D,
            7 => TextureType::Color3D,
            8 => TextureType::ColorArray2D,
            _ => return Err(unsupported("Swizzled texture fetch encoding")),
        };
        if matches!(encoding, 4..=7 | 12) {
            self.texture_pair_alignment(a)?;
        }
        if matches!(encoding, 8 | 12) {
            self.texture_pair_alignment(b)?;
        }
        let coords = match encoding {
            0 | 1 => self.emit.get_reg(a)?,
            2 => self.texture_register_vector(&[a, b], false)?,
            4 | 5 | 6 | 12 => self.texture_register_vector(&[a, a.offset(1)?], false)?,
            7 => self.texture_register_vector(&[a, a.offset(1)?, b], false)?,
            8 => {
                let layer = self.texture_layer(a, false)?;
                let x = self.emit.get_reg(b)?;
                let y = self.emit.get_reg(b.offset(1)?)?;
                self.emit.composite_construct(&[x, y, layer])?
            }
            _ => unreachable!(),
        };
        let mut offset = Value::None;
        let mut lod = Value::U32(0);
        let mut multisample = Value::None;
        match encoding {
            1 | 5 => lod = self.emit.get_reg(b)?,
            4 => {
                let packed = self.emit.get_reg(b)?;
                offset = self.offset_fields(packed, 2, 0, 4, 4)?;
            }
            6 => multisample = self.emit.get_reg(b)?,
            12 => {
                lod = self.emit.get_reg(b)?;
                let packed = self.emit.get_reg(b.offset(1)?)?;
                offset = self.offset_fields(packed, 2, 0, 4, 4)?;
            }
            _ => (),
        }
        let half = !self.bit::<59>();
        let mut info = TextureInstInfo::default();
        info.set_texture_type(kind);
        info.set_relaxed_precision(half);
        let handle = Value::U32(self.field::<36, 13>() * 4);
        let sample = self
            .emit
            .image_fetch(handle, coords, offset, lod, multisample, info)?;
        self.swizzled_result(sample, half, false)
    }
    pub(super) fn tld4s(&mut self) -> Result<()> {
        let a = self.instruction.reg8();
        let b = self.instruction.reg20();
        let has_offset = self.bit::<51>();
        let depth = self.bit::<50>();
        let half = self.bit::<55>();
        if has_offset || depth {
            self.texture_pair_alignment(a)?;
        }
        let coords = self.texture_register_vector(
            &[a, if has_offset || depth { a.offset(1)? } else { b }],
            true,
        )?;
        let offset = if has_offset {
            let packed = self.emit.get_reg(b)?;
            self.offset_fields(packed, 2, 0, 8, 6)?
        } else {
            Value::None
        };
        let dref = if depth {
            if has_offset {
                self.texture_pair_alignment(b)?;
            }
            self.read_float_reg(if has_offset { b.offset(1)? } else { b })?
        } else {
            Value::None
        };
        let mut info = TextureInstInfo::default();
        info.set_texture_type(TextureType::Color2D);
        info.set_is_depth(depth);
        info.set_relaxed_precision(half);
        info.set_gather_component(self.field::<52, 2>());
        let handle = Value::U32(self.field::<36, 13>() * 4);
        let sample = if depth {
            self.emit
                .image_gather_dref(handle, coords, offset, Value::None, dref, info)?
        } else {
            self.emit
                .image_gather(handle, coords, offset, Value::None, info)?
        };
        self.swizzled_result(sample, half, true)
    }
}
