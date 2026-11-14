use super::*;
use crate::{ir::TextureInstInfo, types::TextureType};

impl Visitor<'_> {
    pub(super) fn texture_fetch(&mut self) -> Result<()> {
        let i = self.instruction;
        let uniform = i.opcode() == 0xf66;
        let bindless = i.opcode() == 0x367;
        let dimension = i.bits(61, 3);
        let array = dimension == 5;
        let level = i.bits(87, 3) == 3;
        let offsets = i.bits(76, 1) != 0;
        let half = i.bits(79, 1) != 0;
        if !matches!(dimension, 0 | 1 | 2 | 5)
            || i.bits(60, 1) == 0
            || i.bits(59, 1) != u32::from(bindless)
            || i.bits(72, 4) == 0
            || i.bits(76, 5) & !9 != 0
            || i.bits(81, 3) != 7
            || !matches!(i.bits(87, 3), 1 | 3)
            || i.bits(91, 1) != u32::from(uniform)
            || (uniform && i.bits(46, 13) != 0)
            || (bindless && i.bits(40, 19) != 0)
            || ((uniform || bindless) && (level || offsets || half))
            || (offsets && dimension != 1)
            || (level && (dimension == 2 || (half && array)))
        {
            return Err(unsupported("Native texel fetch mode"));
        }
        // TLD operands are integer coordinates (layer first for arrays), then
        // LOD/offset or a bindless handle, split evenly over Ra and Rb.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L2313-L2480
        let spatial = match dimension {
            0 => 1,
            2 => 3,
            _ => 2,
        };
        let coordinates = spatial + usize::from(array);
        let count = coordinates + usize::from(level) + usize::from(offsets) + usize::from(bindless);
        let in_a = count.div_ceil(2);
        let mut operands = [Value::None; 4];
        for (index, value) in operands.iter_mut().enumerate().take(count) {
            let (base, offset) = if index < in_a {
                (Reg(i.bits(24, 8) as u8), index)
            } else {
                (Reg(i.bits(32, 8) as u8), index - in_a)
            };
            *value = self.emit.get_reg(if base == Reg::RZ {
                base
            } else {
                base.offset(offset as i32)?
            })?;
        }
        let handle = if uniform {
            self.ureg(i.bits(40, 6))?
        } else if bindless {
            operands[coordinates]
        } else {
            self.emit.get_cbuf(
                Value::U32(i.bits(54, 5)),
                Value::U32(i.bits(40, 14) * 4),
                32,
                false,
            )?
        };
        let mut coords = [Value::None; 3];
        coords[..spatial].copy_from_slice(&operands[usize::from(array)..coordinates]);
        if array {
            coords[spatial] =
                self.emit
                    .bit_field_extract(operands[0], Value::U32(0), Value::U32(16), false)?;
        }
        let coords = if coordinates == 1 {
            coords[0]
        } else {
            self.emit.composite_construct(&coords[..coordinates])?
        };
        let lod = if level {
            operands[coordinates]
        } else {
            Value::U32(0)
        };
        let offset = if offsets {
            let packed = operands[coordinates + usize::from(level)];
            let x = self
                .emit
                .bit_field_extract(packed, Value::U32(0), Value::U32(4), true)?;
            let y = self
                .emit
                .bit_field_extract(packed, Value::U32(4), Value::U32(4), true)?;
            self.emit.composite_construct(&[x, y])?
        } else {
            Value::None
        };
        let mut info = TextureInstInfo::default();
        info.set_texture_type(match dimension {
            0 => TextureType::Color1D,
            1 => TextureType::Color2D,
            2 => TextureType::Color3D,
            5 => TextureType::ColorArray2D,
            _ => unreachable!(),
        });
        info.set_relaxed_precision(half);
        let value = self
            .emit
            .image_fetch(handle, coords, offset, lod, Value::None, info)?;
        self.texture_result(value, false, half)
    }

    pub(super) fn texture_lod(&mut self) -> Result<()> {
        let i = self.instruction;
        let uniform = i.opcode() == 0xf69;
        let bindless = i.opcode() == 0x36a;
        let mask = i.bits(72, 4);
        let dimension = i.bits(61, 3);
        if dimension > 3
            || mask == 0
            || mask & !3 != 0
            || i.bits(76, 14) != 0
            || i.bits(59, 1) != u32::from(bindless)
            || i.bits(91, 1) != u32::from(uniform)
            || (uniform && i.bits(46, 13) != 0)
            || (bindless && i.bits(40, 19) != 0)
        {
            return Err(unsupported("Native texture LOD query mode"));
        }
        // Coordinates are packed in Ra when Rb is RZ or contains the handle;
        // otherwise they split between the two source ranges as for TEX.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L2495-L2535
        let count = [1usize, 2, 3, 3][dimension as usize];
        let a = Reg(i.bits(24, 8) as u8);
        let b = Reg(i.bits(32, 8) as u8);
        let in_a = if b == Reg::RZ || bindless {
            count
        } else {
            count.div_ceil(2)
        };
        if u32::from(a.0) + in_a as u32 > 255
            || (in_a < count && u32::from(b.0) + (count - in_a) as u32 > 255)
        {
            return Err(unsupported("Native texture LOD coordinate registers"));
        }
        let mut coords = [Value::None; 3];
        for (index, coord) in coords.iter_mut().enumerate().take(count) {
            let source = if index < in_a {
                a.offset(index as i32)?
            } else {
                b.offset((index - in_a) as i32)?
            };
            let value = self.emit.get_reg(source)?;
            *coord = self.emit.bit_cast(Type::F32, value)?;
        }
        let coords = if count == 1 {
            coords[0]
        } else {
            self.emit.composite_construct(&coords[..count])?
        };
        let handle = if uniform {
            self.ureg(i.bits(40, 6))?
        } else if bindless {
            self.emit.get_reg(b)?
        } else {
            self.emit.get_cbuf(
                Value::U32(i.bits(54, 5)),
                Value::U32(i.bits(40, 14) * 4),
                32,
                false,
            )?
        };
        let mut info = TextureInstInfo::default();
        info.set_texture_type(
            [
                TextureType::Color1D,
                TextureType::Color2D,
                TextureType::Color3D,
                TextureType::ColorCube,
            ][dimension as usize],
        );
        let query = self.emit.image_query_lod(handle, coords, info)?;
        let mut dest = Reg(i.bits(16, 8) as u8);
        for component in 0..2 {
            if mask & (1 << component) == 0 {
                continue;
            }
            let value = crate::frontend::texture_lod_component(&mut self.emit, query, component)?;
            self.emit.set_reg(dest, value)?;
            if dest != Reg::RZ {
                dest = dest.offset(1)?;
            }
        }
        Ok(())
    }

    pub(super) fn texture_gather(&mut self) -> Result<()> {
        let i = self.instruction;
        let uniform = i.opcode() == 0xf63;
        let array = i.bits(61, 3) == 5;
        let offsets = i.bits(76, 2);
        let depth = i.bits(78, 1) != 0;
        let half = i.bits(79, 1) != 0;
        let mask = i.bits(72, 4);
        // Spatial coordinates precede offset/reference operands; arrays carry
        // the layer before x/y. Per-tap offsets contain signed six-bit fields.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L2566-L2790
        if !matches!(i.bits(61, 3), 1 | 5)
            || i.bits(59, 1) != 0
            || i.bits(80, 1) != 0
            || i.bits(81, 3) != 7
            || i.bits(89, 1) != 0
            || mask == 0
            || offsets == 3
            || (array && (offsets != 0 || depth || half))
            || (depth && (uniform || offsets != 0 || half))
            || (half
                && (uniform || offsets == 1 || (mask.count_ones() > 2 && i.bits(64, 8) == 255)))
            || (uniform && (i.bits(46, 13) != 0 || i.bits(91, 1) == 0 || offsets == 2))
            || (!uniform && i.bits(91, 1) != 0)
        {
            return Err(unsupported("Native texture gather mode"));
        }
        let a = Reg(i.bits(24, 8) as u8);
        let b = Reg(i.bits(32, 8) as u8);
        let read = |base: Reg, offset: i32| {
            if base == Reg::RZ {
                Ok(base)
            } else {
                base.offset(offset)
            }
        };
        let x = self.emit.get_reg(read(a, i32::from(array))?)?;
        let y = self.emit.get_reg(if array || (offsets == 0 && !depth) {
            b
        } else {
            read(a, 1)?
        })?;
        let mut coords = [Value::None; 3];
        coords[0] = self.emit.bit_cast(Type::F32, x)?;
        coords[1] = self.emit.bit_cast(Type::F32, y)?;
        if array {
            let layer = self.emit.get_reg(a)?;
            let layer = self
                .emit
                .bit_field_extract(layer, Value::U32(0), Value::U32(16), false)?;
            coords[2] = self
                .emit
                .convert_i_to_f(32, 16, false, layer, FpControl::default())?;
        }
        let coords = self
            .emit
            .composite_construct(&coords[..if array { 3 } else { 2 }])?;
        let mut packed = [Value::None; 2];
        if offsets != 0 {
            for (word, output) in
                packed
                    .iter_mut()
                    .enumerate()
                    .take(if offsets == 2 { 2 } else { 1 })
            {
                let bits = self.emit.get_reg(read(b, word as i32)?)?;
                let mut fields = [Value::None; 4];
                let count = if offsets == 2 { 4 } else { 2 };
                for (index, field) in fields.iter_mut().enumerate().take(count) {
                    *field = self.emit.bit_field_extract(
                        bits,
                        Value::U32(index as u32 * 8),
                        Value::U32(6),
                        true,
                    )?;
                }
                *output = self.emit.composite_construct(&fields[..count])?;
            }
        }
        let handle = if uniform {
            self.register(i.bits(40, 6), true)?
        } else {
            self.emit.get_cbuf(
                Value::U32(i.bits(54, 5)),
                Value::U32(i.bits(40, 14) * 4),
                32,
                false,
            )?
        };
        let mut info = TextureInstInfo::default();
        info.set_texture_type(if array {
            TextureType::ColorArray2D
        } else {
            TextureType::Color2D
        });
        info.set_gather_component(i.bits(87, 2));
        info.set_is_depth(depth);
        info.set_relaxed_precision(half);
        let value = if depth {
            let dref = self.emit.get_reg(b)?;
            let dref = self.emit.bit_cast(Type::F32, dref)?;
            self.emit
                .image_gather_dref(handle, coords, packed[0], packed[1], dref, info)?
        } else {
            self.emit
                .image_gather(handle, coords, packed[0], packed[1], info)?
        };
        self.texture_result(value, false, half)
    }

    pub(super) fn texture_query(&mut self) -> Result<()> {
        let i = self.instruction;
        let bindless = i.opcode() == 0x370;
        let mask = i.bits(72, 4);
        let first = Reg(i.bits(16, 8) as u8);
        let second = Reg(i.bits(64, 8) as u8);
        // TXQ returns selected dimensions and level count; bit 90 is NODEP.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L2283-L2314
        if i.bits(59, 5) != u32::from(bindless)
            || mask == 0
            || i.bits(76, 14) != 0
            || i.bits(91, 1) != 0
            || (bindless && i.bits(32, 27) != 0)
            || (second == Reg::RZ && mask.count_ones() > 2)
        {
            return Err(unsupported("Native texture dimension query mode"));
        }
        let mut source = Reg(i.bits(24, 8) as u8);
        let handle = if bindless {
            let handle = self.emit.get_reg(source)?;
            source = source.offset(1)?;
            handle
        } else {
            self.emit.get_cbuf(
                Value::U32(i.bits(54, 5)),
                Value::U32(i.bits(40, 14) * 4),
                32,
                false,
            )?
        };
        let lod = self.emit.get_reg(source)?;
        let query = self
            .emit
            .image_query_dimension(handle, lod, Value::U1(mask & 8 == 0))?;
        let mut output = 0;
        for component in 0..4 {
            if mask & (1 << component) == 0 {
                continue;
            }
            let (base, offset) = if output < 2 {
                (first, output)
            } else {
                (second, output - 2)
            };
            let dest = if base == Reg::RZ {
                base
            } else {
                base.offset(offset)?
            };
            let value = self.emit.composite_extract(query, component)?;
            self.emit.set_reg(dest, value)?;
            output += 1;
        }
        Ok(())
    }

    pub(super) fn texture_sample(&mut self) -> Result<()> {
        let i = self.instruction;
        let dimension = i.bits(61, 3);
        let lod_mode = i.bits(87, 3);
        let depth = i.bits(78, 1) != 0;
        let offsets = i.bits(76, 1) != 0;
        let mask = i.bits(72, 4);
        if i.bits(59, 1) != 0
            || dimension == 6
            || i.bits(77, 1) != 0
            || i.bits(81, 3) != 7
            || !matches!(lod_mode, 0..=3 | 6 | 7)
            || mask == 0
            || (depth && mask != 1)
            || (offsets && matches!(dimension, 3 | 7))
        {
            return Err(unsupported("Native texture sample dimension/operands/mode"));
        }
        // Native operand lists split between Ra and Rb: 1+1, 2+1,
        // 2+2, then 4+n. The array index precedes spatial coordinates.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L2830-L2868
        let array = matches!(dimension, 4 | 5 | 7);
        let spatial = match dimension {
            0 | 4 => 1,
            1 | 5 => 2,
            _ => 3,
        };
        let coordinates = spatial + usize::from(array);
        let has_lod = matches!(lod_mode, 2 | 3 | 6 | 7);
        let count = coordinates + usize::from(has_lod) + usize::from(offsets) + usize::from(depth);
        let in_a = if count <= 4 { count.div_ceil(2) } else { 4 };
        let mut operands = [Value::None; 7];
        for (index, value) in operands.iter_mut().take(count).enumerate() {
            let (register, offset) = if index < in_a {
                (Reg(i.bits(24, 8) as u8), index)
            } else {
                (Reg(i.bits(32, 8) as u8), index - in_a)
            };
            *value = self.emit.get_reg(if register == Reg::RZ {
                register
            } else {
                register.offset(offset as i32)?
            })?;
        }
        let handle = self.emit.get_cbuf(
            Value::U32(i.bits(54, 5)),
            Value::U32(i.bits(40, 14) * 4),
            32,
            false,
        )?;
        let mut coords = [Value::None; 4];
        for (index, value) in coords.iter_mut().take(spatial).enumerate() {
            *value = self
                .emit
                .bit_cast(Type::F32, operands[index + usize::from(array)])?;
        }
        if array {
            let layer =
                self.emit
                    .bit_field_extract(operands[0], Value::U32(0), Value::U32(16), false)?;
            coords[spatial] =
                self.emit
                    .convert_i_to_f(32, 16, false, layer, FpControl::default())?;
        }
        let coords = if coordinates == 1 {
            coords[0]
        } else {
            self.emit.composite_construct(&coords[..coordinates])?
        };
        let mut next = coordinates;
        let lod = if has_lod {
            let value = self.emit.bit_cast(Type::F32, operands[next])?;
            next += 1;
            value
        } else {
            Value::F32(0)
        };
        let offset = if offsets {
            let packed = operands[next];
            next += 1;
            let mut values = [Value::None; 3];
            for (index, value) in values.iter_mut().take(spatial).enumerate() {
                *value = self.emit.bit_field_extract(
                    packed,
                    Value::U32(index as u32 * 4),
                    Value::U32(4),
                    true,
                )?;
            }
            if spatial == 1 {
                values[0]
            } else {
                self.emit.composite_construct(&values[..spatial])?
            }
        } else {
            Value::None
        };
        let dref = if depth {
            self.emit.bit_cast(Type::F32, operands[next])?
        } else {
            Value::None
        };
        let mut info = TextureInstInfo::default();
        info.set_texture_type(match dimension {
            0 => TextureType::Color1D,
            1 => TextureType::Color2D,
            2 => TextureType::Color3D,
            3 => TextureType::ColorCube,
            4 => TextureType::ColorArray1D,
            5 => TextureType::ColorArray2D,
            7 => TextureType::ColorArrayCube,
            _ => unreachable!(),
        });
        info.set_is_depth(depth);
        info.set_has_bias(matches!(lod_mode, 2 | 6));
        let half = i.bits(79, 1) != 0;
        info.set_relaxed_precision(half);
        // NODEP is a scheduling hint; data dependencies remain explicit in IR.
        let explicit = matches!(lod_mode, 1 | 3 | 7);
        let value = match (depth, explicit) {
            (false, false) => self.emit.image_sample_implicit_lod(
                handle,
                coords,
                lod,
                offset,
                Value::None,
                info,
            )?,
            (false, true) => self
                .emit
                .image_sample_explicit_lod(handle, coords, lod, offset, info)?,
            (true, false) => self.emit.image_sample_dref_implicit_lod(
                handle,
                coords,
                dref,
                lod,
                offset,
                Value::None,
                info,
            )?,
            (true, true) => self
                .emit
                .image_sample_dref_explicit_lod(handle, coords, dref, lod, offset, info)?,
        };
        self.texture_result(value, depth, half)
    }

    fn texture_result(&mut self, value: Value, depth: bool, half: bool) -> Result<()> {
        let i = self.instruction;
        let mask = i.bits(72, 4);
        let first = Reg(i.bits(16, 8) as u8);
        let second = Reg(i.bits(64, 8) as u8);
        let mut output = 0;
        let mut halves = [Value::F32(0); 4];
        for component in 0..4 {
            if mask & (1 << component) == 0 {
                continue;
            }
            let component = if depth {
                value
            } else {
                self.emit.composite_extract(value, component)?
            };
            if half {
                halves[output as usize] = component;
            } else {
                let dest = if second != Reg::RZ && output >= 2 {
                    second.offset(output - 2)?
                } else if first == Reg::RZ {
                    Reg::RZ
                } else {
                    first.offset(output)?
                };
                let component = self.emit.bit_cast(Type::U32, component)?;
                self.emit.set_reg(dest, component)?;
            }
            output += 1;
        }
        if half {
            let pair = self.emit.composite_construct(&halves[..2])?;
            let packed = self.emit.pack_half2x16(pair)?;
            self.emit.set_reg(first, packed)?;
            if output > 2 {
                let pair = self.emit.composite_construct(&halves[2..])?;
                let packed = self.emit.pack_half2x16(pair)?;
                let dest = if second != Reg::RZ {
                    second
                } else if first == Reg::RZ {
                    Reg::RZ
                } else {
                    first.offset(1)?
                };
                self.emit.set_reg(dest, packed)?;
            }
        }
        Ok(())
    }
}
