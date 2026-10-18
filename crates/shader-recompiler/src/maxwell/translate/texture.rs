use super::{Result, Value, Visitor, unsupported};
use crate::{
    ir::{FpControl, Reg, TextureInstInfo},
    types::TextureType,
};

pub(super) fn texture_type(kind: u32) -> Result<TextureType> {
    match kind {
        0 => Ok(TextureType::Color1D),
        1 => Ok(TextureType::ColorArray1D),
        2 => Ok(TextureType::Color2D),
        3 => Ok(TextureType::ColorArray2D),
        4 => Ok(TextureType::Color3D),
        6 => Ok(TextureType::ColorCube),
        7 => Ok(TextureType::ColorArrayCube),
        _ => Err(unsupported("Texture dimension")),
    }
}
impl Visitor<'_> {
    pub(super) fn next_reg(&mut self, reg: &mut Reg, float: bool) -> Result<Value> {
        let value = if float {
            self.read_float_reg(*reg)?
        } else {
            self.emit.get_reg(*reg)?
        };
        *reg = reg.offset(1)?;
        Ok(value)
    }
    pub(super) fn vector_or_scalar(&mut self, values: &[Value]) -> Result<Value> {
        if values.len() == 1 {
            Ok(values[0])
        } else {
            self.emit.composite_construct(values)
        }
    }
    fn texture_coords(&mut self, kind: u32, integer: bool, omit_array: bool) -> Result<Value> {
        texture_type(kind)?;
        let array = matches!(kind, 1 | 3 | 7);
        let dimensions = match kind {
            0 | 1 => 1,
            2 | 3 => 2,
            _ => 3,
        };
        let mut coords = [Value::None; 4];
        let base = self.instruction.reg8();
        for (i, value) in coords.iter_mut().enumerate().take(dimensions) {
            let reg = base.offset((i + usize::from(array)) as i32)?;
            *value = if integer {
                self.emit.get_reg(reg)?
            } else {
                self.read_float_reg(reg)?
            };
        }
        let count = if array && !omit_array {
            let layer = self.emit.get_reg(base)?;
            coords[dimensions] = if integer {
                self.emit
                    .bit_field_extract(layer, Value::U32(0), Value::U32(16), false)?
            } else {
                self.emit
                    .convert_i_to_f(32, 16, false, layer, FpControl::default())?
            };
            dimensions + 1
        } else {
            dimensions
        };
        self.vector_or_scalar(&coords[..count])
    }
    pub(super) fn offset_fields(
        &mut self,
        value: Value,
        count: usize,
        start: u32,
        stride: u32,
        width: u32,
    ) -> Result<Value> {
        let mut values = [Value::None; 4];
        for (i, part) in values.iter_mut().enumerate().take(count) {
            *part = self.emit.bit_field_extract(
                value,
                Value::U32(start + i as u32 * stride),
                Value::U32(width),
                true,
            )?;
        }
        self.vector_or_scalar(&values[..count])
    }
    fn texture_offset(&mut self, reg: &mut Reg, kind: u32, gather: bool) -> Result<Value> {
        let value = self.next_reg(reg, false)?;
        let count = match kind {
            0 | 1 => 1,
            2 | 3 => 2,
            4 | 5 => 3,
            _ => return Err(unsupported("Offset on cube texture")),
        };
        self.offset_fields(
            value,
            count,
            0,
            if gather { 8 } else { 4 },
            if gather { 6 } else { 4 },
        )
    }
    fn texture_handle(&mut self, reg: &mut Reg, bindless: bool) -> Result<Value> {
        if bindless {
            self.next_reg(reg, false)
        } else {
            Ok(Value::U32(self.field::<36, 13>() * 4))
        }
    }
    pub(super) fn texture_result(&mut self, value: Value, depth: bool) -> Result<()> {
        let mut dest = self.instruction.reg0();
        for component in 0..4 {
            if self.field::<31, 4>() & (1 << component) == 0 {
                continue;
            }
            let part = if depth {
                if component < 3 {
                    value
                } else {
                    Value::F32(1.0f32.to_bits())
                }
            } else {
                self.emit.composite_extract(value, component)?
            };
            self.write_float_reg(dest, part)?;
            dest = dest.offset(1)?;
        }
        if self.field::<51, 3>() != 7 {
            let sparse = self.emit.get_sparse_from_op(value)?;
            let pred = self.emit.logical_not(sparse)?;
            self.set_predicate::<51>(pred)?;
        }
        Ok(())
    }
    pub(super) fn tex(&mut self, bindless: bool) -> Result<()> {
        let (offset, lod_mode, clamp) = if bindless {
            (self.bit::<36>(), self.field::<37, 3>(), self.bit::<40>())
        } else {
            (self.bit::<54>(), self.field::<55, 3>(), self.bit::<58>())
        };
        if clamp {
            return Err(unsupported("Texture LOD clamp"));
        }
        let kind = self.field::<28, 3>();
        let coords = self.texture_coords(kind, false, false)?;
        let mut meta = self.instruction.reg20();
        let handle = self.texture_handle(&mut meta, bindless)?;
        let lod = match lod_mode {
            0 | 1 => Value::F32(0),
            2 | 3 | 6 | 7 => self.next_reg(&mut meta, true)?,
            _ => return Err(unsupported("Texture bias/LOD mode")),
        };
        let offset = if offset {
            self.texture_offset(&mut meta, kind, false)?
        } else {
            Value::None
        };
        let depth = self.bit::<50>();
        let dref = if depth {
            self.next_reg(&mut meta, true)?
        } else {
            Value::None
        };
        let mut info = TextureInstInfo::default();
        info.set_texture_type(texture_type(kind)?);
        info.set_is_depth(depth);
        info.set_has_bias(matches!(lod_mode, 2 | 6));
        info.set_ndv_is_active(self.bit::<35>());
        let explicit = matches!(lod_mode, 1 | 3 | 7);
        let sample = match (depth, explicit) {
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
        self.texture_result(sample, depth)
    }
    pub(super) fn tld(&mut self, bindless: bool) -> Result<()> {
        let kind = self.field::<28, 3>();
        let coords = self.texture_coords(kind, true, false)?;
        let mut meta = self.instruction.reg20();
        let handle = self.texture_handle(&mut meta, bindless)?;
        let lod = if self.bit::<55>() {
            self.next_reg(&mut meta, false)?
        } else {
            Value::U32(0)
        };
        let offset = if self.bit::<35>() {
            self.texture_offset(&mut meta, kind, false)?
        } else {
            Value::None
        };
        let sample_index = if self.bit::<50>() {
            self.next_reg(&mut meta, false)?
        } else {
            Value::None
        };
        if self.bit::<54>() {
            return Err(unsupported("Texture fetch clamp"));
        }
        let mut info = TextureInstInfo::default();
        info.set_texture_type(texture_type(kind)?);
        let sample = self
            .emit
            .image_fetch(handle, coords, offset, lod, sample_index, info)?;
        self.texture_result(sample, false)
    }
    pub(super) fn tld4(&mut self, bindless: bool) -> Result<()> {
        let kind = self.field::<28, 3>();
        let coords = self.texture_coords(kind, false, false)?;
        let mut meta = self.instruction.reg20();
        let handle = self.texture_handle(&mut meta, bindless)?;
        let offset_kind = if bindless {
            self.field::<36, 2>()
        } else {
            self.field::<54, 2>()
        };
        let (offset, offset2) = match offset_kind {
            0 => (Value::None, Value::None),
            1 => (self.texture_offset(&mut meta, kind, true)?, Value::None),
            2 => {
                let a = self.next_reg(&mut meta, false)?;
                let b = self.next_reg(&mut meta, false)?;
                (
                    self.offset_fields(a, 4, 0, 8, 6)?,
                    self.offset_fields(b, 4, 0, 8, 6)?,
                )
            }
            _ => return Err(unsupported("Texture gather offset mode")),
        };
        let depth = self.bit::<50>();
        let dref = if depth {
            self.next_reg(&mut meta, true)?
        } else {
            Value::None
        };
        let mut info = TextureInstInfo::default();
        info.set_texture_type(texture_type(kind)?);
        info.set_is_depth(depth);
        info.set_gather_component(if bindless {
            self.field::<38, 2>()
        } else {
            self.field::<56, 2>()
        });
        let sample = if depth {
            self.emit
                .image_gather_dref(handle, coords, offset, offset2, dref, info)?
        } else {
            self.emit
                .image_gather(handle, coords, offset, offset2, info)?
        };
        self.texture_result(sample, false)
    }
    pub(super) fn txd(&mut self, bindless: bool) -> Result<()> {
        if self.bit::<50>() {
            return Err(unsupported("Texture gradient LOD clamp"));
        }
        let kind = self.field::<28, 3>();
        let mut base = self.instruction.reg8();
        let handle = self.texture_handle(&mut base, bindless)?;
        let dimensions = match kind {
            0 | 1 => 1,
            2 | 3 => 2,
            _ => return Err(unsupported("Texture gradient dimension")),
        };
        let last = base.offset(dimensions as i32)?;
        let mut coords = [Value::None; 3];
        for (i, value) in coords.iter_mut().enumerate().take(dimensions) {
            *value = self.read_float_reg(base.offset(i as i32)?)?;
        }
        let array = kind % 2 == 1;
        if array {
            let layer = self.emit.get_reg(last)?;
            let layer = self
                .emit
                .bit_field_extract(layer, Value::U32(0), Value::U32(16), false)?;
            coords[dimensions] =
                self.emit
                    .convert_i_to_f(32, 16, false, layer, FpControl::default())?;
        }
        let coords = self.vector_or_scalar(&coords[..dimensions + usize::from(array)])?;
        let mut derivatives = [Value::None; 4];
        for (i, value) in derivatives.iter_mut().enumerate().take(dimensions * 2) {
            *value = self.read_float_reg(self.instruction.reg20().offset(i as i32)?)?;
        }
        let derivatives = self
            .emit
            .composite_construct(&derivatives[..dimensions * 2])?;
        let offset = if self.bit::<35>() {
            let packed = self.emit.get_reg(last)?;
            self.offset_fields(packed, 2, 16, 4, 4)?
        } else {
            Value::None
        };
        let mut info = TextureInstInfo::default();
        info.set_texture_type(texture_type(kind)?);
        info.set_num_derivatives(dimensions as u32);
        let sample =
            self.emit
                .image_gradient(handle, coords, derivatives, offset, Value::None, info)?;
        self.texture_result(sample, false)
    }
    pub(super) fn tmml(&mut self, bindless: bool) -> Result<()> {
        let mask = self.field::<31, 4>();
        if mask & 12 != 0 {
            return Err(unsupported("Texture LOD query B/A results"));
        }
        let kind = self.field::<28, 3>();
        let coords = self.texture_coords(kind, false, true)?;
        let mut meta = self.instruction.reg20();
        let handle = self.texture_handle(&mut meta, bindless)?;
        let mut info = TextureInstInfo::default();
        info.set_texture_type(texture_type(kind)?);
        let sample = self.emit.image_query_lod(handle, coords, info)?;
        let mut dest = self.instruction.reg0();
        for component in 0..2 {
            if mask & (1 << component) == 0 {
                continue;
            }
            let value = self.emit.composite_extract(sample, component)?;
            let value = self.emit.convert_f_to_i(32, false, value)?;
            let value = self.emit.shift_left_logical(value, Value::U32(8))?;
            self.emit.set_reg(dest, value)?;
            dest = dest.offset(1)?;
        }
        Ok(())
    }
    pub(super) fn txq(&mut self, bindless: bool) -> Result<()> {
        let mut source = self.instruction.reg8();
        let handle = self.texture_handle(&mut source, bindless)?;
        if self.field::<22, 3>() != 1 {
            return Err(unsupported("Texture query mode"));
        }
        let mask = self.field::<31, 4>();
        let lod = self.emit.get_reg(source)?;
        let query = self
            .emit
            .image_query_dimension(handle, lod, Value::U1(mask & 8 == 0))?;
        let mut dest = self.instruction.reg0();
        for component in 0..4 {
            if mask & (1 << component) == 0 {
                continue;
            }
            let value = self.emit.composite_extract(query, component)?;
            self.emit.set_reg(dest, value)?;
            dest = dest.offset(1)?;
        }
        Ok(())
    }
}
