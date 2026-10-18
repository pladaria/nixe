use super::{Operands, Result, Source, Value, Visitor, unsupported};
use crate::{
    ir::{Attribute, FpControl, Patch, Reg},
    maxwell::bits::extract_signed,
    types::{PixelImap, Stage},
};

impl Visitor<'_> {
    pub(super) fn read_float_reg(&mut self, reg: Reg) -> Result<Value> {
        Operands {
            emit: &mut self.emit,
            instruction: self.instruction,
        }
        .read_f32(reg)
    }
    pub(super) fn write_float_reg(&mut self, reg: Reg, value: Value) -> Result<()> {
        Operands {
            emit: &mut self.emit,
            instruction: self.instruction,
        }
        .write_f32(reg, value)
    }
    pub(super) fn al2p(&mut self) -> Result<()> {
        if self.field::<47, 2>() != 0 {
            return Err(unsupported("AL2P size"));
        }
        let index = self.a()?;
        let value = self.emit.i_add(
            index,
            Value::U32(extract_signed::<20, 11>(self.instruction.0) as u32),
        )?;
        self.write(value)
    }
    pub(super) fn attribute_memory(&mut self, store: bool) -> Result<()> {
        let indexed = self.instruction.reg8() != Reg::RZ;
        if store && indexed {
            return Err(unsupported("Indexed attribute store"));
        }
        let offset = self.field::<20, 10>();
        if !offset.is_multiple_of(4) {
            return Err(unsupported("Unaligned attribute address"));
        }
        let vertex = self.emit.get_reg(self.instruction.reg39())?;
        let count = self.field::<47, 2>() + 1;
        let patch = self.bit::<31>();
        if indexed && patch {
            return Err(unsupported("Indexed patch read"));
        }
        let index = if indexed { self.a()? } else { Value::None };
        for element in 0..count {
            let reg = self.instruction.reg0().offset(element as i32)?;
            if indexed {
                let address = if element == 0 {
                    index
                } else {
                    self.emit.i_add(index, Value::U32(element * 4))?
                };
                let value = self.emit.get_attribute_indexed_at_vertex(address, vertex)?;
                self.write_float_reg(reg, value)?;
            } else if patch {
                let slot = offset / 4 + element;
                if slot > u8::MAX as u32 {
                    return Err(unsupported("Patch address out of range"));
                }
                let patch = Patch::new(slot as u8)?;
                if store {
                    let value = self.read_float_reg(reg)?;
                    self.emit.set_patch(patch, value)?;
                } else {
                    let value = self.emit.get_patch(patch)?;
                    self.write_float_reg(reg, value)?;
                }
            } else {
                let attribute = Attribute::from_index((offset / 4 + element) as u16).unwrap();
                if store {
                    let value = self.read_float_reg(reg)?;
                    self.emit.set_attribute(attribute, value, vertex)?;
                } else {
                    let value = self.emit.get_attribute_at_vertex(attribute, vertex)?;
                    self.write_float_reg(reg, value)?;
                }
            }
        }
        Ok(())
    }
    pub(super) fn ipa(&mut self) -> Result<()> {
        let attribute = Attribute::from_index(self.field::<30, 8>() as u16).unwrap();
        let mode = self.field::<54, 2>();
        let mut value = if self.bit::<38>() && self.instruction.reg8() != Reg::RZ {
            let index = self.a()?;
            self.emit.get_attribute_indexed(index)?
        } else {
            self.emit.get_attribute(attribute)?
        };
        let perspective = if attribute.is_legacy() {
            mode != 3
        } else if attribute.is_generic() {
            let map = self
                .metadata
                .sph
                .generic_input_map(attribute.generic_index()? as usize);
            let mode = map
                .into_iter()
                .find(|value| *value != PixelImap::Unused)
                .unwrap_or(PixelImap::Unused);
            matches!(mode, PixelImap::Unused | PixelImap::Perspective)
        } else {
            false
        };
        if perspective {
            let w = self.emit.get_attribute(Attribute::PositionW)?;
            value = self.emit.fp_mul(value, w, FpControl::default())?;
        }
        if mode == 1 {
            let factor = self.read_float_reg(self.instruction.reg20())?;
            value = self.emit.fp_mul(value, factor, FpControl::default())?;
        }
        if self.bit::<51>() {
            if attribute == Attribute::FrontFace {
                return Err(unsupported("Saturated FrontFace interpolation"));
            }
            value = self.emit.fp_saturate(value)?;
        }
        self.write_float_reg(self.instruction.reg0(), value)
    }
    pub(super) fn isberd(&mut self) -> Result<()> {
        let mut index = if self.instruction.reg8() == Reg::RZ {
            Value::U32(self.field::<24, 8>())
        } else {
            let mut base = self.a()?;
            match self.field::<47, 2>() {
                0 => (),
                shift @ 1..=2 => base = self.emit.shift_left_logical(base, Value::U32(shift))?,
                _ => return Err(unsupported("ISBERD index shift")),
            }
            self.emit.i_add(base, Value::U32(self.field::<24, 8>()))?
        };
        if self.bit::<32>() {
            let size = self.field::<36, 4>();
            if size > 3 {
                return Err(unsupported("ISBERD access size"));
            }
            if self.bit::<31>() {
                let lane = self.emit.lane_id()?;
                let skew = if size == 0 {
                    lane
                } else {
                    self.emit
                        .shift_left_logical(lane, Value::U32(size.min(2)))?
                };
                index = self.emit.i_add(index, skew)?;
            }
            let address = self.emit.u_convert(64, index)?;
            let value = match size {
                0 => self.emit.load_global_u8(address)?,
                1 => self.emit.load_global_u16(address)?,
                _ => self.emit.load_global32(address)?,
            };
            return self.write(value);
        }
        let mode = self.field::<33, 2>();
        if mode != 0 {
            if self.bit::<31>() {
                let lane = self.emit.lane_id()?;
                let skew = self.emit.shift_left_logical(lane, Value::U32(2))?;
                index = self.emit.i_add(index, skew)?;
            }
            if mode != 3 {
                return Err(crate::CompileError::new(
                    crate::diagnostics::FailureCategory::UnsupportedReferenceCase,
                    "ISBERD requires a symbolic patch or attribute, but its index is an integer",
                ));
            }
            let value = self.emit.get_attribute_indexed(index)?;
            return self.write_float_reg(self.instruction.reg0(), value);
        }
        let mut value = self.a()?;
        if self.bit::<31>() {
            let lane = self.emit.lane_id()?;
            value = self.emit.i_add(value, lane)?;
        }
        self.write(value)
    }
    pub(super) fn output(&mut self, source: Source) -> Result<()> {
        let stream = self.source(source)?;
        let stream = self.emit.bitwise_and(stream, Value::U32(3))?;
        if self.bit::<39>() {
            self.emit.emit_vertex(stream)?;
        }
        if self.bit::<40>() {
            self.emit.end_primitive(stream)?;
        }
        self.write(Value::U32(0))
    }
    pub(super) fn pixld(&mut self) -> Result<()> {
        if self.field::<45, 3>() != 7 {
            return Err(unsupported("PIXLD destination predicate"));
        }
        if self.instruction.reg8() != Reg::RZ || self.field::<20, 8>() != 0 {
            return Err(unsupported("PIXLD nonzero address"));
        }
        if self.field::<31, 3>() != 5 {
            return Err(unsupported("PIXLD mode"));
        }
        let value = self.emit.sample_id()?;
        self.write(value)
    }
    pub(super) fn exit(&mut self) -> Result<()> {
        if self.metadata.stage != Stage::Fragment {
            return Ok(());
        }
        let mut reg = Reg(0);
        for target in 0..8 {
            if !self.metadata.sph.has_output_components(target) {
                continue;
            }
            for (component, enabled) in self
                .metadata
                .sph
                .enabled_output_components(target)
                .into_iter()
                .enumerate()
            {
                if enabled {
                    let value = self.read_float_reg(reg)?;
                    self.emit
                        .set_frag_color(target as u32, component as u32, value)?;
                }
                reg = reg.offset(1)?;
            }
        }
        if self.metadata.sph.fragment_sample_mask() {
            let value = self.emit.get_reg(reg)?;
            self.emit.set_sample_mask(value)?;
        }
        if self.metadata.sph.fragment_depth() {
            let value = self.read_float_reg(reg.offset(1)?)?;
            self.emit.set_frag_depth(value)?;
        }
        Ok(())
    }
    pub(super) fn membar(&mut self) -> Result<()> {
        if self.field::<8, 2>() == 0 {
            self.emit.workgroup_memory_barrier()?;
        } else {
            self.emit.device_memory_barrier()?;
        }
        Ok(())
    }
    pub(super) fn bar(&mut self) -> Result<()> {
        if self.instruction.0 & 0x9b00000000 != 0x8000000000 {
            return Err(unsupported("BAR mode other than sync"));
        }
        if !self.bit::<43>() || !self.bit::<44>() {
            return Err(unsupported("BAR register operands"));
        }
        if self.field::<8, 8>() != 0 || self.field::<20, 12>() != 0 {
            return Err(unsupported("BAR nonzero operands"));
        }
        if self.field::<39, 3>() != 7 && self.bit::<42>() {
            return Err(unsupported("BAR negated input predicate"));
        }
        self.emit.barrier()?;
        Ok(())
    }
    pub(super) fn vote(&mut self) -> Result<()> {
        let input = self.predicate::<39, 42>()?;
        let value = match self.field::<48, 2>() {
            0 => self.emit.vote_all(input)?,
            1 => self.emit.vote_any(input)?,
            2 => self.emit.vote_equal(input)?,
            _ => return Err(unsupported("VOTE mode")),
        };
        self.set_predicate::<45>(value)?;
        let ballot = self.emit.subgroup_ballot(input)?;
        self.write(ballot)
    }
    pub(super) fn shfl(&mut self) -> Result<()> {
        let index = if self.bit::<28>() {
            Value::U32(self.field::<20, 5>())
        } else {
            self.source(Source::Reg20)?
        };
        let mask = if self.bit::<29>() {
            Value::U32(self.field::<34, 13>())
        } else {
            self.source(Source::Reg39)?
        };
        let mode = self.field::<30, 2>();
        let input = self.a()?;
        if self.metadata.stage == Stage::Fragment
            && mask == Value::U32((28 << 8) | 3)
            && let Value::U32(index) = index
        {
            let value = match (mode, index) {
                (0, 0..=3) => Some(self.emit.quad_broadcast(input, Value::U32(index))?),
                (3, 1..=3) => Some(self.emit.quad_swap(input, Value::U32(index - 1))?),
                _ => None,
            };
            if let Some(value) = value {
                self.write(value)?;
                self.set_predicate::<48>(Value::U1(true))?;
                return Ok(());
            }
        }
        let clamp = self
            .emit
            .bit_field_extract(mask, Value::U32(0), Value::U32(5), false)?;
        let segment = self
            .emit
            .bit_field_extract(mask, Value::U32(8), Value::U32(5), false)?;
        let value = match mode {
            0 => self.emit.shuffle_index(input, index, clamp, segment)?,
            1 => self.emit.shuffle_up(input, index, clamp, segment)?,
            2 => self.emit.shuffle_down(input, index, clamp, segment)?,
            _ => self.emit.shuffle_butterfly(input, index, clamp, segment)?,
        };
        let in_bounds = self.emit.get_in_bounds_from_op(value)?;
        self.set_predicate::<48>(in_bounds)?;
        self.write(value)
    }
}
