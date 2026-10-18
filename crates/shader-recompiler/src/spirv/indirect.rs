use super::{BufferKind as K, EmitContext, Id, InputLoad};
use crate::{
    CompileError, Result,
    artifact::ShaderInfo,
    diagnostics::FailureCategory,
    ir::{Attribute, Type},
    types::Stage,
};
use spirv::Op;
impl EmitContext<'_> {
    fn cbuf_accessor(&mut self, kind: K) -> Result<Id> {
        let (ty, _) = self.buffer_element(kind);
        let u = self.types.u32[0];
        let ft = self.builder.type_function(ty, &[u, u])?;
        let function = self.builder.begin_function(ty, ft)?;
        let bank = self.builder.parameter(u)?;
        let offset = self.builder.parameter(u)?;
        self.builder.new_label()?;
        let end = self.builder.reserve_id()?;
        let labels = (0..14)
            .map(|_| self.builder.reserve_id())
            .collect::<Result<Vec<_>>>()?;
        let mut switch = vec![bank.0, labels[0].0];
        for (i, label) in labels.iter().enumerate() {
            switch.extend([i as u32, label.0]);
        }
        self.builder.instruction(Op::SelectionMerge, &[end.0, 0])?;
        self.builder.instruction(Op::Switch, &switch)?;
        for (index, &label) in labels.iter().enumerate() {
            self.builder.label(label)?;
            let pointer = self.builder.result(
                Op::AccessChain,
                self.uniform_types[kind as usize],
                &[
                    self.cbufs[index][kind as usize].0,
                    self.constants.u32_zero.0,
                    offset.0,
                ],
            )?;
            let value = self.builder.result(Op::Load, ty, &[pointer.0])?;
            self.builder.instruction(Op::ReturnValue, &[value.0])?;
        }
        self.builder.label(end)?;
        self.builder.instruction(Op::Unreachable, &[])?;
        self.builder.end_function()?;
        Ok(function)
    }
    pub(super) fn define_cbuf_accessors(&mut self, info: &ShaderInfo) -> Result<()> {
        if !info.uses_cbuf_indirect {
            return Ok(());
        }
        let mut types = info.used_indirect_cbuf_types;
        let alias = self.profile.support_descriptor_aliasing;
        if alias && (types & Type::U8).0 != 0 {
            if self.profile.support_int8 && self.profile.support_uniform_and_storage_buffer_8bit {
                self.objects.load_const_func_u8 = self.cbuf_accessor(K::U8)?;
            } else {
                types |= Type::U32;
            }
        }
        if alias && (types & Type::U16).0 != 0 {
            if self.profile.support_int16 && self.profile.support_uniform_and_storage_buffer_16bit {
                self.objects.load_const_func_u16 = self.cbuf_accessor(K::U16)?;
            } else {
                types |= Type::U32;
            }
        }
        if alias && (types & Type::F32).0 != 0 {
            self.objects.load_const_func_f32 = self.cbuf_accessor(K::F32)?;
        }
        if alias && (types & Type::U32).0 != 0 {
            self.objects.load_const_func_u32 = self.cbuf_accessor(K::U32)?;
        }
        if alias && (types & Type::U32x2).0 != 0 {
            self.objects.load_const_func_u32x2 = self.cbuf_accessor(K::U32x2)?;
        }
        if !alias || (types & Type::U32x4).0 != 0 {
            self.objects.load_const_func_u32x4 = self.cbuf_accessor(K::U32x4)?;
        }
        Ok(())
    }
    fn attribute_load(&mut self, info: &ShaderInfo) -> Result<Id> {
        let array = self.stage == Stage::Geometry;
        let u = self.types.u32[0];
        let f = self.types.f32[0];
        let params = if array { vec![u, u] } else { vec![u] };
        let ft = self.builder.type_function(f, &params)?;
        let function = self.builder.begin_function(f, ft)?;
        let offset = self.builder.parameter(u)?;
        let vertex = if array {
            self.builder.parameter(u)?
        } else {
            Id::NONE
        };
        self.builder.new_label()?;
        let (component, vector) = self.attribute_indices(offset)?;
        let end = self.builder.reserve_id()?;
        let default = self.builder.reserve_id()?;
        let mut cases = Vec::new();
        if info.loads.any_component(Attribute::PositionX) {
            cases.push((
                Attribute::PositionX.index() as u32 / 4,
                None,
                self.builder.reserve_id()?,
            ));
        }
        for index in 0..32 {
            if info.loads.generic(index) {
                cases.push((
                    Attribute::generic(index, 0).unwrap().index() as u32 / 4,
                    Some(index),
                    self.builder.reserve_id()?,
                ));
            }
        }
        let mut switch = vec![vector.0, default.0];
        for &(literal, _, label) in &cases {
            switch.extend([literal, label.0]);
        }
        self.builder.instruction(Op::SelectionMerge, &[end.0, 0])?;
        self.builder.instruction(Op::Switch, &switch)?;
        self.builder.label(default)?;
        self.builder
            .instruction(Op::ReturnValue, &[self.constants.f32_zero.0])?;
        for (_, index, label) in cases {
            self.builder.label(label)?;
            let (base, pointer_type, ty, load) = if let Some(index) = index {
                let generic = self.input_generics[index];
                if generic.id == Id::NONE {
                    self.builder
                        .instruction(Op::ReturnValue, &[self.constants.f32_zero.0])?;
                    continue;
                }
                (generic.id, generic.pointer, generic.component, generic.load)
            } else {
                (
                    self.objects.input_position,
                    self.types.input_f32,
                    f,
                    InputLoad::None,
                )
            };
            let mut indices = vec![base.0];
            if array {
                indices.push(vertex.0);
            }
            if index.is_none() && self.need_input_position_indirect {
                indices.push(self.constants.u32_zero.0);
            }
            indices.push(component.0);
            let pointer = self
                .builder
                .result(Op::AccessChain, pointer_type, &indices)?;
            let mut value = self.builder.result(Op::Load, ty, &[pointer.0])?;
            let convert = match load {
                InputLoad::None => None,
                InputLoad::Bitcast => Some(Op::Bitcast),
                InputLoad::SignedToFloat => Some(Op::ConvertSToF),
                InputLoad::UnsignedToFloat => Some(Op::ConvertUToF),
            };
            if let Some(convert) = convert {
                value = self.builder.result(convert, f, &[value.0])?;
            }
            self.builder.instruction(Op::ReturnValue, &[value.0])?;
        }
        self.builder.label(end)?;
        self.builder.instruction(Op::Unreachable, &[])?;
        self.builder.end_function()?;
        Ok(function)
    }
    fn attribute_indices(&mut self, offset: Id) -> Result<(Id, Id)> {
        let two = self.const_u32(2)?;
        let three = self.const_u32(3)?;
        let u = self.types.u32[0];
        let word = self
            .builder
            .result(Op::ShiftRightArithmetic, u, &[offset.0, two.0])?;
        let component = self.builder.result(Op::BitwiseAnd, u, &[word.0, three.0])?;
        let vector = self
            .builder
            .result(Op::ShiftRightArithmetic, u, &[word.0, two.0])?;
        Ok((component, vector))
    }
    fn attribute_store(&mut self, info: &ShaderInfo) -> Result<Id> {
        let ft = self
            .builder
            .type_function(self.types.void, &[self.types.u32[0], self.types.f32[0]])?;
        let function = self.builder.begin_function(self.types.void, ft)?;
        let offset = self.builder.parameter(self.types.u32[0])?;
        let value = self.builder.parameter(self.types.f32[0])?;
        self.builder.new_label()?;
        let (component, vector) = self.attribute_indices(offset)?;
        let end = self.builder.reserve_id()?;
        let default = self.builder.reserve_id()?;
        let mut cases = Vec::new();
        if info.stores.any_component(Attribute::PositionX) {
            cases.push((
                Attribute::PositionX.index() as u32 / 4,
                self.objects.output_position,
                false,
                self.builder.reserve_id()?,
            ));
        }
        for i in 0..32 {
            if info.stores.generic(i) {
                let generic = self.output_generics[i][0];
                if generic.components != 4 {
                    return Err(CompileError::new(
                        FailureCategory::Unsupported,
                        "Indexed attribute stores with split transform feedback outputs",
                    ));
                }
                cases.push((
                    Attribute::generic(i, 0).unwrap().index() as u32 / 4,
                    generic.id,
                    false,
                    self.builder.reserve_id()?,
                ));
            }
        }
        if info.stores.clip_distances() {
            if self.profile.max_user_clip_distances >= 4 {
                cases.push((
                    Attribute::ClipDistance0.index() as u32 / 4,
                    self.objects.clip_distances,
                    false,
                    self.builder.reserve_id()?,
                ));
            }
            if self.profile.max_user_clip_distances >= 8 {
                cases.push((
                    Attribute::ClipDistance4.index() as u32 / 4,
                    self.objects.clip_distances,
                    true,
                    self.builder.reserve_id()?,
                ));
            }
        }
        let mut switch = vec![vector.0, default.0];
        for &(literal, _, _, label) in &cases {
            switch.extend([literal, label.0]);
        }
        self.builder.instruction(Op::SelectionMerge, &[end.0, 0])?;
        self.builder.instruction(Op::Switch, &switch)?;
        self.builder.label(default)?;
        self.builder.instruction(Op::Return, &[])?;
        for (_, base, high, label) in cases {
            self.builder.label(label)?;
            let index = if high {
                let four = self.const_u32(4)?;
                self.builder
                    .result(Op::IAdd, self.types.u32[0], &[component.0, four.0])?
            } else {
                component
            };
            let pointer =
                self.builder
                    .result(Op::AccessChain, self.types.output_f32, &[base.0, index.0])?;
            self.builder.instruction(Op::Store, &[pointer.0, value.0])?;
            self.builder.instruction(Op::Return, &[])?;
        }
        self.builder.label(end)?;
        self.builder.instruction(Op::Unreachable, &[])?;
        self.builder.end_function()?;
        Ok(function)
    }
    pub(super) fn define_attribute_accessors(&mut self, info: &ShaderInfo) -> Result<()> {
        if info.loads_indexed_attributes {
            self.objects.indexed_load_func = self.attribute_load(info)?;
        }
        if info.stores_indexed_attributes {
            self.objects.indexed_store_func = self.attribute_store(info)?;
        }
        Ok(())
    }
}
