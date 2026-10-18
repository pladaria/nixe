use super::*;
use crate::{
    ir::{Attribute as A, Patch},
    spirv::InputLoad,
    types::{AttributeType, Stage},
};
use spirv::StorageClass;
impl EmitContext<'_> {
    fn input_attr_pointer(
        &mut self,
        pointer: Id,
        object: Id,
        vertex: Id,
        components: &[Id],
    ) -> Result<Id> {
        let mut args = [Id::NONE; 4];
        args[0] = object;
        let start = if matches!(
            self.stage,
            Stage::Geometry | Stage::TessellationControl | Stage::TessellationEval
        ) {
            args[1] = vertex;
            2
        } else {
            1
        };
        args[start..start + components.len()].copy_from_slice(components);
        self.op(Op::AccessChain, pointer, &args[..start + components.len()])
    }
    fn output_attr_pointer(&mut self, object: Id, component: Id) -> Result<Id> {
        let args = if self.stage == Stage::TessellationControl {
            let invocation = self.op(Op::Load, self.types.u32[0], &[self.objects.invocation_id])?;
            [object, invocation, component]
        } else {
            [object, component, Id::NONE]
        };
        self.op(
            Op::AccessChain,
            self.types.output_f32,
            &args[..if self.stage == Stage::TessellationControl {
                3
            } else {
                2
            }],
        )
    }
    fn integer_attribute(&mut self, attr: A) -> Result<Id> {
        let object = match attr {
            A::PrimitiveId => self.objects.primitive_id,
            A::Layer => self.objects.layer,
            A::InstanceId if self.profile.support_vertex_instance_id => self.objects.instance_id,
            A::InstanceId => {
                let index = self.op(Op::Load, self.types.u32[0], &[self.objects.instance_index])?;
                let base = if let Some(base) = self.runtime.base_instance {
                    self.const_u32(base)?
                } else {
                    self.op(Op::Load, self.types.u32[0], &[self.objects.base_instance])?
                };
                return self.op(Op::ISub, self.types.u32[0], &[index, base]);
            }
            A::VertexId => {
                if self.profile.support_vertex_instance_id {
                    self.objects.vertex_id
                } else {
                    self.objects.vertex_index
                }
            }
            A::BaseInstance => {
                if let Some(base) = self.runtime.base_instance {
                    return self.const_u32(base);
                }
                self.objects.base_instance
            }
            A::BaseVertex => self.objects.base_vertex,
            A::DrawID => self.objects.draw_index,
            _ => {
                return Err(unsupported(format!(
                    "Unsupported integer input attribute {attr:?}"
                )));
            }
        };
        self.op(Op::Load, self.types.u32[0], &[object])
    }
    fn get_attribute(&mut self, attr: A, vertex: Id, integer: bool) -> Result<Id> {
        let f = self.types.f32[0];
        if integer {
            if attr == A::Layer {
                return Err(unsupported("Layer U32 input attribute"));
            }
            return self.integer_attribute(attr);
        }
        let element = attr.index() as u32 % 4;
        if attr.is_generic() {
            let generic = self.input_generics[attr.generic_index()? as usize];
            if generic.id == Id::NONE {
                return self.const_f32(if element == 3 { 1.0 } else { 0.0 });
            }
            let element = self.const_u32(element)?;
            let pointer =
                self.input_attr_pointer(generic.pointer, generic.id, vertex, &[element])?;
            let value = self.op(Op::Load, generic.component, &[pointer])?;
            return match generic.load {
                InputLoad::None => Ok(value),
                InputLoad::Bitcast => self.op(Op::Bitcast, f, &[value]),
                InputLoad::SignedToFloat => self.op(Op::ConvertSToF, f, &[value]),
                InputLoad::UnsignedToFloat => self.op(Op::ConvertUToF, f, &[value]),
            };
        }
        match attr {
            A::PositionX | A::PositionY | A::PositionZ | A::PositionW => {
                let element = self.const_u32(element)?;
                let indices = [self.constants.u32_zero, element];
                let pointer = self.input_attr_pointer(
                    self.types.input_f32,
                    self.objects.input_position,
                    vertex,
                    if self.need_input_position_indirect {
                        &indices
                    } else {
                        &indices[1..]
                    },
                )?;
                self.op(Op::Load, f, &[pointer])
            }
            A::FrontFace => {
                let front = self.op(Op::Load, self.types.bool, &[self.objects.front_face])?;
                let bits = self.const_u32(u32::MAX)?;
                let value = self.op(Op::Bitcast, f, &[bits])?;
                self.op(Op::Select, f, &[front, value, self.constants.f32_zero])
            }
            A::PointSpriteS
            | A::PointSpriteT
            | A::TessellationEvaluationPointU
            | A::TessellationEvaluationPointV => {
                let index = self.const_u32(u32::from(matches!(
                    attr,
                    A::PointSpriteT | A::TessellationEvaluationPointV
                )))?;
                let object = if matches!(attr, A::PointSpriteS | A::PointSpriteT) {
                    self.objects.point_coord
                } else {
                    self.objects.tess_coord
                };
                let pointer = self.op(Op::AccessChain, self.types.input_f32, &[object, index])?;
                self.op(Op::Load, f, &[pointer])
            }
            _ => {
                let value = self.integer_attribute(attr)?;
                self.op(Op::Bitcast, f, &[value])
            }
        }
    }
    fn set_attribute(&mut self, attr: A, mut value: Id) -> Result<()> {
        let mut integer = false;
        let pointer = if attr.is_generic() {
            let element = attr.generic_element()?;
            let info = self.output_generics[attr.generic_index()? as usize][element as usize];
            if info.components == 1 {
                info.id
            } else {
                let component = self.const_u32(element - info.first)?;
                self.output_attr_pointer(info.id, component)?
            }
        } else if (A::ClipDistance0.index()..=A::ClipDistance7.index()).contains(&attr.index()) {
            let index = (attr.index() - A::ClipDistance0.index()) as u32;
            if index >= self.profile.max_user_clip_distances {
                return Ok(());
            }
            self.clip_distance_written |= 1 << index;
            let index = self.const_u32(index)?;
            self.output_attr_pointer(self.objects.clip_distances, index)?
        } else {
            match attr {
                A::PointSize => self.objects.output_point_size,
                A::PositionX | A::PositionY | A::PositionZ | A::PositionW => {
                    let index = self.const_u32(attr.index() as u32 % 4)?;
                    self.output_attr_pointer(self.objects.output_position, index)?
                }
                A::Layer | A::ViewportIndex => {
                    if attr == A::ViewportIndex && !self.profile.support_multi_viewport {
                        return Ok(());
                    }
                    if !self.profile.support_viewport_index_layer_non_geometry
                        && self.stage != Stage::Geometry
                    {
                        return Ok(());
                    }
                    integer = true;
                    if attr == A::Layer {
                        self.objects.layer
                    } else {
                        self.objects.viewport_index
                    }
                }
                A::ViewportMask => {
                    if !self.profile.support_viewport_mask {
                        return Ok(());
                    }
                    integer = true;
                    self.op(
                        Op::AccessChain,
                        self.types.output_u32,
                        &[self.objects.viewport_mask, self.constants.u32_zero],
                    )?
                }
                _ => {
                    return Err(unsupported(format!(
                        "Unsupported output attribute {attr:?}"
                    )));
                }
            }
        };
        if integer {
            value = self.op(Op::Bitcast, self.types.u32[0], &[value])?;
        }
        if pointer == Id::NONE {
            return Err(invariant("Output attribute is not declared"));
        }
        self.builder.instruction(Op::Store, &[pointer.0, value.0])
    }
    fn patch_pointer(&mut self, patch: Patch, store: bool) -> Result<Id> {
        let (object, index) = if patch.is_generic() {
            (
                self.patches[patch.generic_index()? as usize],
                patch.generic_element()?,
            )
        } else if !store {
            return Err(unsupported("Non-generic patch load"));
        } else if patch.index() < 4 {
            (self.objects.output_tess_level_outer, patch.index() as u32)
        } else {
            (
                self.objects.output_tess_level_inner,
                patch.index() as u32 - 4,
            )
        };
        let index = self.const_u32(index)?;
        let ty = if store || self.stage == Stage::TessellationControl {
            self.types.output_f32
        } else {
            self.types.input_f32
        };
        self.op(Op::AccessChain, ty, &[object, index])
    }
    pub(super) fn attribute_instruction(&mut self, ir: &Ir, id: InstId) -> Result<Option<Id>> {
        let inst = ir.inst(id)?;
        let f = self.types.f32[0];
        let value = match inst.opcode() {
            O::GetAttribute | O::GetAttributeU32 | O::SetAttribute => {
                let Value::Attribute(attr) = inst.arg(0)? else {
                    return Err(invariant("Expected attribute operand"));
                };
                let value = self.def(ir, inst.arg(1)?)?;
                if inst.opcode() == O::SetAttribute {
                    self.set_attribute(attr, value)?;
                    Id::NONE
                } else {
                    self.get_attribute(attr, value, inst.opcode() == O::GetAttributeU32)?
                }
            }
            O::GetAttributeIndexed => {
                let offset = self.def(ir, inst.arg(0)?)?;
                let vertex = self.def(ir, inst.arg(1)?)?;
                let args = [self.objects.indexed_load_func, offset, vertex];
                let n = if matches!(
                    self.stage,
                    Stage::Geometry | Stage::TessellationControl | Stage::TessellationEval
                ) {
                    3
                } else {
                    2
                };
                self.op(Op::FunctionCall, f, &args[..n])?
            }
            O::SetAttributeIndexed => {
                let offset = self.def(ir, inst.arg(0)?)?;
                let value = self.def(ir, inst.arg(1)?)?;
                self.op(
                    Op::FunctionCall,
                    self.types.void,
                    &[self.objects.indexed_store_func, offset, value],
                )?;
                Id::NONE
            }
            O::GetPatch | O::SetPatch => {
                let Value::Patch(patch) = inst.arg(0)? else {
                    return Err(invariant("Expected patch operand"));
                };
                let store = inst.opcode() == O::SetPatch;
                let pointer = self.patch_pointer(patch, store)?;
                if store {
                    let value = self.def(ir, inst.arg(1)?)?;
                    self.builder.instruction(Op::Store, &[pointer.0, value.0])?;
                    Id::NONE
                } else {
                    self.op(Op::Load, f, &[pointer])?
                }
            }
            O::SetFragColor => {
                let index = literal(ir, inst.arg(0)?)? as usize;
                let component = self.const_u32(literal(ir, inst.arg(1)?)?)?;
                let mut value = self.def(ir, inst.arg(2)?)?;
                let output = *self
                    .frag_colors
                    .get(index)
                    .ok_or_else(|| invariant("Fragment color index out of range"))?;
                let ty = match self.runtime.color_output_types[index] {
                    AttributeType::UnsignedInt => self.types.u32[0],
                    AttributeType::SignedInt => self.types.s32[0],
                    _ => f,
                };
                let pointer_ty = self.builder.type_pointer(StorageClass::Output, ty)?;
                let pointer = self.op(Op::AccessChain, pointer_ty, &[output, component])?;
                if ty != f {
                    value = self.op(Op::Bitcast, ty, &[value])?;
                }
                self.builder.instruction(Op::Store, &[pointer.0, value.0])?;
                Id::NONE
            }
            O::SetSampleMask => {
                let value = self.def(ir, inst.arg(0)?)?;
                let pointer = self.op(
                    Op::AccessChain,
                    self.types.output_u32,
                    &[self.objects.sample_mask, self.constants.u32_zero],
                )?;
                self.builder.instruction(Op::Store, &[pointer.0, value.0])?;
                Id::NONE
            }
            O::SetFragDepth => {
                let mut value = self.def(ir, inst.arg(0)?)?;
                if self.runtime.convert_depth_mode && !self.profile.support_native_ndc {
                    let half = self.const_f32(0.5)?;
                    value = self.ext(GlslStd450Op::Fma, f, &[value, half, half])?;
                }
                self.builder
                    .instruction(Op::Store, &[self.objects.frag_depth.0, value.0])?;
                Id::NONE
            }
            _ => return Ok(None),
        };
        Ok(Some(value))
    }
}
