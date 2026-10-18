use super::*;
use crate::types::{CompareFunction, Stage};
use spirv::{MemorySemantics as M, Scope, StorageClass};
impl EmitContext<'_> {
    fn depth_mode(&mut self) -> Result<()> {
        let f = self.types.f32[0];
        let vector = self.op(Op::Load, self.types.f32[3], &[self.objects.output_position])?;
        let z = self.extract(f, vector, 2)?;
        let w = self.extract(f, vector, 3)?;
        let sum = self.op(Op::FAdd, f, &[z, w])?;
        let half = self.const_f32(0.5)?;
        let depth = self.op(Op::FMul, f, &[sum, half])?;
        let vector = self.builder.result(
            Op::CompositeInsert,
            self.types.f32[3],
            &[depth.0, vector.0, 2],
        )?;
        self.builder
            .instruction(Op::Store, &[self.objects.output_position.0, vector.0])
    }
    fn fixed_point_size(&mut self) -> Result<()> {
        if let Some(size) = self.runtime.fixed_state_point_size {
            let value = self.const_f32(size)?;
            self.builder
                .instruction(Op::Store, &[self.objects.output_point_size.0, value.0])?;
        }
        Ok(())
    }
    fn default_varying(&mut self, first: u32, count: u32) -> Result<Id> {
        if !(1..=4).contains(&count) || first + count > 4 {
            return Err(invariant("Invalid varying component group"));
        }
        let zero = self.const_f32(0.0)?;
        let one = self.const_f32(1.0)?;
        let mut values = [zero; 4];
        if first + count == 4 {
            values[count as usize - 1] = one;
        }
        if count == 1 {
            Ok(values[0])
        } else {
            self.builder.constant_composite(
                self.types.f32[count as usize - 1],
                &values[..count as usize],
            )
        }
    }
    fn prologue(&mut self) -> Result<()> {
        if self.stage == Stage::Fragment && self.runtime.dual_source_blend {
            let value = self.default_varying(0, 4)?;
            for output in self.frag_colors[..2].iter().copied() {
                if output != Id::NONE {
                    self.builder.instruction(Op::Store, &[output.0, value.0])?;
                }
            }
        }
        if self.stage == Stage::VertexB {
            let value = self.default_varying(0, 4)?;
            self.builder
                .instruction(Op::Store, &[self.objects.output_position.0, value.0])?;
            for i in 0..32 {
                if self.output_generics[i][0].components == 0 {
                    continue;
                }
                let mut element = 0;
                while element < 4 {
                    let info = self.output_generics[i][element];
                    let value = self.default_varying(element as u32, info.components)?;
                    self.builder.instruction(Op::Store, &[info.id.0, value.0])?;
                    element += info.components as usize;
                }
            }
            if self.objects.clip_distances != Id::NONE {
                for i in 0..self.profile.max_user_clip_distances.min(8) {
                    if self.clip_distance_written & (1 << i) != 0 {
                        continue;
                    }
                    let index = self.const_u32(i)?;
                    let pointer = self.op(
                        Op::AccessChain,
                        self.types.output_f32,
                        &[self.objects.clip_distances, index],
                    )?;
                    self.builder
                        .instruction(Op::Store, &[pointer.0, self.constants.f32_zero.0])?;
                }
            }
        }
        if matches!(self.stage, Stage::VertexB | Stage::Geometry) {
            self.fixed_point_size()?;
        }
        Ok(())
    }
    fn alpha_test(&mut self) -> Result<()> {
        let Some(comparison) = self.runtime.alpha_test_func else {
            return Ok(());
        };
        if comparison == CompareFunction::Always || self.frag_colors[0] == Id::NONE {
            return Ok(());
        }
        let color = self.op(Op::Load, self.types.f32[3], &[self.frag_colors[0]])?;
        let alpha = self.extract(self.types.f32[0], color, 3)?;
        let reference = self.const_f32(self.runtime.alpha_test_reference)?;
        let condition = if comparison == CompareFunction::Never {
            self.constants.false_value
        } else {
            let op = match comparison {
                CompareFunction::Less => Op::FOrdLessThan,
                CompareFunction::Equal => Op::FOrdEqual,
                CompareFunction::LessThanEqual => Op::FOrdLessThanEqual,
                CompareFunction::Greater => Op::FOrdGreaterThan,
                CompareFunction::NotEqual => Op::FOrdNotEqual,
                CompareFunction::GreaterThanEqual => Op::FOrdGreaterThanEqual,
                _ => unreachable!(),
            };
            self.op(op, self.types.bool, &[alpha, reference])?
        };
        let merge = self.builder.reserve_id()?;
        let discard = self.builder.reserve_id()?;
        self.builder
            .instruction(Op::SelectionMerge, &[merge.0, 0])?;
        self.builder
            .instruction(Op::BranchConditional, &[condition.0, merge.0, discard.0])?;
        self.builder.label(discard)?;
        self.builder.instruction(Op::Kill, &[])?;
        self.builder.label(merge)
    }
    pub(super) fn special_instruction(&mut self, ir: &Ir, id: InstId) -> Result<Option<Id>> {
        let opcode = ir.inst(id)?.opcode();
        let f = self.types.f32[0];
        let u = self.types.u32[0];
        let value = match opcode {
            O::Prologue => {
                self.prologue()?;
                Id::NONE
            }
            O::Epilogue => {
                if self.stage == Stage::VertexB
                    && self.runtime.convert_depth_mode
                    && !self.profile.support_native_ndc
                {
                    self.depth_mode()?;
                }
                if self.stage == Stage::Fragment {
                    self.alpha_test()?;
                }
                Id::NONE
            }
            O::DemoteToHelperInvocation => {
                if self.profile.support_demote_to_helper_invocation {
                    self.builder
                        .instruction(Op::DemoteToHelperInvocation, &[])?;
                } else {
                    let kill = self.builder.reserve_id()?;
                    let merge = self.builder.reserve_id()?;
                    self.builder
                        .instruction(Op::SelectionMerge, &[merge.0, 0])?;
                    self.builder.instruction(
                        Op::BranchConditional,
                        &[self.constants.true_value.0, kill.0, merge.0],
                    )?;
                    self.builder.label(kill)?;
                    self.builder.instruction(Op::Kill, &[])?;
                    self.builder.label(merge)?;
                }
                Id::NONE
            }
            O::EmitVertex | O::EndPrimitive => {
                if opcode == O::EmitVertex
                    && self.runtime.convert_depth_mode
                    && !self.profile.support_native_ndc
                {
                    self.depth_mode()?;
                }
                if !self.profile.support_geometry_streams {
                    return Err(unsupported("Geometry streams"));
                }
                let stream = ir.inst(id)?.arg(0)?;
                let stream = if ir.is_immediate(stream)? {
                    self.def(ir, stream)?
                } else {
                    self.constants.u32_zero
                };
                self.builder.instruction(
                    if opcode == O::EmitVertex {
                        Op::EmitStreamVertex
                    } else {
                        Op::EndStreamPrimitive
                    },
                    &[stream.0],
                )?;
                if opcode == O::EmitVertex {
                    self.fixed_point_size()?;
                }
                Id::NONE
            }
            O::Barrier | O::WorkgroupMemoryBarrier | O::DeviceMemoryBarrier => {
                let scope = self.const_u32(if opcode == O::DeviceMemoryBarrier {
                    Scope::Device
                } else {
                    Scope::Workgroup
                } as u32)?;
                let semantics = if opcode == O::Barrier {
                    M::ACQUIRE_RELEASE | M::WORKGROUP_MEMORY
                } else {
                    M::ACQUIRE_RELEASE
                        | M::UNIFORM_MEMORY
                        | M::WORKGROUP_MEMORY
                        | M::ATOMIC_COUNTER_MEMORY
                        | M::IMAGE_MEMORY
                };
                let semantics = self.const_u32(semantics.bits())?;
                if opcode == O::Barrier {
                    self.builder
                        .instruction(Op::ControlBarrier, &[scope.0, scope.0, semantics.0])?;
                } else {
                    self.builder
                        .instruction(Op::MemoryBarrier, &[scope.0, semantics.0])?;
                }
                Id::NONE
            }
            O::WorkgroupId => self.op(Op::Load, self.types.u32[2], &[self.objects.workgroup_id])?,
            O::LocalInvocationId => self.op(
                Op::Load,
                self.types.u32[2],
                &[self.objects.local_invocation_id],
            )?,
            O::InvocationId => self.op(Op::Load, u, &[self.objects.invocation_id])?,
            O::SampleId => self.op(Op::Load, u, &[self.objects.sample_id])?,
            O::IsHelperInvocation => self.op(
                Op::Load,
                self.types.bool,
                &[self.objects.is_helper_invocation],
            )?,
            O::InvocationInfo => {
                let vertices = match self.stage {
                    Stage::TessellationControl | Stage::TessellationEval => {
                        self.op(Op::Load, u, &[self.objects.patch_vertices_in])?
                    }
                    Stage::Geometry => self.const_u32(self.runtime.input_topology.vertices())?,
                    _ => return Ok(Some(self.const_u32(0x00ff0000)?)),
                };
                let shift = self.const_u32(16)?;
                self.op(Op::ShiftLeftLogical, u, &[vertices, shift])?
            }
            O::SR_WScaleFactorXY | O::SR_WScaleFactorZ => self.const_u32(0x00ff0000)?,
            O::YDirection => self.const_f32(if self.runtime.y_negate { -1.0 } else { 1.0 })?,
            O::ResolutionDownFactor => {
                if self.profile.unified_descriptor_binding {
                    let pointer_ty = self.builder.type_pointer(StorageClass::PushConstant, f)?;
                    let member = self.const_u32(2)?;
                    let pointer = self.op(
                        Op::AccessChain,
                        pointer_ty,
                        &[self.objects.rescaling_push_constants, member],
                    )?;
                    self.op(Op::Load, f, &[pointer])?
                } else {
                    let vector = self.op(
                        Op::Load,
                        self.types.f32[3],
                        &[self.objects.rescaling_uniform_constant],
                    )?;
                    self.extract(f, vector, 2)?
                }
            }
            O::RenderArea => {
                if !self.profile.unified_descriptor_binding {
                    return Err(unsupported(
                        "Render area requires unified descriptor bindings",
                    ));
                }
                let ty = self.types.f32[3];
                let pointer_ty = self.builder.type_pointer(StorageClass::PushConstant, ty)?;
                let pointer = self.op(
                    Op::AccessChain,
                    pointer_ty,
                    &[
                        self.objects.render_area_push_constant,
                        self.constants.u32_zero,
                    ],
                )?;
                self.op(Op::Load, ty, &[pointer])?
            }
            _ => return Ok(None),
        };
        Ok(Some(value))
    }
}
