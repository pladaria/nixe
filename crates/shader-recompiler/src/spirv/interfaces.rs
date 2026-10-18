use super::{
    EmitContext, Id,
    context::{InputGeneric, InputLoad, OutputGeneric},
};
use crate::{
    CompileError, Result,
    diagnostics::FailureCategory,
    ir::{Attribute, Program},
    types::{AttributeType, InputTopology, Interpolation, Stage},
};
use spirv::{BuiltIn as B, Capability as C, Decoration as D, StorageClass as S};

impl EmitContext<'_> {
    fn interface(
        &mut self,
        ty: Id,
        class: S,
        builtin: Option<B>,
        initial: Option<Id>,
    ) -> Result<Id> {
        let pointer = self.builder.type_pointer(class, ty)?;
        let id = self.builder.variable(pointer, class, initial)?;
        if let Some(builtin) = builtin {
            self.builder.decorate(id, D::BuiltIn, &[builtin as u32])?;
        }
        self.interfaces.push(id);
        Ok(id)
    }
    fn input(&mut self, mut ty: Id, per_vertex: bool, builtin: Option<B>) -> Result<Id> {
        let vertices = if per_vertex {
            match self.stage {
                Stage::TessellationControl | Stage::TessellationEval => Some(32),
                Stage::Geometry => Some(match self.runtime.input_topology {
                    InputTopology::Points => 1,
                    InputTopology::Lines => 2,
                    InputTopology::LinesAdjacency => 4,
                    InputTopology::Triangles => 3,
                    InputTopology::TrianglesAdjacency => 6,
                }),
                _ => None,
            }
        } else {
            None
        };
        if let Some(vertices) = vertices {
            let count = self.const_u32(vertices)?;
            ty = self.builder.type_array(ty, count)?;
        }
        self.interface(ty, S::Input, builtin, None)
    }
    fn output(
        &mut self,
        mut ty: Id,
        invocations: Option<u32>,
        builtin: Option<B>,
        initial: Option<Id>,
    ) -> Result<Id> {
        if self.stage == Stage::TessellationControl
            && let Some(invocations) = invocations
        {
            let count = self.const_u32(invocations)?;
            ty = self.builder.type_array(ty, count)?;
        }
        self.interface(ty, S::Output, builtin, initial)
    }
    fn attribute_type(&self, ty: AttributeType) -> Result<(Id, Id, Id, InputLoad)> {
        Ok(match ty {
            AttributeType::Float => (
                self.types.f32[3],
                self.types.f32[0],
                self.types.input_f32,
                InputLoad::None,
            ),
            AttributeType::SignedInt => (
                self.types.s32[3],
                self.types.s32[0],
                self.types.input_s32,
                InputLoad::Bitcast,
            ),
            AttributeType::UnsignedInt => (
                self.types.u32[3],
                self.types.u32[0],
                self.types.input_u32,
                InputLoad::Bitcast,
            ),
            AttributeType::SignedScaled | AttributeType::UnsignedScaled
                if self.profile.support_scaled_attributes =>
            {
                (
                    self.types.f32[3],
                    self.types.f32[0],
                    self.types.input_f32,
                    InputLoad::None,
                )
            }
            AttributeType::SignedScaled => (
                self.types.s32[3],
                self.types.s32[0],
                self.types.input_s32,
                InputLoad::SignedToFloat,
            ),
            AttributeType::UnsignedScaled => (
                self.types.u32[3],
                self.types.u32[0],
                self.types.input_u32,
                InputLoad::UnsignedToFloat,
            ),
            AttributeType::Disabled => {
                return Err(CompileError::new(
                    FailureCategory::InvalidArgument,
                    "Disabled interface attribute type",
                ));
            }
        })
    }
    pub(super) fn define_inputs(&mut self, p: &Program) -> Result<()> {
        let info = &p.info;
        let mut loads = info.loads.clone();
        for i in 0..8 {
            loads.mask[i] |= info.passthrough.mask[i];
        }
        macro_rules! input {
            ($field:ident,$ty:expr,$per:expr,$builtin:ident) => {
                self.objects.$field = self.input($ty, $per, Some(B::$builtin))?;
            };
        }
        macro_rules! flat {
            ($field:ident) => {
                if self.stage == Stage::Fragment {
                    self.builder.decorate(self.objects.$field, D::Flat, &[])?;
                }
            };
        }
        if info.uses_workgroup_id {
            input!(workgroup_id, self.types.u32[2], false, WorkgroupId);
        }
        if info.uses_local_invocation_id {
            input!(
                local_invocation_id,
                self.types.u32[2],
                false,
                LocalInvocationId
            );
        }
        if info.uses_invocation_id {
            input!(invocation_id, self.types.u32[0], false, InvocationId);
        }
        if info.uses_invocation_info
            && matches!(
                self.stage,
                Stage::TessellationControl | Stage::TessellationEval
            )
        {
            input!(patch_vertices_in, self.types.u32[0], false, PatchVertices);
        }
        if info.uses_sample_id {
            input!(sample_id, self.types.u32[0], false, SampleId);
            flat!(sample_id);
        }
        if info.uses_is_helper_invocation {
            input!(
                is_helper_invocation,
                self.types.bool,
                false,
                HelperInvocation
            );
        }
        if info.uses_subgroup_mask && self.profile.supports_subgroup_stage(self.stage) {
            input!(subgroup_mask_eq, self.types.u32[3], false, SubgroupEqMask);
            input!(subgroup_mask_lt, self.types.u32[3], false, SubgroupLtMask);
            input!(subgroup_mask_le, self.types.u32[3], false, SubgroupLeMask);
            input!(subgroup_mask_gt, self.types.u32[3], false, SubgroupGtMask);
            input!(subgroup_mask_ge, self.types.u32[3], false, SubgroupGeMask);
            flat!(subgroup_mask_eq);
            flat!(subgroup_mask_lt);
            flat!(subgroup_mask_le);
            flat!(subgroup_mask_gt);
            flat!(subgroup_mask_ge);
        }
        if (info.uses_fswzadd
            || info.uses_subgroup_invocation_id
            || info.uses_subgroup_shuffles
            || self.profile.warp_size_potentially_larger_than_guest
                && (info.uses_subgroup_vote || info.uses_subgroup_mask))
            && self.profile.supports_subgroup_stage(self.stage)
        {
            self.builder.capability(C::GroupNonUniform)?;
            input!(
                subgroup_local_invocation_id,
                self.types.u32[0],
                false,
                SubgroupLocalInvocationId
            );
            flat!(subgroup_local_invocation_id);
        }
        if info.uses_fswzadd {
            let one = self.const_f32(1.0)?;
            let negative = self.const_f32(-1.0)?;
            let zero = self.const_f32(0.0)?;
            self.objects.fswzadd_lut_a = self
                .builder
                .constant_composite(self.types.f32[3], &[negative, one, negative, zero])?;
            self.objects.fswzadd_lut_b = self
                .builder
                .constant_composite(self.types.f32[3], &[negative, negative, one, negative])?;
        }
        if loads.get(Attribute::PrimitiveId) {
            input!(primitive_id, self.types.u32[0], false, PrimitiveId);
            flat!(primitive_id);
        }
        if loads.get(Attribute::Layer) {
            self.builder.capability(C::Geometry)?;
            input!(layer, self.types.u32[0], false, Layer);
            self.builder.decorate(self.objects.layer, D::Flat, &[])?;
        }
        if loads.any_component(Attribute::PositionX) {
            if self.stage != Stage::Fragment && self.profile.has_broken_spirv_position_input {
                self.need_input_position_indirect = true;
                let structure = self.builder.type_struct(&[self.types.f32[3]])?;
                self.objects.input_position = self.input(structure, true, None)?;
                self.builder
                    .member_decorate(structure, 0, D::BuiltIn, &[B::Position as u32])?;
                self.builder.decorate(structure, D::Block, &[])?;
            } else {
                let builtin = if self.stage == Stage::Fragment {
                    B::FragCoord
                } else {
                    B::Position
                };
                self.objects.input_position = self.input(self.types.f32[3], true, Some(builtin))?;
                if self.profile.support_geometry_shader_passthrough
                    && info.passthrough.any_component(Attribute::PositionX)
                {
                    self.builder
                        .decorate(self.objects.input_position, D::PassthroughNV, &[])?;
                }
            }
        }
        if loads.get(Attribute::InstanceId) {
            if self.profile.support_vertex_instance_id {
                input!(instance_id, self.types.u32[0], true, InstanceId);
                if loads.get(Attribute::BaseInstance) {
                    input!(base_instance, self.types.u32[0], true, BaseInstance);
                }
            } else {
                input!(instance_index, self.types.u32[0], true, InstanceIndex);
                input!(base_instance, self.types.u32[0], true, BaseInstance);
            }
        } else if loads.get(Attribute::BaseInstance) {
            input!(base_instance, self.types.u32[0], true, BaseInstance);
        }
        if loads.get(Attribute::VertexId) {
            if self.profile.support_vertex_instance_id {
                input!(vertex_id, self.types.u32[0], true, VertexId);
                if loads.get(Attribute::BaseVertex) {
                    input!(base_vertex, self.types.u32[0], true, BaseVertex);
                }
            } else {
                input!(vertex_index, self.types.u32[0], true, VertexIndex);
                input!(base_vertex, self.types.u32[0], true, BaseVertex);
            }
        } else if loads.get(Attribute::BaseVertex) {
            input!(base_vertex, self.types.u32[0], true, BaseVertex);
        }
        if loads.get(Attribute::DrawID) {
            input!(draw_index, self.types.u32[0], true, DrawIndex);
        }
        if loads.get(Attribute::FrontFace) {
            input!(front_face, self.types.bool, true, FrontFacing);
        }
        if loads.get(Attribute::PointSpriteS) || loads.get(Attribute::PointSpriteT) {
            input!(point_coord, self.types.f32[1], true, PointCoord);
        }
        if loads.get(Attribute::TessellationEvaluationPointU)
            || loads.get(Attribute::TessellationEvaluationPointV)
        {
            input!(tess_coord, self.types.f32[2], false, TessCoord);
        }
        for index in 0..32 {
            let ty = self.runtime.generic_input_types[index];
            if !self.runtime.previous_stage_stores.generic(index)
                || !loads.generic(index)
                || ty == AttributeType::Disabled
            {
                continue;
            }
            let (vector, component, pointer, load) = self.attribute_type(ty)?;
            let id = self.input(vector, true, None)?;
            self.builder.decorate(id, D::Location, &[index as u32])?;
            self.input_generics[index] = InputGeneric {
                id,
                pointer,
                component,
                load,
            };
            if info.passthrough.generic(index) && self.profile.support_geometry_shader_passthrough {
                self.builder.decorate(id, D::PassthroughNV, &[])?;
            }
            if self.stage == Stage::Fragment {
                let decoration =
                    if matches!(ty, AttributeType::SignedInt | AttributeType::UnsignedInt) {
                        Some(D::Flat)
                    } else {
                        match info.interpolation[index] {
                            Interpolation::Smooth => None,
                            Interpolation::Flat => Some(D::Flat),
                            Interpolation::NoPerspective => Some(D::NoPerspective),
                        }
                    };
                if let Some(decoration) = decoration {
                    self.builder.decorate(id, decoration, &[])?;
                }
            }
        }
        if self.stage == Stage::TessellationEval {
            for index in 0..30 {
                if info.uses_patches[index] {
                    let id = self.input(self.types.f32[3], false, None)?;
                    self.builder.decorate(id, D::Patch, &[])?;
                    self.builder.decorate(id, D::Location, &[index as u32])?;
                    self.patches[index] = id;
                }
            }
        }
        Ok(())
    }
    fn generic_output(&mut self, index: usize, invocations: u32) -> Result<()> {
        let mut element = 0;
        while element < 4 {
            let offset = Attribute::generic(index, element).unwrap().index();
            let xfb = (offset < (self.runtime.xfb_count as usize))
                .then(|| self.runtime.xfb_varyings[offset])
                .filter(|v| v.components > 0);
            let count = xfb.map_or(4 - element, |v| v.components as usize);
            if count > 4 - element {
                return Err(CompileError::new(
                    FailureCategory::InvalidArgument,
                    "Transform feedback varying exceeds a generic vector",
                ));
            }
            let id = self.output(self.types.f32[count - 1], Some(invocations), None, None)?;
            self.builder.decorate(id, D::Location, &[index as u32])?;
            if element > 0 {
                self.builder.decorate(id, D::Component, &[element as u32])?;
            }
            if let Some(xfb) = xfb {
                self.builder.decorate(id, D::XfbBuffer, &[xfb.buffer])?;
                self.builder.decorate(id, D::XfbStride, &[xfb.stride])?;
                self.builder.decorate(id, D::Offset, &[xfb.offset])?;
                if self.stage == Stage::Geometry && xfb.stream != 0 {
                    self.builder.decorate(id, D::Stream, &[xfb.stream])?;
                }
            }
            self.output_generics[index][element..element + count].fill(OutputGeneric {
                id,
                first: element as u32,
                components: count as u32,
            });
            element += count;
        }
        Ok(())
    }
    pub(super) fn define_outputs(&mut self, p: &Program) -> Result<()> {
        let info = &p.info;
        let invocations = Some(p.execution.invocations);
        if self.runtime.convert_depth_mode
            || info.stores.any_component(Attribute::PositionX)
            || self.stage == Stage::VertexB
        {
            self.objects.output_position =
                self.output(self.types.f32[3], invocations, Some(B::Position), None)?;
        }
        if info.stores.get(Attribute::PointSize) || self.runtime.fixed_state_point_size.is_some() {
            self.reject_fragment_output("PointSize")?;
            self.objects.output_point_size =
                self.output(self.types.f32[0], invocations, Some(B::PointSize), None)?;
        }
        if info.stores.clip_distances() {
            self.reject_fragment_output("ClipDistance")?;
            if self.profile.max_user_clip_distances > 0 {
                let used = self.profile.max_user_clip_distances.min(8);
                let count = self.const_u32(used)?;
                let ty = self.builder.type_array(self.types.f32[0], count)?;
                let initial = self
                    .builder
                    .constant_composite(ty, &vec![self.constants.f32_zero; used as usize])?;
                self.objects.clip_distances =
                    self.output(ty, invocations, Some(B::ClipDistance), Some(initial))?;
            }
        }
        if self.profile.support_viewport_index_layer_non_geometry || self.stage == Stage::Geometry {
            if info.stores.get(Attribute::Layer) {
                self.reject_fragment_output("Layer")?;
                self.objects.layer =
                    self.output(self.types.u32[0], invocations, Some(B::Layer), None)?;
            }
            if info.stores.get(Attribute::ViewportIndex) {
                self.reject_fragment_output("ViewportIndex")?;
                self.objects.viewport_index =
                    self.output(self.types.u32[0], invocations, Some(B::ViewportIndex), None)?;
            }
        }
        if info.stores.get(Attribute::ViewportMask) && self.profile.support_viewport_mask {
            let count = self.const_u32(1)?;
            let ty = self.builder.type_array(self.types.u32[0], count)?;
            self.objects.viewport_mask = self.output(ty, None, Some(B::ViewportMaskNV), None)?;
        }
        for index in 0..32 {
            if info.stores.generic(index) {
                self.generic_output(index, p.execution.invocations)?;
            }
        }
        match self.stage {
            Stage::TessellationControl => {
                for (enabled, count, builtin) in [
                    (info.stores_tess_level_outer, 4, B::TessLevelOuter),
                    (info.stores_tess_level_inner, 2, B::TessLevelInner),
                ] {
                    if !enabled {
                        continue;
                    }
                    let count = self.const_u32(count)?;
                    let ty = self.builder.type_array(self.types.f32[0], count)?;
                    let id = self.output(ty, None, Some(builtin), None)?;
                    self.builder.decorate(id, D::Patch, &[])?;
                    if builtin == B::TessLevelOuter {
                        self.objects.output_tess_level_outer = id;
                    } else {
                        self.objects.output_tess_level_inner = id;
                    }
                }
                for index in 0..30 {
                    if info.uses_patches[index] {
                        let id = self.output(self.types.f32[3], None, None, None)?;
                        self.builder.decorate(id, D::Patch, &[])?;
                        self.builder.decorate(id, D::Location, &[index as u32])?;
                        self.patches[index] = id;
                    }
                }
            }
            Stage::Fragment => {
                for index in 0..8 {
                    let dual = self.runtime.dual_source_blend && index <= 1;
                    if !dual
                        && !info.stores_frag_color[index]
                        && !self.profile.need_declared_frag_colors
                    {
                        continue;
                    }
                    let (ty, _, _, _) =
                        self.attribute_type(self.runtime.color_output_types[index])?;
                    let id = self.output(ty, None, None, None)?;
                    self.frag_colors[index] = id;
                    self.builder.decorate(
                        id,
                        D::Location,
                        &[if dual { 0 } else { index as u32 }],
                    )?;
                    if dual {
                        self.builder.decorate(id, D::Index, &[index as u32])?;
                    }
                }
                if info.stores_frag_depth {
                    self.objects.frag_depth =
                        self.output(self.types.f32[0], None, Some(B::FragDepth), None)?;
                }
                if info.stores_sample_mask {
                    let count = self.const_u32(1)?;
                    let ty = self.builder.type_array(self.types.u32[0], count)?;
                    self.objects.sample_mask = self.output(ty, None, Some(B::SampleMask), None)?;
                }
            }
            _ => (),
        }
        Ok(())
    }
    fn reject_fragment_output(&self, name: &str) -> Result<()> {
        if self.stage == Stage::Fragment {
            Err(CompileError::new(
                FailureCategory::Unsupported,
                format!("Storing {name} in fragment stage"),
            ))
        } else {
            Ok(())
        }
    }
}
