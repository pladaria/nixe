use super::{EmitContext, Id};
use crate::{
    CompileError, Result,
    diagnostics::FailureCategory,
    ir::{Attribute as A, Program},
    types::{InputTopology as I, OutputTopology as O, Stage, TessPrimitive as T, TessSpacing as S},
};
use spirv::{Capability as C, ExecutionMode as M, ExecutionModel as E};
impl EmitContext<'_> {
    pub(super) fn entry_point(&mut self, p: &Program, main: Id) -> Result<()> {
        let model = match self.stage {
            Stage::Compute => {
                self.builder
                    .execution_mode(main, M::LocalSize, &p.execution.workgroup_size)?;
                E::GLCompute
            }
            Stage::VertexB => E::Vertex,
            Stage::TessellationControl => {
                self.builder.capability(C::Tessellation)?;
                self.builder
                    .execution_mode(main, M::OutputVertices, &[p.execution.invocations])?;
                E::TessellationControl
            }
            Stage::TessellationEval => {
                self.builder.capability(C::Tessellation)?;
                self.builder.execution_mode(
                    main,
                    match self.runtime.tess_primitive {
                        T::Isolines => M::Isolines,
                        T::Triangles => M::Triangles,
                        T::Quads => M::Quads,
                    },
                    &[],
                )?;
                self.builder.execution_mode(
                    main,
                    match self.runtime.tess_spacing {
                        S::Equal => M::SpacingEqual,
                        S::FractionalOdd => M::SpacingFractionalOdd,
                        S::FractionalEven => M::SpacingFractionalEven,
                    },
                    &[],
                )?;
                self.builder.execution_mode(
                    main,
                    if self.runtime.tess_clockwise {
                        M::VertexOrderCw
                    } else {
                        M::VertexOrderCcw
                    },
                    &[],
                )?;
                E::TessellationEvaluation
            }
            Stage::Geometry => {
                self.builder.capability(C::Geometry)?;
                if self.profile.support_geometry_streams {
                    self.builder.capability(C::GeometryStreams)?;
                }
                self.builder.execution_mode(
                    main,
                    match self.runtime.input_topology {
                        I::Points => M::InputPoints,
                        I::Lines => M::InputLines,
                        I::LinesAdjacency => M::InputLinesAdjacency,
                        I::Triangles => M::Triangles,
                        I::TrianglesAdjacency => M::InputTrianglesAdjacency,
                    },
                    &[],
                )?;
                let output = p.execution.output_topology.ok_or_else(|| {
                    CompileError::new(
                        FailureCategory::InvalidArgument,
                        "Geometry shader has no output topology",
                    )
                })?;
                self.builder.execution_mode(
                    main,
                    match output {
                        O::PointList => M::OutputPoints,
                        O::LineStrip => M::OutputLineStrip,
                        O::TriangleStrip => M::OutputTriangleStrip,
                    },
                    &[],
                )?;
                if p.info.stores.get(A::PointSize) {
                    self.builder.capability(C::GeometryPointSize)?;
                }
                self.builder.execution_mode(
                    main,
                    M::OutputVertices,
                    &[p.execution.output_vertices],
                )?;
                self.builder
                    .execution_mode(main, M::Invocations, &[p.execution.invocations])?;
                if p.execution.is_geometry_passthrough
                    && self.profile.support_geometry_shader_passthrough
                {
                    self.builder
                        .extension("SPV_NV_geometry_shader_passthrough")?;
                    self.builder.capability(C::GeometryShaderPassthroughNV)?;
                }
                E::Geometry
            }
            Stage::Fragment => {
                self.builder.execution_mode(
                    main,
                    if self.profile.lower_left_origin_mode {
                        M::OriginLowerLeft
                    } else {
                        M::OriginUpperLeft
                    },
                    &[],
                )?;
                if p.info.stores_frag_depth {
                    self.builder.execution_mode(main, M::DepthReplacing, &[])?;
                }
                if self.runtime.force_early_z {
                    self.builder
                        .execution_mode(main, M::EarlyFragmentTests, &[])?;
                }
                if self.profile.support_shader_quad_control && p.info.uses_quad_shuffles {
                    self.builder.extension("SPV_KHR_quad_control")?;
                    self.builder.capability(C::QuadControlKHR)?;
                    self.builder
                        .execution_mode(main, M::RequireFullQuadsKHR, &[])?;
                }
                E::Fragment
            }
            _ => {
                return Err(CompileError::new(
                    FailureCategory::UnsupportedReferenceCase,
                    "Vertex A must be merged before SPIR-V emission",
                ));
            }
        };
        self.builder
            .entry_point(model, main, "main", &self.interfaces)
    }
    fn float_controls(&mut self, p: &Program, main: Id) -> Result<()> {
        let info = &p.info;
        let profile = self.profile;
        self.builder.extension("SPV_KHR_float_controls")?;
        for (width, flush, preserve, can_flush, can_preserve) in [
            (
                32,
                info.uses_fp32_denorms_flush,
                info.uses_fp32_denorms_preserve,
                profile.support_fp32_denorm_flush && !profile.has_broken_fp32_denorm_flush,
                profile.support_fp32_denorm_preserve,
            ),
            (
                16,
                info.uses_fp16_denorms_flush,
                info.uses_fp16_denorms_preserve,
                profile.support_fp16_denorm_flush,
                profile.support_fp16_denorm_preserve,
            ),
        ] {
            if width == 16
                && (!profile.support_separate_denorm_behavior
                    || profile.has_broken_fp16_float_controls)
            {
                continue;
            }
            if flush && !preserve && can_flush {
                self.builder.capability(C::DenormFlushToZero)?;
                self.builder
                    .execution_mode(main, M::DenormFlushToZero, &[width])?;
            }
            if preserve && !flush && can_preserve {
                self.builder.capability(C::DenormPreserve)?;
                self.builder
                    .execution_mode(main, M::DenormPreserve, &[width])?;
            }
        }
        if profile.has_broken_fp16_float_controls && info.uses_fp16 {
            return Ok(());
        }
        for (width, enabled) in [
            (
                16,
                info.uses_fp16 && profile.support_fp16_signed_zero_nan_preserve,
            ),
            (32, profile.support_fp32_signed_zero_nan_preserve),
            (
                64,
                info.uses_fp64 && profile.support_fp64_signed_zero_nan_preserve,
            ),
        ] {
            if enabled {
                self.builder.capability(C::SignedZeroInfNanPreserve)?;
                self.builder
                    .execution_mode(main, M::SignedZeroInfNanPreserve, &[width])?;
            }
        }
        Ok(())
    }
    pub(super) fn setup_capabilities(&mut self, p: &Program, main: Id) -> Result<()> {
        let info = &p.info;
        let profile = self.profile;
        if profile.support_float_controls {
            self.float_controls(p, main)?;
        }
        for (enabled, cap) in [
            (info.uses_sampled_1d, C::Sampled1D),
            (info.uses_image_1d, C::Image1D),
            (info.uses_sparse_residency, C::SparseResidency),
            (
                info.stores.get(A::ViewportIndex) && profile.support_multi_viewport,
                C::MultiViewport,
            ),
            (
                info.uses_int64_bit_atomics && profile.support_int64_atomics,
                C::Int64Atomics,
            ),
            (
                info.uses_typeless_image_reads && profile.support_typeless_image_loads,
                C::StorageImageReadWithoutFormat,
            ),
            (
                info.uses_typeless_image_writes,
                C::StorageImageWriteWithoutFormat,
            ),
            (info.uses_image_buffers, C::ImageBuffer),
            (info.uses_sample_id, C::SampleRateShading),
            (info.uses_derivatives, C::DerivativeControl),
            (true, C::ImageGatherExtended),
            (true, C::ImageQuery),
            (true, C::SampledBuffer),
        ] {
            if enabled {
                self.builder.capability(cap)?;
            }
        }
        if info.uses_demote_to_helper_invocation && profile.support_demote_to_helper_invocation {
            if profile.supported_spirv < 0x00010600 {
                self.builder
                    .extension("SPV_EXT_demote_to_helper_invocation")?;
            }
            self.builder.capability(C::DemoteToHelperInvocation)?;
        }
        if info.stores.get(A::ViewportMask) && profile.support_viewport_mask {
            self.builder.extension("SPV_NV_viewport_array2")?;
            self.builder.capability(C::ShaderViewportMaskNV)?;
        }
        if (info.stores.get(A::Layer) || info.stores.get(A::ViewportIndex))
            && profile.support_viewport_index_layer_non_geometry
            && self.stage != Stage::Geometry
        {
            self.builder
                .extension("SPV_EXT_shader_viewport_index_layer")?;
            self.builder.capability(C::ShaderViewportIndexLayerEXT)?;
        }
        if !profile.support_vertex_instance_id
            && (info.loads.get(A::InstanceId) || info.loads.get(A::VertexId))
        {
            self.builder.extension("SPV_KHR_shader_draw_parameters")?;
            self.builder.capability(C::DrawParameters)?;
        }
        if (info.uses_subgroup_vote
            || info.uses_subgroup_invocation_id
            || info.uses_subgroup_shuffles
            || info.uses_subgroup_mask)
            && profile.support_vote
            && profile.supports_subgroup_stage(self.stage)
        {
            self.builder.capability(C::GroupNonUniformBallot)?;
            self.builder.capability(C::GroupNonUniformShuffle)?;
            if info.uses_subgroup_shuffles && profile.support_shuffle_relative {
                self.builder.capability(C::GroupNonUniformShuffleRelative)?;
            }
            if !profile.warp_size_potentially_larger_than_guest {
                self.builder.capability(C::GroupNonUniformVote)?;
            }
        }
        if info.uses_quad_shuffles {
            if profile.support_quad_shuffles {
                self.builder.capability(C::GroupNonUniformQuad)?;
            }
            self.builder.capability(C::GroupNonUniformShuffle)?;
        }
        if !self.non_uniform_ids.is_empty() {
            if profile.supported_spirv < 0x00010500 {
                self.builder.extension("SPV_EXT_descriptor_indexing")?;
            }
            self.builder.capability(C::ShaderNonUniform)?;
            for (enabled, cap) in self.uses_nonuniform.into_iter().zip([
                C::SampledImageArrayNonUniformIndexing,
                C::StorageImageArrayNonUniformIndexing,
                C::UniformTexelBufferArrayNonUniformIndexing,
                C::StorageTexelBufferArrayNonUniformIndexing,
            ]) {
                if enabled {
                    self.builder.capability(cap)?;
                }
            }
        }
        if self.runtime.xfb_count != 0 {
            self.builder.capability(C::TransformFeedback)?;
            self.builder.execution_mode(main, M::Xfb, &[])?;
        }
        Ok(())
    }
}
