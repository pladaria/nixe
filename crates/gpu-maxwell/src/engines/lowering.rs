//! Consumed Maxwell graphics state and native frontend preparation.
mod buffer;
mod color;
mod indexed;
mod multisample;
pub(crate) mod native;
mod raster;
mod resolve;
use super::threed::*;
use crate::MaxwellMethodSource;
use nixe_gpu::*;
use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

#[derive(Clone, Debug)]
struct DrawAttachmentSelection {
    colors: Vec<(u8, MaxwellThreeDResourceRole)>,
    color_outputs: [nixe_gpu::ColorOutputState; 8],
    depth_stencil: Option<MaxwellThreeDResourceRole>,
}

impl DrawAttachmentSelection {
    fn attachment_indices_iter(&self) -> impl Iterator<Item = MaxwellThreeDResourceRole> + '_ {
        self.colors
            .iter()
            .map(|(_, index)| *index)
            .chain(self.depth_stencil)
    }

    fn color_targets(&self) -> impl Iterator<Item = u8> + '_ {
        self.colors.iter().map(|(target, _)| *target)
    }
}

/// One execution trigger retained at its exact method location.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDOperationTrigger {
    ClearSurface {
        source: MaxwellMethodSource,
    },
    DrawVertexArray {
        source: MaxwellMethodSource,
        vertex_count: u32,
    },
    DrawIndexBuffer {
        source: MaxwellMethodSource,
        index_count: u32,
    },
}

impl MaxwellThreeDOperationTrigger {
    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        match self {
            Self::ClearSurface { source }
            | Self::DrawVertexArray { source, .. }
            | Self::DrawIndexBuffer { source, .. } => source,
        }
    }

    pub(crate) const fn is_draw(self) -> bool {
        matches!(
            self,
            Self::DrawVertexArray { .. } | Self::DrawIndexBuffer { .. }
        )
    }

    const fn is_indexed(self) -> bool {
        matches!(self, Self::DrawIndexBuffer { .. })
    }

    /// Appends the resources consumed directly by this trigger.
    ///
    /// Shader translation contributes constant-buffer, texture, and sampler
    /// roles separately. Keeping the trigger-specific selection exhaustive
    /// ensures that indexed draws also resolve the index storage.
    pub(crate) fn append_resource_roles(
        self,
        state: &MaxwellThreeDState,
        roles: &mut Vec<MaxwellThreeDResourceRole>,
    ) {
        match self {
            Self::ClearSurface { .. } => {
                if let Some(surface) = state.render_targets().clear().last_surface().value() {
                    if surface.color_mask() != 0 {
                        roles.push(MaxwellThreeDResourceRole::ColorTarget(
                            surface.color_target(),
                        ));
                    }
                    if surface.depth() || surface.stencil() {
                        roles.push(MaxwellThreeDResourceRole::DepthStencilTarget);
                    }
                }
            }
            Self::DrawVertexArray { .. } | Self::DrawIndexBuffer { .. } => {
                if self.is_indexed() && state.inline_index().is_none() {
                    roles.push(MaxwellThreeDResourceRole::IndexBuffer);
                }
                roles.extend(
                    consumed_vertex_streams(state).map(MaxwellThreeDResourceRole::VertexStream),
                );
                if let Some(selection) = state.render_targets().color_target_selection().value() {
                    roles.extend(
                        selection
                            .active_targets()
                            .iter()
                            .copied()
                            .map(MaxwellThreeDResourceRole::ColorTarget),
                    );
                }
                if draw_depth_stencil_resource_required(state) {
                    roles.push(MaxwellThreeDResourceRole::DepthStencilTarget);
                }
            }
        }
    }
}

/// A stream's enable bit alone does not cause vertex fetches. Draws can leave
/// old streams configured while disabling all their attributes (for example,
/// when a subsequent shader generates geometry from the vertex ID).
fn consumed_vertex_streams(state: &MaxwellThreeDState) -> impl Iterator<Item = u8> {
    let mut mask = state.vertex_input().attributes().iter().enumerate().fold(
        0_u32,
        |mask, (index, attribute)| match attribute.value().filter(|attribute| {
            attribute.enabled() && state.vertex_input().attribute_skip_mask(index as u8) != 15
        }) {
            Some(attribute) => mask | (1 << attribute.stream()),
            None => mask,
        },
    );
    std::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let stream = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        Some(stream)
    })
}

/// The binding source supplies declared sizes when no smaller bound is proved.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum MaxwellShaderBufferSize {
    Proven(u64),
    Declared,
}

impl MaxwellShaderBufferSize {
    fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::Proven(a), Self::Proven(b)) => Self::Proven(a.max(b)),
            _ => Self::Declared,
        }
    }
}

fn validate_draw_iterated_blend_state(
    state: &MaxwellThreeDState,
    attachments: &DrawAttachmentSelection,
) -> Result<(), MaxwellLoweringError> {
    if attachments.colors.is_empty() {
        return Ok(());
    }
    let controls = state.fixed_function().blend_controls();
    let Some(value) = controls
        .iterated_blend()
        .value()
        .copied()
        .filter(|value| value.enabled())
    else {
        return Ok(());
    };
    Err(MaxwellLoweringError::UnsupportedIteratedBlendSemantics {
        value,
        pass_count: controls
            .iterated_blend_pass_count()
            .value()
            .map(|value| value.pass_count()),
    })
}

fn validate_draw_logic_op_state(
    state: &MaxwellThreeDState,
    attachments: &DrawAttachmentSelection,
) -> Result<(), MaxwellLoweringError> {
    if attachments.colors.is_empty()
        || state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::LogicOpEnable)
            .value()
            != Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        return Ok(());
    }
    let function = match state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::LogicOpFunction)
        .value()
    {
        Some(MaxwellThreeDFixedFunctionValue::LogicOp(value)) => *value,
        None => return Err(MaxwellLoweringError::IncompleteLogicOpState),
        Some(_) => {
            return Err(MaxwellLoweringError::ContradictoryState {
                reason: "logic-operation function register has the wrong typed value",
            });
        }
    };
    Err(MaxwellLoweringError::UnsupportedLogicOpSemantics(function))
}

fn draw_alpha_test_state(
    state: &MaxwellThreeDState,
) -> Result<Option<AlphaTest>, MaxwellLoweringError> {
    let fixed = state.fixed_function();
    if fixed
        .register(MaxwellThreeDFixedFunctionRegister::AlphaTestEnable)
        .value()
        != Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        return Ok(None);
    }
    let reference = match fixed
        .register(MaxwellThreeDFixedFunctionRegister::AlphaTestReference)
        .value()
    {
        Some(MaxwellThreeDFixedFunctionValue::FloatBits(value)) => *value,
        None => {
            return Err(MaxwellLoweringError::IncompleteAlphaTestState("reference"));
        }
        Some(_) => {
            return Err(MaxwellLoweringError::ContradictoryState {
                reason: "alpha-test reference register has the wrong typed value",
            });
        }
    };
    let function = match fixed
        .register(MaxwellThreeDFixedFunctionRegister::AlphaTestFunction)
        .value()
    {
        Some(MaxwellThreeDFixedFunctionValue::Compare(value)) => *value,
        None => {
            return Err(MaxwellLoweringError::IncompleteAlphaTestState("function"));
        }
        Some(_) => {
            return Err(MaxwellLoweringError::ContradictoryState {
                reason: "alpha-test function register has the wrong typed value",
            });
        }
    };
    let comparison = match function {
        MaxwellThreeDCompareOp::Never => AlphaCompareOperation::Never,
        MaxwellThreeDCompareOp::Less => AlphaCompareOperation::Less,
        MaxwellThreeDCompareOp::Equal => AlphaCompareOperation::Equal,
        MaxwellThreeDCompareOp::LessEqual => AlphaCompareOperation::LessEqual,
        MaxwellThreeDCompareOp::Greater => AlphaCompareOperation::Greater,
        MaxwellThreeDCompareOp::NotEqual => AlphaCompareOperation::NotEqual,
        MaxwellThreeDCompareOp::GreaterEqual => AlphaCompareOperation::GreaterEqual,
        MaxwellThreeDCompareOp::Always => AlphaCompareOperation::Always,
    };
    Ok(Some(AlphaTest {
        comparison,
        reference_bits: reference.get(),
    }))
}

fn validate_direct_line_rasterization_state(
    state: &MaxwellThreeDState,
) -> Result<(), MaxwellLoweringError> {
    let direct_line_primitive =
        state.generated_primitive() == Some(super::threed::state::GeneratedPrimitive::Lines);
    if !direct_line_primitive {
        return Ok(());
    }

    if state.line().anti_aliased_line_enable().value()
        == Some(&MaxwellThreeDAntiAliasedLineEnable::Enabled)
    {
        return raster::smooth_line(state).map(|_| ());
    }
    if state.line().stipple_enable().value() == Some(&true) {
        let parameters = state.line().stipple_parameters().value().copied().ok_or(
            MaxwellLoweringError::IncompleteDraw("SET_LINE_STIPPLE_PARAMETERS"),
        )?;
        return Err(MaxwellLoweringError::UnsupportedLineStippleSemantics {
            factor: parameters.factor(),
            pattern: parameters.pattern(),
        });
    }
    match state.line().aliased_line_width_enable().value() {
        None => Err(MaxwellLoweringError::IncompleteDraw(
            "SET_ALIASED_LINE_WIDTH_ENABLE",
        )),
        Some(MaxwellThreeDAliasedLineWidthEnable::Disabled) => state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::LineWidth)
            .value()
            .is_some()
            .then_some(())
            .ok_or(MaxwellLoweringError::IncompleteDraw("SET_LINE_WIDTH_FLOAT")),
        Some(MaxwellThreeDAliasedLineWidthEnable::Enabled) => {
            Err(MaxwellLoweringError::UnsupportedAliasedLineWidthSemantics)
        }
    }
}

/// An enabled mask of 0xff still overwrites the entire eight-bit stencil
/// aspect. Use the operation snapshot: deko3d restores SET_STENCIL_MASK via
/// MME shadow replay immediately after issuing the clear.
/// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
/// https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/gpu_3d_base.cpp#L470-L511
fn clear_stencil_mask_is_full(state: &MaxwellThreeDState) -> Result<bool, MaxwellLoweringError> {
    if !state
        .render_targets()
        .clear()
        .surface_control()
        .value()
        .is_some_and(|control| control.respect_stencil_mask())
    {
        return Ok(true);
    }
    match state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::FrontStencilWriteMask)
        .value()
    {
        Some(MaxwellThreeDFixedFunctionValue::Mask(mask)) => Ok(*mask == 0xff),
        _ => Err(MaxwellLoweringError::IncompleteClear("SET_STENCIL_MASK")),
    }
}

fn select_draw_attachments(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
) -> Result<DrawAttachmentSelection, MaxwellLoweringError> {
    let selection = state
        .render_targets()
        .color_target_selection()
        .value()
        .ok_or(MaxwellLoweringError::IncompleteDraw("SET_CT_SELECT"))?;
    for (slot, target) in selection.active_targets().iter().copied().enumerate() {
        if selection.active_targets()[..slot].contains(&target) {
            return Err(MaxwellLoweringError::DuplicateColorTargetRoute { target });
        }
    }
    let mut colors = Vec::with_capacity(selection.target_count() as usize);
    for (slot, target) in selection.active_targets().iter().copied().enumerate() {
        let configured = &state.render_targets().color()[target as usize];
        match configured.readiness(true) {
            super::threed::MaxwellThreeDAttachmentReadiness::Unprogrammed => {
                return Err(MaxwellLoweringError::ColorTargetRouteUnprogrammed {
                    slot: slot as u8,
                    target,
                });
            }
            super::threed::MaxwellThreeDAttachmentReadiness::Disabled => {
                return Err(MaxwellLoweringError::ColorTargetRouteDisabled {
                    slot: slot as u8,
                    target,
                });
            }
            super::threed::MaxwellThreeDAttachmentReadiness::Ready => {}
            _ => {
                return Err(MaxwellLoweringError::ColorTargetRouteIncomplete {
                    slot: slot as u8,
                    target,
                });
            }
        }
        let index = MaxwellThreeDResourceRole::ColorTarget(target);
        let image = resolved_image(resources, index)?;
        if image.description().kind() != nixe_gpu::ImageKind::Color {
            return Err(MaxwellLoweringError::ResolvedResourceKindMismatch);
        }
        colors.push((target, index));
    }
    let depth_stencil = draw_depth_stencil_attachment_required(state)
        .then_some(MaxwellThreeDResourceRole::DepthStencilTarget)
        .filter(|role| resources.resource(*role).is_some());
    if let Some(index) = depth_stencil {
        let image = resolved_image(resources, index)?;
        if image.description().kind() != nixe_gpu::ImageKind::DepthStencil {
            return Err(MaxwellLoweringError::ResolvedResourceKindMismatch);
        }
    }
    Ok(DrawAttachmentSelection {
        colors,
        color_outputs: [nixe_gpu::ColorOutputState::REPLACE; 8],
        depth_stencil,
    })
}

/// A configured depth/stencil target is not an attachment dependency when both
/// fragment tests are explicitly disabled. Unknown state stays conservative:
/// it must not silently discard a guest depth/stencil dependency.
fn draw_depth_stencil_attachment_required(state: &MaxwellThreeDState) -> bool {
    let (depth, stencil) = draw_depth_stencil_aspects(state);
    depth || stencil
}

/// Unknown test-enable state must still be validated by pipeline lowering, but
/// it does not prove that this operation references depth/stencil memory.
fn draw_depth_stencil_resource_required(state: &MaxwellThreeDState) -> bool {
    let (depth, stencil) = draw_depth_stencil_enable_state(state);

    depth == Some(true) || stencil == Some(true)
}

/// Returns the aspects that a draw may observe. Missing enable state remains
/// conservative and therefore requires the corresponding guest contents.
fn draw_depth_stencil_aspects(state: &MaxwellThreeDState) -> (bool, bool) {
    let (depth, stencil) = draw_depth_stencil_enable_state(state);
    (depth.unwrap_or(true), stencil.unwrap_or(true))
}

fn draw_depth_stencil_enable_state(state: &MaxwellThreeDState) -> (Option<bool>, Option<bool>) {
    // SET_ZT_SELECT.TARGET_COUNT=0 unbinds Z independently of the test enables.
    // Only explicit absence suppresses consumption; an unknown selector must
    // not hide a missing descriptor. ClearSurface has its own resource roles.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2748-L2749
    if state.render_targets().depth_target_count().value()
        == Some(&super::threed::MaxwellThreeDDepthTargetCount::None)
    {
        return (Some(false), Some(false));
    }
    let boolean = |register| {
        state
            .fixed_function()
            .register(register)
            .value()
            .and_then(|value| match value {
                MaxwellThreeDFixedFunctionValue::Boolean(value) => Some(*value),
                _ => None,
            })
    };
    (
        boolean(MaxwellThreeDFixedFunctionRegister::DepthTestEnable),
        boolean(MaxwellThreeDFixedFunctionRegister::StencilTestEnable),
    )
}

fn draw_depth_state(state: &MaxwellThreeDState) -> Result<DepthState, MaxwellLoweringError> {
    let register = |register| state.fixed_function().register(register).value();
    let Some(MaxwellThreeDFixedFunctionValue::Boolean(test_enabled)) =
        register(MaxwellThreeDFixedFunctionRegister::DepthTestEnable)
    else {
        return Err(MaxwellLoweringError::IncompleteDraw("SET_DEPTH_TEST"));
    };
    if !test_enabled {
        return Ok(DepthState::DISABLED);
    }
    let Some(MaxwellThreeDFixedFunctionValue::Boolean(write_enabled)) =
        register(MaxwellThreeDFixedFunctionRegister::DepthWriteEnable)
    else {
        return Err(MaxwellLoweringError::IncompleteDraw("SET_DEPTH_WRITE"));
    };
    let Some(MaxwellThreeDFixedFunctionValue::Compare(compare)) =
        register(MaxwellThreeDFixedFunctionRegister::DepthCompare)
    else {
        return Err(MaxwellLoweringError::IncompleteDraw("SET_DEPTH_FUNC"));
    };
    Ok(DepthState::new(
        true,
        *write_enabled,
        neutral_depth_compare(*compare),
    ))
}

fn draw_stencil_test_state(
    state: &MaxwellThreeDState,
) -> Result<Option<nixe_gpu::StencilTestState>, MaxwellLoweringError> {
    use MaxwellThreeDFixedFunctionRegister as R;
    use MaxwellThreeDFixedFunctionValue as V;
    if draw_depth_stencil_enable_state(state).1 != Some(true) {
        return Ok(None);
    }
    let value = |r| state.fixed_function().register(r).value();
    let two_sided = match value(R::TwoSidedStencilTestEnable) {
        Some(V::Boolean(enabled)) => *enabled,
        _ => {
            return Err(MaxwellLoweringError::IncompleteDraw(
                "two-sided stencil enable",
            ));
        }
    };
    let face = |compare, mask, write_mask, reference, ops: [R; 3]| {
        let Some(V::Compare(compare)) = value(compare) else {
            return Err(MaxwellLoweringError::IncompleteDraw("stencil compare"));
        };
        let Some(V::Mask(mask)) = value(mask) else {
            return Err(MaxwellLoweringError::IncompleteDraw("stencil compare mask"));
        };
        let Some(V::Mask(reference)) = value(reference) else {
            return Err(MaxwellLoweringError::IncompleteDraw("stencil reference"));
        };
        let operation = |r| match value(r) {
            Some(V::StencilOp(op)) => Ok(neutral_stencil_operation(*op)),
            _ => Err(MaxwellLoweringError::IncompleteDraw("stencil operation")),
        };
        let [fail, depth_fail, pass] = ops.map(operation);
        let [fail, depth_fail, pass] = [fail?, depth_fail?, pass?];
        let write_mask = match value(write_mask) {
            Some(V::Mask(mask)) => *mask as u8,
            _ if [fail, depth_fail, pass]
                .iter()
                .all(|op| *op == nixe_gpu::StencilOperation::Keep) =>
            {
                0
            }
            _ => return Err(MaxwellLoweringError::IncompleteDraw("stencil write mask")),
        };
        Ok(nixe_gpu::StencilFaceState {
            compare: neutral_depth_compare(*compare),
            read_mask: *mask as u8,
            write_mask,
            reference: *reference as u8,
            fail,
            depth_fail,
            pass,
        })
    };
    let front = face(
        R::FrontStencilCompare,
        R::FrontStencilCompareMask,
        R::FrontStencilWriteMask,
        R::FrontStencilReference,
        [
            R::FrontStencilFail,
            R::FrontStencilDepthFail,
            R::FrontStencilPass,
        ],
    )?;
    let back = if two_sided {
        face(
            R::BackStencilCompare,
            R::BackStencilCompareMask,
            R::BackStencilWriteMask,
            R::BackStencilReference,
            [
                R::BackStencilFail,
                R::BackStencilDepthFail,
                R::BackStencilPass,
            ],
        )?
    } else {
        front
    };
    Ok(Some(nixe_gpu::StencilTestState { front, back }))
}

const fn neutral_stencil_operation(
    op: super::threed::MaxwellThreeDStencilOp,
) -> nixe_gpu::StencilOperation {
    use super::threed::MaxwellThreeDStencilOp as M;
    use nixe_gpu::StencilOperation as N;
    match op {
        M::Keep => N::Keep,
        M::Zero => N::Zero,
        M::Replace => N::Replace,
        M::IncrementClamp => N::IncrementClamp,
        M::DecrementClamp => N::DecrementClamp,
        M::Invert => N::Invert,
        M::IncrementWrap => N::IncrementWrap,
        M::DecrementWrap => N::DecrementWrap,
    }
}

const fn neutral_depth_compare(compare: MaxwellThreeDCompareOp) -> DepthCompareOperation {
    match compare {
        MaxwellThreeDCompareOp::Never => DepthCompareOperation::Never,
        MaxwellThreeDCompareOp::Less => DepthCompareOperation::Less,
        MaxwellThreeDCompareOp::Equal => DepthCompareOperation::Equal,
        MaxwellThreeDCompareOp::LessEqual => DepthCompareOperation::LessEqual,
        MaxwellThreeDCompareOp::Greater => DepthCompareOperation::Greater,
        MaxwellThreeDCompareOp::NotEqual => DepthCompareOperation::NotEqual,
        MaxwellThreeDCompareOp::GreaterEqual => DepthCompareOperation::GreaterEqual,
        MaxwellThreeDCompareOp::Always => DepthCompareOperation::Always,
    }
}

fn draw_scissor(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    attachments: &DrawAttachmentSelection,
) -> Result<nixe_gpu::ScissorRect, MaxwellLoweringError> {
    let (mut width, mut height) = (u32::MAX, u32::MAX);
    for index in attachments.attachment_indices_iter() {
        let extent = resolved_image(resources, index)?.description().extent();
        width = width.min(extent.width);
        height = height.min(extent.height);
    }
    draw_scissor_region(state, width, height)
}

fn draw_scissor_region(
    state: &MaxwellThreeDState,
    width: u32,
    height: u32,
) -> Result<nixe_gpu::ScissorRect, MaxwellLoweringError> {
    let mut region = draw_surface_region(state, width, height)?;
    let scissor = &state.fixed_function().scissor()[0];
    if scissor.enable().value() == Some(&true) {
        let mut scissor_y =
            scissor
                .vertical()
                .value()
                .copied()
                .ok_or(MaxwellLoweringError::IncompleteDraw(
                    "SET_SCISSOR_VERTICAL(0)",
                ))?;
        if lower_left_window_origin(state) {
            let height = window_origin_height(state)?;
            scissor_y = super::threed::MaxwellThreeDRectangle {
                min: height.saturating_sub(scissor_y.max),
                max: height.saturating_sub(scissor_y.min),
            };
        }
        region.intersect(
            scissor
                .horizontal()
                .value()
                .copied()
                .ok_or(MaxwellLoweringError::IncompleteDraw(
                    "SET_SCISSOR_HORIZONTAL(0)",
                ))?,
            scissor_y,
        );
    }
    Ok(region.scissor(width, height))
}

fn draw_surface_region(
    state: &MaxwellThreeDState,
    width: u32,
    height: u32,
) -> Result<MaxwellThreeDRasterRegion, MaxwellLoweringError> {
    // Surface clip uses origin + extent; scissor uses half-open min/max.
    // Both restrict window-space fragments, preserving vertex execution and
    // shader-visible coordinates even when their intersection is empty.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1386-L1392
    let mut region = MaxwellThreeDRasterRegion::attachment(width, height);
    let horizontal = state.fixed_function().surface_clip_horizontal().value();
    let vertical = state.fixed_function().surface_clip_vertical().value();
    match (horizontal, vertical) {
        (Some(horizontal), Some(vertical)) => {
            region.min_x = u32::from(horizontal.origin()).min(width);
            region.max_x =
                (u32::from(horizontal.origin()) + u32::from(horizontal.extent())).min(width);
            region.min_y = u32::from(vertical.origin()).min(height);
            region.max_y =
                (u32::from(vertical.origin()) + u32::from(vertical.extent())).min(height);
        }
        (None, None) => {}
        _ => {
            return Err(MaxwellLoweringError::IncompleteDraw(
                "SET_SURFACE_CLIP_HORIZONTAL/VERTICAL",
            ));
        }
    }
    Ok(region)
}

fn lower_left_window_origin(state: &MaxwellThreeDState) -> bool {
    matches!(state.fixed_function().register(MaxwellThreeDFixedFunctionRegister::WindowOrigin).value(),
        Some(MaxwellThreeDFixedFunctionValue::Mask(value)) if value & 1 != 0)
}

fn window_origin_height(state: &MaxwellThreeDState) -> Result<u16, MaxwellLoweringError> {
    state
        .fixed_function()
        .surface_clip_vertical()
        .value()
        .map(|axis| axis.extent())
        .ok_or(MaxwellLoweringError::IncompleteDraw(
            "SET_SURFACE_CLIP_VERTICAL",
        ))
}

#[cfg(test)]
fn draw_viewport_transform(
    state: &MaxwellThreeDState,
) -> Result<Option<ViewportTransform>, MaxwellLoweringError> {
    draw_viewport_transform_with_depth_usage(state, true)
}

fn draw_viewport_transform_with_depth_usage(
    state: &MaxwellThreeDState,
    pixel_depth_observed: bool,
) -> Result<Option<ViewportTransform>, MaxwellLoweringError> {
    let enabled = match state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable)
        .value()
    {
        Some(MaxwellThreeDFixedFunctionValue::ViewportScaleOffsetEnable(value)) => {
            *value == MaxwellThreeDViewportScaleOffsetEnable::Enabled
        }
        _ => {
            return Err(MaxwellLoweringError::IncompleteDraw(
                "SET_VIEWPORT_SCALE_OFFSET",
            ));
        }
    };
    // The current draw contract selects viewport zero. Shader-selected or
    // replicated viewports require a distinct draw contract.
    let viewport = &state.fixed_function().viewport()[0];
    if let Some(&precision) = viewport.snap_grid_precision().value()
        && precision != [0, 0]
    {
        return Err(MaxwellLoweringError::UnsupportedViewportSnapGridPrecision {
            viewport: 0,
            precision,
        });
    }
    let swizzle = viewport.coordinate_swizzle().value().copied();
    let reflect_y = if let Some(swizzle) = swizzle {
        use MaxwellThreeDViewportSwizzleComponent::{NegativeY, PositiveW, PositiveX, PositiveZ};
        if swizzle.is_identity() {
            false
        } else if enabled && swizzle.components() == [PositiveX, NegativeY, PositiveZ, PositiveW] {
            true
        } else {
            return Err(
                MaxwellLoweringError::UnsupportedViewportCoordinateSwizzleSemantics {
                    viewport: 0,
                    swizzle,
                },
            );
        }
    } else {
        false
    };
    if !enabled {
        if lower_left_window_origin(state) {
            return Err(MaxwellLoweringError::UnsupportedWindowOrigin(1));
        }
        return Ok(None);
    }
    let scale = viewport
        .scale()
        .each_ref()
        .map(|register| register.value().copied())
        .map(|value| value.map(|value| f32::from_bits(value.get())));
    let offset = viewport
        .offset()
        .each_ref()
        .map(|register| register.value().copied())
        .map(|value| value.map(|value| f32::from_bits(value.get())));
    let [Some(scale_x), Some(scale_y), Some(scale_z)] = scale else {
        return Err(MaxwellLoweringError::IncompleteDraw(
            "SET_VIEWPORT_SCALE_X/Y/Z(0)",
        ));
    };
    let [Some(offset_x), Some(offset_y), Some(offset_z)] = offset else {
        return Err(MaxwellLoweringError::IncompleteDraw(
            "SET_VIEWPORT_OFFSET_X/Y/Z(0)",
        ));
    };
    let Some(clip_min_z) = viewport
        .clip_min_z()
        .value()
        .copied()
        .map(|value| f32::from_bits(value.get()))
    else {
        return Err(MaxwellLoweringError::IncompleteDraw(
            "SET_VIEWPORT_CLIP_MIN_Z(0)",
        ));
    };
    let Some(clip_max_z) = viewport
        .clip_max_z()
        .value()
        .copied()
        .map(|value| f32::from_bits(value.get()))
    else {
        return Err(MaxwellLoweringError::IncompleteDraw(
            "SET_VIEWPORT_CLIP_MAX_Z(0)",
        ));
    };
    // NV_viewport_swizzle acts before clipping and perspective division.
    // A Y-only reflection preserves the symmetric -w..w clip volume and w;
    // its exact window transform is therefore (y/w)*(-scale_y)+offset_y.
    // Fold the sign into the existing neutral transform; the backend applies
    // it at vertex output, preserving varyings, depth, and polygon facing.
    // https://registry.khronos.org/OpenGL/extensions/NV/NV_viewport_swizzle.txt
    // SET_VIEWPORT_Z_CLIP selects the API depth range; the independently
    // programmed CLIP_CONTROL pixel limits do not change this affine mode.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L951-L955
    // https://github.com/eden-emulator/mirror/blob/67bada77f8a43a90da2e94e89b8e7da73c256989/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L95-L110
    let negative_one_to_one = match state.viewport().z_clip_range().value() {
        Some(super::threed::MaxwellThreeDViewportZClipRange::NegativeWToPositiveW) => true,
        Some(super::threed::MaxwellThreeDViewportZClipRange::ZeroToPositiveW) => false,
        None => return Err(MaxwellLoweringError::IncompleteDraw("SET_VIEWPORT_Z_CLIP")),
    };
    // ClipMin/MaxZ are post-transform pixel bounds, not viewport endpoints.
    // Unbounded values are legal. The affine range comes from the configured
    // clip volume and ScaleZ/OffsetZ, without guessing the guest API's mode.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3372-L3399
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L98-L110
    let mut fragment_depth_transform = None;
    let mut depth_range = [
        if negative_one_to_one {
            offset_z - scale_z
        } else {
            offset_z
        },
        offset_z + scale_z,
    ];
    if clip_min_z.is_nan()
        || clip_max_z.is_nan()
        || clip_min_z > depth_range[0].min(depth_range[1])
        || clip_max_z < depth_range[0].max(depth_range[1])
    {
        let clamp_both = matches!(state.fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ViewportClipControl).value(),
            Some(MaxwellThreeDFixedFunctionValue::ClipControl(control)) if control.raw() & 0x18 == 0x18);
        if !clamp_both
            || !clip_min_z.is_finite()
            || !clip_max_z.is_finite()
            || clip_min_z < 0.0
            || clip_max_z > 1.0
            || clip_min_z > clip_max_z
        {
            return Err(MaxwellLoweringError::UnsupportedViewportPixelDepthBounds);
        }
        if pixel_depth_observed {
            fragment_depth_transform = Some(
                nixe_gpu::FragmentDepthTransform::new(
                    depth_range[1] - depth_range[0],
                    depth_range[0],
                    clip_min_z,
                    clip_max_z,
                )
                .map_err(MaxwellLoweringError::Command)?,
            );
        }
        // Clamping cannot discard fragments. If no shader or test observes
        // window depth, its affine mapping is dead state; keep the original
        // geometric clip volume and use a legal host depth range.
        depth_range = [0.0, 1.0];
    }
    let mut effective_scale_y = if reflect_y { -scale_y } else { scale_y };
    let mut effective_offset_y = offset_y;
    if lower_left_window_origin(state) {
        // Convert bottom-left window coordinates to the neutral top-left
        // framebuffer convention: y_host = surface_height - y_guest.
        // Scissor bounds use the same origin, independently of FLIP_Y facing.
        // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L128-L134
        effective_scale_y = -effective_scale_y;
        effective_offset_y = f32::from(window_origin_height(state)?) - offset_y;
    }
    ViewportTransform::new(
        [scale_x, effective_scale_y, scale_z],
        [offset_x, effective_offset_y, offset_z],
        depth_range,
    )
    .map(|transform| {
        let transform = transform.with_negative_one_to_one_depth(negative_one_to_one);
        fragment_depth_transform.map_or(transform, |depth| {
            transform.with_fragment_depth_transform(depth)
        })
    })
    .map(Some)
    .map_err(MaxwellLoweringError::Command)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MaxwellThreeDRasterRegion {
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
}

impl MaxwellThreeDRasterRegion {
    fn scissor(self, width: u32, height: u32) -> nixe_gpu::ScissorRect {
        nixe_gpu::ScissorRect {
            x: self.min_x.min(width),
            y: self.min_y.min(height),
            width: self.max_x.saturating_sub(self.min_x),
            height: self.max_y.saturating_sub(self.min_y),
        }
    }

    const fn attachment(width: u32, height: u32) -> Self {
        Self {
            min_x: 0,
            max_x: width,
            min_y: 0,
            max_y: height,
        }
    }

    fn intersect(
        &mut self,
        horizontal: super::threed::MaxwellThreeDRectangle,
        vertical: super::threed::MaxwellThreeDRectangle,
    ) {
        self.min_x = self.min_x.max(u32::from(horizontal.min));
        self.max_x = self.max_x.min(u32::from(horizontal.max));
        self.min_y = self.min_y.max(u32::from(vertical.min));
        self.max_y = self.max_y.min(u32::from(vertical.max));
    }

    const fn is_empty(self) -> bool {
        self.min_x >= self.max_x || self.min_y >= self.max_y
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MaxwellThreeDClearRegions {
    clear: Option<(
        super::threed::MaxwellThreeDRectangle,
        super::threed::MaxwellThreeDRectangle,
    )>,
    scissor: Option<(
        super::threed::MaxwellThreeDRectangle,
        super::threed::MaxwellThreeDRectangle,
    )>,
    viewport_clip: Option<(
        super::threed::MaxwellThreeDClipAxis,
        super::threed::MaxwellThreeDClipAxis,
    )>,
}

impl MaxwellThreeDClearRegions {
    fn from_state(state: &MaxwellThreeDState) -> Result<Self, MaxwellLoweringError> {
        let clear = state.render_targets().clear();
        let control = clear.surface_control().value().copied();
        let clear = control
            .is_none_or(|control| control.use_clear_rect())
            .then(|| {
                Ok((
                    clear.horizontal().value().copied().ok_or(
                        MaxwellLoweringError::IncompleteClear("horizontal rectangle"),
                    )?,
                    clear
                        .vertical()
                        .value()
                        .copied()
                        .ok_or(MaxwellLoweringError::IncompleteClear("vertical rectangle"))?,
                ))
            })
            .transpose()?;
        let scissor = control
            .is_some_and(|control| control.use_scissor_zero())
            .then(|| {
                let scissor = &state.fixed_function().scissor()[0];
                Ok((
                    scissor.horizontal().value().copied().ok_or(
                        MaxwellLoweringError::IncompleteClear("SET_SCISSOR_HORIZONTAL(0)"),
                    )?,
                    scissor.vertical().value().copied().ok_or(
                        MaxwellLoweringError::IncompleteClear("SET_SCISSOR_VERTICAL(0)"),
                    )?,
                ))
            })
            .transpose()?;
        let viewport_clip = control
            .is_some_and(|control| control.use_viewport_clip_zero())
            .then(|| {
                let viewport = &state.fixed_function().viewport()[0];
                Ok((
                    viewport.clip_horizontal().value().copied().ok_or(
                        MaxwellLoweringError::IncompleteClear("SET_VIEWPORT_CLIP_HORIZONTAL(0)"),
                    )?,
                    viewport.clip_vertical().value().copied().ok_or(
                        MaxwellLoweringError::IncompleteClear("SET_VIEWPORT_CLIP_VERTICAL(0)"),
                    )?,
                ))
            })
            .transpose()?;
        Ok(Self {
            clear,
            scissor,
            viewport_clip,
        })
    }

    fn for_attachment(self, width: u32, height: u32) -> MaxwellThreeDRasterRegion {
        let mut region = MaxwellThreeDRasterRegion::attachment(width, height);
        for (horizontal, vertical) in [self.clear, self.scissor].into_iter().flatten() {
            region.intersect(horizontal, vertical);
        }
        if let Some((horizontal, vertical)) = self.viewport_clip {
            region.min_x = region.min_x.max(u32::from(horizontal.origin()));
            region.max_x = region
                .max_x
                .min(u32::from(horizontal.origin()) + u32::from(horizontal.extent()));
            region.min_y = region.min_y.max(u32::from(vertical.origin()));
            region.max_y = region
                .max_y
                .min(u32::from(vertical.origin()) + u32::from(vertical.extent()));
        }
        region
    }
}

fn primitive_topology(
    begin: MaxwellThreeDBegin,
) -> Result<PrimitiveTopology, MaxwellLoweringError> {
    if begin.preserve_primitive_id() {
        return Err(MaxwellLoweringError::UnsupportedPrimitiveIdContinuation);
    }
    if begin.split_mode() != 0 {
        return Err(MaxwellLoweringError::UnsupportedPrimitiveSplitMode(
            begin.split_mode(),
        ));
    }
    match begin.topology() {
        0 => Ok(PrimitiveTopology::Points),
        1 => Ok(PrimitiveTopology::Lines),
        3 => Ok(PrimitiveTopology::LineStrip),
        4 => Ok(PrimitiveTopology::Triangles),
        5 => Ok(PrimitiveTopology::TriangleStrip),
        6 => Ok(PrimitiveTopology::TriangleFan),
        // NVB197_BEGIN_OP_QUADS; keep primitive assembly backend-independent.
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
        7 => Ok(PrimitiveTopology::Quads),
        14 => Ok(PrimitiveTopology::Patches),
        topology => Err(MaxwellLoweringError::UnsupportedTopology(topology)),
    }
}

fn neutral_first_instance(base: u32, relative: u32) -> Result<u32, MaxwellLoweringError> {
    base.checked_add(relative)
        .ok_or(MaxwellLoweringError::InstanceIndexOverflow { base, relative })
}

fn draw_arguments(
    state: &MaxwellThreeDState,
    trigger: MaxwellThreeDOperationTrigger,
) -> Result<DrawArguments, MaxwellLoweringError> {
    if let MaxwellThreeDOperationTrigger::DrawIndexBuffer { index_count, .. } = trigger {
        return indexed::draw_arguments(state, index_count);
    }
    let MaxwellThreeDOperationTrigger::DrawVertexArray { vertex_count, .. } = trigger else {
        unreachable!("only drawing triggers have draw arguments")
    };
    if vertex_count == 0 {
        return Err(MaxwellLoweringError::EmptyDraw);
    }
    let first_vertex = *state
        .vertex_input()
        .primitive()
        .vertex_array_start()
        .value()
        .ok_or(MaxwellLoweringError::IncompleteDraw("VERTEX_ARRAY_START"))?;
    let base_instance = state
        .vertex_input()
        .assembly()
        .global_base_instance_index()
        .value()
        .copied()
        .unwrap_or(0);
    let relative_instance = state.vertex_input().primitive().instance_index();
    Ok(DrawArguments::NonIndexed {
        first_vertex,
        vertex_count,
        first_instance: neutral_first_instance(base_instance, relative_instance)?,
        instance_count: 1,
    })
}

fn neutral_vertex_format(
    attribute: u8,
    format: super::threed::MaxwellThreeDVertexAttributeFormat,
) -> Result<VertexFormat, MaxwellLoweringError> {
    let widths = format
        .component_widths()
        .ok_or(MaxwellLoweringError::IncompleteDraw(
            "SET_VERTEX_ATTRIBUTE_A",
        ))?;
    let numerical = format
        .numerical_type()
        .ok_or(MaxwellLoweringError::IncompleteDraw(
            "SET_VERTEX_ATTRIBUTE_A",
        ))?;
    if format.swap_red_blue() {
        return Err(MaxwellLoweringError::UnsupportedVertexAttributeFormat {
            attribute,
            component_widths: widths,
            numerical_type: numerical,
            swap_red_blue: true,
        });
    }

    // Maxwell field values are pinned to NVIDIA's public class header:
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/cl9097.h#L1021-L1055
    let scaled_layout = || {
        let (width, components) = match widths.raw() {
            0x1d => (VertexComponentWidth::Bits8, VertexComponentCount::One),
            0x18 => (VertexComponentWidth::Bits8, VertexComponentCount::Two),
            0x13 => (VertexComponentWidth::Bits8, VertexComponentCount::Three),
            0x0a => (VertexComponentWidth::Bits8, VertexComponentCount::Four),
            0x1b => (VertexComponentWidth::Bits16, VertexComponentCount::One),
            0x0f => (VertexComponentWidth::Bits16, VertexComponentCount::Two),
            0x05 => (VertexComponentWidth::Bits16, VertexComponentCount::Three),
            0x03 => (VertexComponentWidth::Bits16, VertexComponentCount::Four),
            0x12 => (VertexComponentWidth::Bits32, VertexComponentCount::One),
            0x04 => (VertexComponentWidth::Bits32, VertexComponentCount::Two),
            0x02 => (VertexComponentWidth::Bits32, VertexComponentCount::Three),
            0x01 => (VertexComponentWidth::Bits32, VertexComponentCount::Four),
            _ => return None,
        };
        Some((width, components))
    };
    let vertex = match (widths.raw(), numerical) {
        (0x1d, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint8,
        (0x01, MaxwellThreeDVertexNumericalType::Float) => VertexFormat::Float32x4,
        (0x02, MaxwellThreeDVertexNumericalType::Float) => VertexFormat::Float32x3,
        (0x04, MaxwellThreeDVertexNumericalType::Float) => VertexFormat::Float32x2,
        (0x12, MaxwellThreeDVertexNumericalType::Float) => VertexFormat::Float32,
        (0x03, MaxwellThreeDVertexNumericalType::Float) => VertexFormat::Float16x4,
        (0x0f, MaxwellThreeDVertexNumericalType::Float) => VertexFormat::Float16x2,
        (0x01, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint32x4,
        (0x02, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint32x3,
        (0x04, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint32x2,
        (0x12, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint32,
        (0x01, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint32x4,
        (0x02, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint32x3,
        (0x04, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint32x2,
        (0x12, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint32,
        (0x03, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint16x4,
        (0x0f, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint16x2,
        (0x03, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint16x4,
        (0x0f, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint16x2,
        (0x03, MaxwellThreeDVertexNumericalType::SignedNormalized) => VertexFormat::Snorm16x4,
        (0x0f, MaxwellThreeDVertexNumericalType::SignedNormalized) => VertexFormat::Snorm16x2,
        (0x03, MaxwellThreeDVertexNumericalType::UnsignedNormalized) => VertexFormat::Unorm16x4,
        (0x0f, MaxwellThreeDVertexNumericalType::UnsignedNormalized) => VertexFormat::Unorm16x2,
        (0x0a, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint8x4,
        (0x18, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint8x2,
        (0x0a, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint8x4,
        (0x18, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint8x2,
        (0x0a, MaxwellThreeDVertexNumericalType::SignedNormalized) => VertexFormat::Snorm8x4,
        (0x18, MaxwellThreeDVertexNumericalType::SignedNormalized) => VertexFormat::Snorm8x2,
        (0x0a, MaxwellThreeDVertexNumericalType::UnsignedNormalized) => VertexFormat::Unorm8x4,
        (0x18, MaxwellThreeDVertexNumericalType::UnsignedNormalized) => VertexFormat::Unorm8x2,
        (0x30, MaxwellThreeDVertexNumericalType::UnsignedNormalized) => {
            VertexFormat::Unorm10_10_10_2
        }
        (0x30, MaxwellThreeDVertexNumericalType::SignedNormalized) => VertexFormat::Snorm10_10_10_2,
        (_, MaxwellThreeDVertexNumericalType::UnsignedScaled) => {
            let (width, components) =
                scaled_layout().ok_or(MaxwellLoweringError::UnsupportedVertexAttributeFormat {
                    attribute,
                    component_widths: widths,
                    numerical_type: numerical,
                    swap_red_blue: false,
                })?;
            VertexFormat::Uscaled { width, components }
        }
        (_, MaxwellThreeDVertexNumericalType::SignedScaled) => {
            let (width, components) =
                scaled_layout().ok_or(MaxwellLoweringError::UnsupportedVertexAttributeFormat {
                    attribute,
                    component_widths: widths,
                    numerical_type: numerical,
                    swap_red_blue: false,
                })?;
            VertexFormat::Sscaled { width, components }
        }
        _ => {
            return Err(MaxwellLoweringError::UnsupportedVertexAttributeFormat {
                attribute,
                component_widths: widths,
                numerical_type: numerical,
                swap_red_blue: false,
            });
        }
    };
    Ok(vertex)
}

fn resolved_buffer(
    resources: &MaxwellThreeDResolvedResources,
    index: MaxwellThreeDResourceRole,
) -> Result<&super::threed::MaxwellThreeDResolvedBuffer, MaxwellLoweringError> {
    match resources
        .resource(index)
        .ok_or(MaxwellLoweringError::MissingResolvedResource { role: index })?
    {
        MaxwellThreeDResolvedResource::Buffer(value) => Ok(value),
        _ => Err(MaxwellLoweringError::ResolvedResourceKindMismatch),
    }
}

fn resolved_image(
    resources: &MaxwellThreeDResolvedResources,
    index: MaxwellThreeDResourceRole,
) -> Result<&super::threed::MaxwellThreeDResolvedImage, MaxwellLoweringError> {
    match resources
        .resource(index)
        .ok_or(MaxwellLoweringError::MissingResolvedResource { role: index })?
    {
        MaxwellThreeDResolvedResource::Image(value) => Ok(value),
        _ => Err(MaxwellLoweringError::ResolvedResourceKindMismatch),
    }
}

/// Typed failure before any cache or backend effect is published.
#[derive(Clone, Debug)]
pub enum MaxwellLoweringError {
    ComputeLaunch(crate::MaxwellComputeLaunchError),
    BufferBacking(String),
    ImageBacking(String),
    UnsupportedMultisampleState(&'static str),
    BlitSourceNotResident,
    BlitFormatMismatch,
    UnsupportedBlit(&'static str),
    UnsupportedPolygonRasterization(&'static str),
    UnsupportedWindowOrigin(u32),
    ContradictoryState {
        reason: &'static str,
    },
    TriggerStateMismatch,
    UnsupportedRenderEnableMode(MaxwellThreeDRenderEnableMode),
    UnsupportedConditionalLoadConstantBufferSemantics,
    UnsupportedStreamOutputSemantics,
    VisibleCallLimitExceeded {
        stage: ShaderStage,
        required: u16,
        limit: u16,
    },
    UnsupportedColorReductionSemantics,
    UnsupportedConstantColorRenderingSemantics,
    UnsupportedApiMandatedEarlyZSemantics,
    UnsupportedPostPsInitialCoverageSemantics,
    UnsupportedViewportPixelDepthBounds,
    UnsupportedWindowOriginFragmentPosition,
    UnsupportedPostZPixelShaderImaskSemantics,
    UnsupportedPixelShaderInterlockSemantics(MaxwellThreeDPixelShaderInterlockControl),
    UnsupportedGlobalBaseVertexIndex(u32),
    UnsupportedVertexIdBase(u32),
    UnsupportedIndexFormat(super::threed::MaxwellThreeDIndexElementSize),
    UnsupportedIndexedDraw(&'static str),
    UnsupportedCsaaSemantics,
    UnsupportedAntiAliasAlphaControl {
        alpha_to_coverage: bool,
        alpha_to_one: bool,
    },
    UnsupportedCoverageToColorSemantics(MaxwellThreeDCoverageToColor),
    UnsupportedAlphaToCoverageOverrideSemantics(MaxwellThreeDAlphaToCoverageOverride),
    UnsupportedTirSemantics {
        control: Option<MaxwellThreeDTirControl>,
    },
    UnsupportedHybridAntiAliasSemantics(MaxwellThreeDHybridAntiAliasControl),
    UnsupportedSampleLocationsSemantics {
        group: u8,
        value: MaxwellThreeDSampleLocationGroup,
    },
    UnsupportedPsOutputSampleMaskSemantics,
    UnsupportedReplicatedColorTargetOutputSemantics,
    UnsupportedRenderTargetIndexOffsetSemantics(MaxwellThreeDRenderTargetIndexOffset),
    UnsupportedRenderTargetLayerSemantics(MaxwellThreeDRenderTargetLayer),
    UnsupportedShaderLocalMemorySemantics {
        default_size_per_warp: MaxwellThreeDShaderLocalMemoryPerWarpSize,
    },
    UnsupportedViewportPixelCenterSemantics(MaxwellThreeDViewportPixelCenter),
    UnsupportedViewportSnapGridPrecision {
        viewport: u8,
        precision: [u8; 2],
    },
    UnsupportedViewportCoordinateSwizzleSemantics {
        viewport: u8,
        swizzle: MaxwellThreeDViewportCoordinateSwizzle,
    },
    UnsupportedWindowClipSemantics,
    UnsupportedClipIdTestSemantics,
    UnsupportedClearStencilMaskSemantics,
    UnsupportedAliasedLineWidthSemantics,
    UnsupportedLineStippleSemantics {
        factor: u8,
        pattern: u16,
    },
    UnsupportedPolygonClipGeneratedEdgeSemantics,
    UnsupportedSkippedVertexComponent {
        attribute: u8,
        component: u8,
    },
    UnsupportedVertexAttributeFormat {
        attribute: u8,
        component_widths: super::threed::MaxwellThreeDVertexComponentWidths,
        numerical_type: super::threed::MaxwellThreeDVertexNumericalType,
        swap_red_blue: bool,
    },
    UnsupportedVertexInstanceDivisor {
        stream: u8,
        divisor: u32,
    },
    InvalidPatchSize(MaxwellThreeDPatchSize),
    TessellationStageTopology,
    TessellationMode {
        value: super::threed::MaxwellThreeDTessellationMode,
        source: Option<MaxwellMethodSource>,
        reason: super::threed::MaxwellTessellationModeError,
    },
    UnsupportedPointSpriteCoordinatesSemantics(MaxwellThreeDPointSpriteSelect),
    UnsupportedAttributePointSizeSemantics {
        slot: u8,
    },
    UnsupportedPointSpriteSemantics,
    UnsupportedAntiAliasedPointSemantics,
    UnsupportedPointCenterSemantics(MaxwellThreeDPointCenterMode),
    UnsupportedFillViaTriangleSemantics(MaxwellThreeDFillViaTriangleMode),
    UnsupportedFillRectangleDraw(&'static str),
    UnsupportedConservativeRasterSemantics,
    UnsupportedPolygonSmoothSemantics,
    UnsupportedPolygonStippleSemantics,
    UnsupportedEdgeFlagSemantics(MaxwellThreeDEdgeFlag),
    UnsupportedShadeModeSemantics(MaxwellThreeDShadeMode),
    UnsupportedProvokingVertexSemantics(MaxwellThreeDProvokingVertex),
    UnsupportedTwoSidedLightSemantics,
    UnsupportedPixelShaderSaturateSemantics {
        output: u8,
        range: MaxwellThreeDPixelShaderClampRange,
    },
    UnsupportedBlendFactor {
        target: Option<u8>,
        value: u32,
    },
    UnsupportedBlendFormat {
        target: u8,
        format: nixe_gpu::ImageFormat,
    },
    UnsupportedIteratedBlendSemantics {
        value: MaxwellThreeDIteratedBlend,
        pass_count: Option<u8>,
    },
    IncompleteLogicOpState,
    UnsupportedLogicOpSemantics(MaxwellThreeDLogicOp),
    IncompleteColorWriteState {
        target: u8,
        mask_register: u8,
    },
    IncompleteAlphaTestState(&'static str),
    CompressedDepthImportRequired {
        kind: u8,
    },
    CompressedColorImportRequired {
        target: u8,
    },
    CompressedSampledImageImportRequired {
        role: MaxwellThreeDResourceRole,
        kind: u8,
    },
    ShaderTranslationRequired,
    InvalidTranslatedShaders,
    TranslatedShaderStageMismatch,
    TranslatedShaderMemoryConfigurationMismatch {
        stage: ShaderStage,
        configured: MaxwellThreeDDirectlyAddressableMemory,
        required: MaxwellThreeDDirectlyAddressableMemory,
    },
    UnsupportedShaderStage(MaxwellShaderStage),
    InvalidShaderResourceUse {
        role: MaxwellThreeDResourceRole,
    },
    MissingResolvedResource {
        role: MaxwellThreeDResourceRole,
    },
    ResolvedResourceKindMismatch,
    InvalidResolvedView {
        role: MaxwellThreeDResourceRole,
    },
    AllocationDescriptionChanged {
        allocation: nixe_gpu::GpuAllocationId,
    },
    IncompleteClear(&'static str),
    EmptyClearMask,
    EmptyClearRectangle,
    PartialColorClearUnsupported {
        mask: u8,
    },
    ClearOutsideAttachment,
    IncompleteDraw(&'static str),
    IncompleteBlendState {
        target: Option<u8>,
        field: &'static str,
    },
    ColorTargetRouteUnprogrammed {
        slot: u8,
        target: u8,
    },
    ColorTargetRouteDisabled {
        slot: u8,
        target: u8,
    },
    ColorTargetRouteIncomplete {
        slot: u8,
        target: u8,
    },
    DuplicateColorTargetRoute {
        target: u8,
    },
    EmptyDraw,
    UnsupportedPrimitiveIdContinuation,
    UnsupportedPrimitiveSplitMode(u8),
    InstanceIndexOverflow {
        base: u32,
        relative: u32,
    },
    UnsupportedTopology(u8),
    AliasedDrawResources {
        first: MaxwellThreeDResourceRole,
        second: MaxwellThreeDResourceRole,
    },
    InvalidTransition,
    InvalidResourceCreation,
    Command(CommandDescriptionError),
    ResourceExhausted,
}

impl Display for MaxwellLoweringError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ComputeLaunch(error) => Display::fmt(error, formatter),
            Self::BufferBacking(reason) => write!(formatter, "Maxwell buffer backing cannot be lowered: {reason}"),
            Self::ImageBacking(reason) => write!(formatter, "Maxwell image backing cannot be lowered: {reason}"),
            Self::UnsupportedBlit(reason) => write!(formatter, "unsupported Maxwell 2D blit: {reason}"),
            Self::BlitFormatMismatch => formatter.write_str("2D image copy requires matching packed source and destination formats"),
            Self::BlitSourceNotResident => formatter.write_str("Maxwell color blit requires a current, fully initialized resident source"),
            Self::UnsupportedMultisampleState(reason) => write!(formatter, "Maxwell multisample state has no neutral lowering: {reason}"),
            Self::UnsupportedPolygonRasterization(reason) => write!(formatter, "MAXWELL_B polygon rasterization is unsupported: {reason}"),
            Self::UnsupportedWindowOrigin(value) => write!(formatter,
                "MAXWELL_B window origin/flip has no neutral coordinate lowering: value=0x{value:x}"),
            Self::ContradictoryState { reason } => {
                write!(formatter, "contradictory Maxwell 3D state: {reason}")
            }
            Self::TriggerStateMismatch => {
                formatter.write_str("3D trigger does not match its immutable state snapshot")
            }
            Self::UnsupportedRenderEnableMode(mode) => write!(
                formatter,
                "MAXWELL_B render-enable mode has no verified neutral execution: mode={mode:?}"
            ),
            Self::UnsupportedStreamOutputSemantics => formatter.write_str(
                "MAXWELL_B enabled stream output requires vertex capture to guest memory",
            ),
            Self::UnsupportedConditionalLoadConstantBufferSemantics => formatter.write_str(
                "MAXWELL_B conditional constant-buffer load has no verified execution semantics",
            ),
            Self::VisibleCallLimitExceeded {
                stage,
                required,
                limit,
            } => write!(
                formatter,
                "translated Maxwell shader exceeds SET_API_VISIBLE_CALL_LIMIT: stage={stage:?} required={required} limit={limit}"
            ),
            Self::UnsupportedColorReductionSemantics => formatter.write_str(
                "MAXWELL_B enabled color reduction has no verified neutral threshold evaluation or color-output semantics",
            ),
            Self::UnsupportedConstantColorRenderingSemantics => formatter.write_str(
                "MAXWELL_B enabled constant-color rendering is not represented by the neutral pipeline",
            ),
            Self::UnsupportedApiMandatedEarlyZSemantics => formatter.write_str(
                "MAXWELL_B API-mandated early depth/stencil ordering is not represented by the neutral pipeline",
            ),
            Self::UnsupportedPostPsInitialCoverageSemantics => formatter.write_str("MAXWELL_B pre-shader initial coverage for post-shader tests is not represented by the neutral pipeline"),
            Self::UnsupportedViewportPixelDepthBounds => formatter.write_str("MAXWELL_B pixel depth bounds narrow the affine viewport range and require explicit clip/clamp lowering"),
            Self::UnsupportedWindowOriginFragmentPosition => formatter.write_str("MAXWELL_B lower-left fragment position Y requires shader coordinate lowering"),
            Self::UnsupportedPostZPixelShaderImaskSemantics => formatter.write_str(
                "MAXWELL_B post-Z pixel-shader invocation mask is not represented by the neutral pipeline",
            ),
            Self::UnsupportedPixelShaderInterlockSemantics(value) => write!(
                formatter,
                "MAXWELL_B pixel-shader interlock is not represented by the neutral pipeline: control={value:?}"
            ),
            Self::UnsupportedGlobalBaseVertexIndex(value) => write!(
                formatter,
                "MAXWELL_B global base vertex index cannot be represented independently from vertex-buffer addressing: value={value}"
            ),
            Self::UnsupportedVertexIdBase(value) => write!(
                formatter,
                "MAXWELL_B SET_VERTEX_ID_BASE requires an independent shader vertex-ID adjustment: value=0x{value:08x}"
            ),
            Self::UnsupportedIndexFormat(format) => write!(formatter, "Maxwell index format has no neutral representation: {format:?}"),
            Self::UnsupportedIndexedDraw(reason) => write!(formatter, "Maxwell indexed draw is not represented: {reason}"),
            Self::UnsupportedCsaaSemantics => formatter.write_str(
                "MAXWELL_B enabled CSAA has no verified coverage sampling, resolve, capability, or coherency semantics",
            ),
            Self::UnsupportedAntiAliasAlphaControl { alpha_to_coverage, alpha_to_one } => write!(
                formatter,
                "MAXWELL_B alpha-to-coverage/dithering or alpha-to-one is not represented by the neutral pipeline: alpha-to-coverage={alpha_to_coverage} alpha-to-one={alpha_to_one}"
            ),
            Self::UnsupportedCoverageToColorSemantics(value) => write!(
                formatter,
                "MAXWELL_B coverage-to-color output is not represented by the neutral pipeline: color-target={}",
                value.color_target()
            ),
            Self::UnsupportedAlphaToCoverageOverrideSemantics(value) => write!(
                formatter,
                "MAXWELL_B alpha-to-coverage override qualification is not represented by the neutral pipeline: qualify-by-aa={} qualify-by-ps-sample-mask={}",
                value.qualify_by_anti_alias_enable(),
                value.qualify_by_pixel_shader_sample_mask()
            ),
            Self::UnsupportedTirSemantics { control } => write!(
                formatter,
                "MAXWELL_B enabled target-independent rasterization has no neutral raster, coverage, alpha-to-coverage, or query representation: control={control:?}"
            ),
            Self::UnsupportedHybridAntiAliasSemantics(value) => write!(
                formatter,
                "MAXWELL_B hybrid antialiasing is not represented by the neutral raster pipeline: passes={} centroid={:?} passes-extended={}",
                value.passes(),
                value.centroid(),
                value.passes_extended()
            ),
            Self::UnsupportedSampleLocationsSemantics { group, value } => write!(
                formatter,
                "MAXWELL_B custom sample locations are not represented by the neutral raster pipeline: group={group} raw=0x{:08x}",
                value.raw()
            ),
            Self::UnsupportedPsOutputSampleMaskSemantics => formatter.write_str(
                "MAXWELL_B effective pixel-shader sample-mask output has no shader translation or neutral backend representation",
            ),
            Self::UnsupportedReplicatedColorTargetOutputSemantics => formatter.write_str(
                "MAXWELL_B disabled separate MRT fragment data requires replicating fragment color output zero to every active color target",
            ),
            Self::UnsupportedRenderTargetIndexOffsetSemantics(value) => write!(
                formatter,
                "MAXWELL_B viewport-index render-target routing is not represented by the neutral attachment contract: mode={value:?}"
            ),
            Self::UnsupportedRenderTargetLayerSemantics(value) => write!(
                formatter,
                "MAXWELL_B render-target layer routing is not represented by shader translation or the neutral attachment contract: layer={} control={:?}",
                value.layer(),
                value.control()
            ),
            Self::UnsupportedShaderLocalMemorySemantics {
                default_size_per_warp,
            } => write!(
                formatter,
                "MAXWELL_B active shader-local-memory allocation has no translated-shader or neutral backend representation: default-size-per-warp={}",
                default_size_per_warp.bytes()
            ),
            Self::UnsupportedViewportPixelCenterSemantics(center) => write!(
                formatter,
                "MAXWELL_B viewport pixel-center convention is not represented by the neutral pipeline contract: center={center:?}"
            ),
            Self::UnsupportedViewportSnapGridPrecision { viewport, precision } => write!(formatter, "MAXWELL_B viewport {viewport} increased snap-grid precision is unsupported: X={} Y={}", precision[0], precision[1]),
            Self::UnsupportedViewportCoordinateSwizzleSemantics { viewport, swizzle } => write!(
                formatter,
                "MAXWELL_B viewport coordinate swizzle is not represented by the neutral pipeline contract: viewport={viewport} components={:?}",
                swizzle.components()
            ),
            Self::UnsupportedWindowClipSemantics => formatter.write_str(
                "MAXWELL_B enabled window clipping has no neutral pipeline or backend rasterization semantics",
            ),
            Self::UnsupportedClipIdTestSemantics => formatter.write_str(
                "MAXWELL_B enabled clip-ID testing has no implemented extent, surface-ID, comparison, or backend rasterization semantics",
            ),

            Self::UnsupportedClearStencilMaskSemantics => formatter.write_str(
                "MAXWELL_B partial stencil write mask on clear has no neutral backend representation",
            ),
            Self::UnsupportedAliasedLineWidthSemantics => formatter.write_str(
                "MAXWELL_B aliased line-width selection has no represented width register or host rasterization semantics",
            ),
            Self::UnsupportedLineStippleSemantics { factor, pattern } => write!(
                formatter,
                "MAXWELL_B line stippling has no neutral backend representation: factor={factor} pattern=0x{pattern:04x}"
            ),
            Self::UnsupportedPolygonClipGeneratedEdgeSemantics => formatter.write_str(
                "MAXWELL_B suppression of polygon-clip-generated edges has no neutral backend representation",
            ),
            Self::UnsupportedSkippedVertexComponent { attribute, component } => write!(formatter,
                "vertex shader consumes a skipped DA attribute component: attribute={attribute} component={component}; DA default input lowering is not implemented"),
            Self::UnsupportedVertexAttributeFormat {
                attribute,
                component_widths,
                numerical_type,
                swap_red_blue,
            } => write!(
                formatter,
                "MAXWELL_B vertex attribute has no exact neutral format: attribute={attribute} component-widths=0x{:02x} numerical-type={numerical_type:?} swap-red-blue={swap_red_blue}",
                component_widths.raw()
            ),
            Self::UnsupportedVertexInstanceDivisor { stream, divisor } => write!(
                formatter,
                "MAXWELL_B vertex stream instance divisor is not representable: stream={stream} divisor={divisor}"
            ),
            Self::InvalidPatchSize(size) => write!(
                formatter,
                "MAXWELL_B patch draw has an invalid control-point count: {}",
                size.control_points()
            ),
            Self::TessellationStageTopology => formatter.write_str("MAXWELL_B tessellation shaders require patch topology"),
            Self::TessellationMode { value, source, reason } => write!(formatter, "MAXWELL_B tessellation mode cannot be consumed: value={:#x} reason={reason:?} source={source:?}", value.raw()),
            Self::UnsupportedPointSpriteCoordinatesSemantics(select) => write!(
                formatter,
                "MAXWELL_B generated point-sprite coordinates are not represented by shader translation or the neutral pipeline contract: texture-mask=0x{:03x} r-mode={:?} origin={:?}",
                select.generated_texture_mask(),
                select.r_mode(),
                select.origin()
            ),
            Self::UnsupportedAttributePointSizeSemantics { slot } => write!(
                formatter,
                "MAXWELL_B shader-provided point size is not represented by shader or neutral backend lowering: slot={slot}"
            ),
            Self::UnsupportedPointSpriteSemantics => formatter.write_str(
                "MAXWELL_B enabled point-sprite rasterization is not represented by the neutral backend",
            ),
            Self::UnsupportedAntiAliasedPointSemantics => formatter.write_str(
                "MAXWELL_B anti-aliased point rasterization is not represented by the neutral backend",
            ),
            Self::UnsupportedPointCenterSemantics(mode) => write!(
                formatter,
                "MAXWELL_B point-center convention is not represented by the neutral pipeline contract: mode={mode:?}"
            ),
            Self::UnsupportedFillViaTriangleSemantics(mode) => write!(
                formatter,
                "MAXWELL_B fill-via-triangle mode is not represented by the neutral pipeline contract: mode={mode:?}"
            ),
            Self::UnsupportedFillRectangleDraw(reason) => write!(
                formatter,
                "MAXWELL_B fill-rectangle draw is not representable: {reason}"
            ),
            Self::UnsupportedConservativeRasterSemantics => formatter.write_str(
                "MAXWELL_B conservative rasterization is not represented by the neutral pipeline contract",
            ),
            Self::UnsupportedPolygonSmoothSemantics => formatter.write_str(
                "MAXWELL_B polygon smoothing is not represented by the neutral pipeline contract",
            ),
            Self::UnsupportedPolygonStippleSemantics => formatter.write_str(
                "MAXWELL_B polygon stippling is not represented by the neutral pipeline contract",
            ),
            Self::UnsupportedEdgeFlagSemantics(flag) => write!(
                formatter,
                "MAXWELL_B disabled polygon edge flag is not represented by the neutral pipeline contract: flag={flag:?}"
            ),
            Self::UnsupportedShadeModeSemantics(mode) => write!(
                formatter,
                "MAXWELL_B shade mode is not representable in the neutral pipeline contract: mode={mode:?}"
            ),
            Self::UnsupportedProvokingVertexSemantics(vertex) => write!(
                formatter,
                "MAXWELL_B provoking vertex is not representable in the neutral pipeline or shader interpolation contract: vertex={vertex:?}"
            ),
            Self::UnsupportedTwoSidedLightSemantics => formatter.write_str(
                "MAXWELL_B enabled two-sided fixed-function lighting is not represented by shader or neutral backend lowering",
            ),
            Self::UnsupportedPixelShaderSaturateSemantics { output, range } => write!(
                formatter,
                "MAXWELL_B pixel-shader output saturation is not represented by shader or neutral backend lowering: output={output} range={range:?}"
            ),
            Self::UnsupportedBlendFactor { target, value } => write!(formatter,
                "MAXWELL_B blend factor requires unsupported coupled/constant/dual-source semantics: target={target:?} value=0x{value:04x}"),
            Self::UnsupportedBlendFormat { target, format } => write!(formatter,
                "MAXWELL_B blend attachment format is not implemented: target={target} format={format:?}"),
            Self::UnsupportedIteratedBlendSemantics { value, pass_count } => write!(
                formatter,
                "MAXWELL_B iterated blending has no neutral backend representation: color={} alpha={} pass-count={pass_count:?}",
                value.color_enabled(),
                value.alpha_enabled()
            ),
            Self::IncompleteLogicOpState => formatter.write_str(
                "MAXWELL_B enabled logic operations require SET_LOGIC_OP_FUNC",
            ),
            Self::UnsupportedLogicOpSemantics(function) => write!(
                formatter,
                "MAXWELL_B color logic operation {:?} is not represented by the neutral render pipeline",
                function
            ),
            Self::IncompleteColorWriteState {
                target,
                mask_register,
            } => write!(
                formatter,
                "MAXWELL_B color target {target} selects unprogrammed SET_CT_WRITE({mask_register})",
            ),
            Self::IncompleteAlphaTestState(field) => write!(
                formatter,
                "MAXWELL_B enabled alpha testing requires SET_ALPHA_{field}"
            ),
            Self::CompressedDepthImportRequired { kind } => write!(
                formatter,
                "Maxwell compressed depth contents require materialization before use: kind=0x{kind:02x}"
            ),
            Self::CompressedSampledImageImportRequired { role, kind } => write!(
                formatter,
                "Maxwell compressed sampled image has no current materialized resident image: role={role:?} kind={kind:#04x}"
            ),
            Self::CompressedColorImportRequired { target } => write!(
                formatter,
                "Maxwell compressed color contents require materialization before use: target={target}"
            ),
            Self::ShaderTranslationRequired => {
                formatter.write_str("Maxwell shader translation is required before draw lowering")
            }
            Self::InvalidTranslatedShaders => {
                formatter.write_str("translated shader evidence is empty, duplicated, or invalid")
            }
            Self::TranslatedShaderStageMismatch => formatter
                .write_str("translated shader stages do not match enabled Maxwell pipeline stages"),
            Self::TranslatedShaderMemoryConfigurationMismatch {
                stage,
                configured,
                required,
            } => write!(
                formatter,
                "translated shader requires a different Maxwell directly addressable memory configuration: stage={stage:?} configured-bytes={} translated-for-bytes={}",
                configured.bytes(),
                required.bytes()
            ),
            Self::UnsupportedShaderStage(stage) => write!(
                formatter,
                "Maxwell shader stage has no neutral lowering: {stage:?}"
            ),
            Self::InvalidShaderResourceUse { role } => write!(
                formatter,
                "translated shader declares an invalid resource use: role={role:?}"
            ),
            Self::MissingResolvedResource { role } => write!(
                formatter,
                "complete 3D snapshot lacks resolved resource: role={role:?}"
            ),
            Self::ResolvedResourceKindMismatch => {
                formatter.write_str("resolved 3D resource kind contradicts its role")
            }
            Self::InvalidResolvedView { role } => write!(
                formatter,
                "resolved 3D view cannot be re-identified neutrally: role={role:?}"
            ),
            Self::AllocationDescriptionChanged { allocation } => write!(
                formatter,
                "cached GPU allocation changed immutable description: {allocation}"
            ),
            Self::IncompleteClear(field) => {
                write!(formatter, "clear state is incomplete: missing={field}")
            }
            Self::EmptyClearMask => {
                formatter.write_str("CLEAR_SURFACE selects no color, depth, or stencil component")
            }
            Self::EmptyClearRectangle => formatter.write_str(
                "clear rectangle, scissor, and viewport-clip intersection is empty",
            ),
            Self::PartialColorClearUnsupported { mask } => write!(
                formatter,
                "partial color-channel clear is not represented yet: mask={mask:#x}"
            ),
            Self::ClearOutsideAttachment => {
                formatter.write_str("clear rectangle or layer lies outside the resolved attachment")
            }
            Self::IncompleteDraw(field) => {
                write!(formatter, "draw state is incomplete: missing={field}")
            }
            Self::IncompleteBlendState { target, field } => match target {
                Some(target) => write!(
                    formatter,
                    "blend state is incomplete: target={target} missing={field}"
                ),
                None => write!(formatter, "common blend state is incomplete: missing={field}"),
            },
            Self::ColorTargetRouteUnprogrammed { slot, target } => write!(
                formatter,
                "SET_CT_SELECT routes output slot {slot} to unprogrammed color target {target}"
            ),
            Self::ColorTargetRouteDisabled { slot, target } => write!(
                formatter,
                "SET_CT_SELECT routes output slot {slot} to disabled color target {target}"
            ),
            Self::ColorTargetRouteIncomplete { slot, target } => write!(
                formatter,
                "SET_CT_SELECT routes output slot {slot} to incomplete color target {target}"
            ),
            Self::DuplicateColorTargetRoute { target } => write!(
                formatter,
                "SET_CT_SELECT routes one color target more than once: target={target}"
            ),
            Self::EmptyDraw => formatter.write_str("draw vertex count is zero"),
            Self::UnsupportedPrimitiveIdContinuation => formatter
                .write_str("BEGIN requests unsupported primitive-ID continuation semantics"),
            Self::UnsupportedPrimitiveSplitMode(mode) => write!(
                formatter,
                "BEGIN requests unsupported split-primitive semantics: mode={mode}"
            ),
            Self::InstanceIndexOverflow { base, relative } => write!(
                formatter,
                "Maxwell instance index exceeds the neutral u32 domain: base={base} relative={relative}"
            ),
            Self::UnsupportedTopology(topology) => write!(
                formatter,
                "primitive topology has no neutral lowering: topology={topology:#x}"
            ),
            Self::AliasedDrawResources { first, second } => write!(
                formatter,
                "draw has a read/write or attachment alias without modeled feedback semantics: first={first:?} second={second:?}"
            ),
            Self::InvalidTransition => {
                formatter.write_str("derived neutral resource transition is invalid")
            }
            Self::InvalidResourceCreation => {
                formatter.write_str("derived neutral resource creation is invalid")
            }
            Self::Command(error) => {
                write!(formatter, "neutral command construction failed: {error}")
            }
            Self::ResourceExhausted => {
                formatter.write_str("GPU lowering exhausted host resources or identities")
            }
        }
    }
}

impl std::error::Error for MaxwellLoweringError {}

fn validate_consumed_state(
    state: &MaxwellThreeDState,
    trigger: MaxwellThreeDOperationTrigger,
) -> Result<
    (
        Option<nixe_gpu::TessellationState>,
        Option<raster::DrawRasterState>,
    ),
    MaxwellLoweringError,
> {
    let mut raster_state = None;
    let tessellation = if trigger.is_draw() {
        super::threed::tessellation::draw_state(state)?
    } else {
        None
    };
    if state
        .render_enable()
        .conditional_load_constant_buffer()
        .value()
        == Some(&MaxwellThreeDConditionalLoadConstantBuffer::Enabled)
    {
        return Err(MaxwellLoweringError::UnsupportedConditionalLoadConstantBufferSemantics);
    }
    if trigger.is_draw()
        && let Some(MaxwellThreeDFixedFunctionValue::ShadeMode(MaxwellThreeDShadeMode::Flat)) =
            state
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::ShadeMode)
                .value()
    {
        // Smooth preserves the interpolation selected by translated shader
        // inputs and therefore needs no fixed-function override. Flat shading
        // changes the primitive-wide source value and remains a typed boundary
        // until the shader translator represents that override explicitly.
        return Err(MaxwellLoweringError::UnsupportedShadeModeSemantics(
            MaxwellThreeDShadeMode::Flat,
        ));
    }
    if trigger.is_draw()
        && state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::TwoSidedLightEnable)
            .value()
            == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        return Err(MaxwellLoweringError::UnsupportedTwoSidedLightSemantics);
    }
    // SET_COLOR_CLAMP applies to legacy vertex COLOR/BCOLOR attributes,
    // not generic varyings or fragment outputs. Their SPH maps/attribute
    // transfers are rejected explicitly by the shader translator until that
    // interface is implemented, so no supported draw consumes this clamp.
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/gallium/drivers/nouveau/nvc0/nvc0_state.c#L231-L233
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/gallium/drivers/nouveau/nvc0/nvc0_program.c#L53-L54
    if trigger.is_draw()
        && let Some(MaxwellThreeDFixedFunctionValue::PixelShaderSaturate(value)) = state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::PixelShaderSaturate)
            .value()
        && let Some(output) = value.first_enabled_output()
    {
        return Err(
            MaxwellLoweringError::UnsupportedPixelShaderSaturateSemantics {
                output,
                range: value
                    .clamp_range(output)
                    .expect("enabled output is within the eight-output register"),
            },
        );
    }
    if trigger.is_draw() && state.shader_bindings().has_enabled_pipeline() {
        let local_memory = state.shader_execution().shader_local_memory();
        if local_memory.region_is_partially_programmed() {
            return Err(MaxwellLoweringError::IncompleteDraw(
                "SET_SHADER_LOCAL_MEMORY_A-D",
            ));
        }
        if let Some(default_size_per_warp) = local_memory
            .default_size_per_warp()
            .value()
            .copied()
            .filter(|size| size.bytes() != 0)
        {
            if local_memory.address().is_none() || local_memory.size().is_none() {
                return Err(MaxwellLoweringError::IncompleteDraw(
                    "SET_SHADER_LOCAL_MEMORY_A-D",
                ));
            }
            return Err(
                MaxwellLoweringError::UnsupportedShaderLocalMemorySemantics {
                    default_size_per_warp,
                },
            );
        }
    }
    if trigger.is_draw()
        && state.color_reduction().enable().value()
            == Some(&MaxwellThreeDColorReductionThresholdsEnable::Enabled)
    {
        // NVIDIA exposes a dedicated activation method, so merely programming
        // a threshold is not enough to make it effective. Once explicitly
        // enabled, however, the current neutral pipeline cannot represent the
        // reduction decision and must stop before cache/backend effects.
        return Err(MaxwellLoweringError::UnsupportedColorReductionSemantics);
    }
    if trigger.is_draw() && state.constant_color_rendering().enabled().value() == Some(&true) {
        return Err(MaxwellLoweringError::UnsupportedConstantColorRenderingSemantics);
    }
    if trigger.is_draw()
        && state.shader_execution().api_mandated_early_z().value()
            == Some(&MaxwellThreeDApiMandatedEarlyZ::Enabled)
    {
        return Err(MaxwellLoweringError::UnsupportedApiMandatedEarlyZSemantics);
    }
    if trigger.is_draw()
        && state.coverage().post_ps_initial_coverage().value() == Some(&true)
        && state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::AlphaTestEnable)
            .value()
            == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        // The host implements alpha test through shader discard. Its
        // interaction with a pre-PS initial mask needs explicit lowering.
        return Err(MaxwellLoweringError::UnsupportedPostPsInitialCoverageSemantics);
    }
    if trigger.is_draw()
        && state.coverage().post_z_pixel_shader_imask().value()
            == Some(&MaxwellThreeDPostZPixelShaderImask::Enabled)
    {
        return Err(MaxwellLoweringError::UnsupportedPostZPixelShaderImaskSemantics);
    }
    if trigger.is_draw()
        && let Some(value) = state
            .shader_execution()
            .pixel_shader_interlock_control()
            .value()
            .copied()
            .filter(|value| value.conflict_detection_enabled())
    {
        return Err(MaxwellLoweringError::UnsupportedPixelShaderInterlockSemantics(value));
    }
    if !trigger.is_indexed()
        && trigger.is_draw()
        && let Some(base_vertex) = state
            .vertex_input()
            .assembly()
            .global_base_vertex_index()
            .value()
            .copied()
            .filter(|value| *value != 0)
    {
        // The neutral non-indexed draw currently has one first-vertex value,
        // which controls both vertex-buffer addressing and the shader-visible
        // vertex index. Maxwell's global base changes only the latter; mapping
        // it to first_vertex would therefore silently fetch different data.
        return Err(MaxwellLoweringError::UnsupportedGlobalBaseVertexIndex(
            base_vertex,
        ));
    }
    if !trigger.is_indexed()
        && trigger.is_draw()
        && let Some(base) = state
            .vertex_input()
            .assembly()
            .vertex_id_base()
            .value()
            .copied()
            .filter(|value| *value != 0)
    {
        // A zero base preserves the existing shader-visible index. Nonzero
        // bases need a separate shader-ID adjustment, not first_vertex (which
        // would also change vertex-buffer fetches). Keep this register in the
        // prepared-draw semantic revision so a cached draw cannot bypass it.
        return Err(MaxwellLoweringError::UnsupportedVertexIdBase(base));
    }
    if trigger.is_draw()
        && state.coverage().csaa_enable().value() == Some(&MaxwellThreeDCsaaEnable::Enabled)
    {
        return Err(MaxwellLoweringError::UnsupportedCsaaSemantics);
    }
    // Dither footprint is only configuration while alpha-to-coverage is off.
    // Neither coverage generation (including dithering) nor alpha-to-one is
    // represented by the current neutral pipeline. Reject their activation,
    // including after a cached draw, rather than ignoring a stored selector.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
    if trigger.is_draw()
        && let Some(MaxwellThreeDFixedFunctionValue::AlphaControl {
            alpha_to_coverage,
            alpha_to_one,
        }) = state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::AlphaToCoverageEnable)
            .value()
        && (*alpha_to_coverage || *alpha_to_one)
    {
        return Err(MaxwellLoweringError::UnsupportedAntiAliasAlphaControl {
            alpha_to_coverage: *alpha_to_coverage,
            alpha_to_one: *alpha_to_one,
        });
    }
    if trigger.is_draw()
        && let Some(value) = state
            .coverage()
            .coverage_to_color()
            .value()
            .copied()
            .filter(|value| value.enabled())
    {
        return Err(MaxwellLoweringError::UnsupportedCoverageToColorSemantics(
            value,
        ));
    }
    if trigger.is_draw()
        && let Some(value) = state
            .coverage()
            .alpha_to_coverage_override()
            .value()
            .copied()
            .filter(|value| value.raw() != 0)
    {
        return Err(MaxwellLoweringError::UnsupportedAlphaToCoverageOverrideSemantics(value));
    }
    if trigger.is_draw()
        && state.coverage().tir_mode().value() == Some(&MaxwellThreeDTirMode::RasterNTargetM)
    {
        return Err(MaxwellLoweringError::UnsupportedTirSemantics {
            control: state.coverage().tir_control().value().copied(),
        });
    }
    if trigger.is_draw()
        && let Some(value) = state
            .coverage()
            .hybrid_anti_alias_control()
            .value()
            .copied()
            .filter(|value| !value.is_single_pass_per_fragment())
    {
        return Err(MaxwellLoweringError::UnsupportedHybridAntiAliasSemantics(
            value,
        ));
    }
    if trigger.is_draw() {
        multisample::validate(state)?;
    }
    if trigger.is_draw() {
        // Validate geometry here; shader-dependent depth observability is
        // checked when preparing the actual draw below.
        draw_viewport_transform_with_depth_usage(state, false)?;
    }
    if trigger.is_draw()
        && state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::WindowClipEnable)
            .value()
            == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        return Err(MaxwellLoweringError::UnsupportedWindowClipSemantics);
    }
    if trigger.is_draw()
        && state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable)
            .value()
            == Some(&MaxwellThreeDFixedFunctionValue::ClipIdTestEnable(
                MaxwellThreeDClipIdTestEnable::Enabled,
            ))
    {
        return Err(MaxwellLoweringError::UnsupportedClipIdTestSemantics);
    }
    if trigger.is_draw() {
        if state.viewport().pixel_center().value()
            == Some(&MaxwellThreeDViewportPixelCenter::Integers)
        {
            return Err(
                MaxwellLoweringError::UnsupportedViewportPixelCenterSemantics(
                    MaxwellThreeDViewportPixelCenter::Integers,
                ),
            );
        }
        if let Some(mode) = state
            .raster()
            .fill_via_triangle()
            .value()
            .copied()
            .filter(|mode| *mode == MaxwellThreeDFillViaTriangleMode::FillAll)
        {
            return Err(MaxwellLoweringError::UnsupportedFillViaTriangleSemantics(
                mode,
            ));
        }
        if state.raster().conservative_raster().value()
            == Some(&MaxwellThreeDConservativeRasterEnable::Enabled)
        {
            return Err(MaxwellLoweringError::UnsupportedConservativeRasterSemantics);
        }
        if state.shader_bindings().has_enabled_pipeline()
            && state
                .shader_bindings()
                .program_region()
                .is_partially_programmed()
        {
            return Err(MaxwellLoweringError::IncompleteDraw(
                "SET_PROGRAM_REGION_A/B",
            ));
        }
        if state.generated_primitive() == Some(super::threed::state::GeneratedPrimitive::Points) {
            if let Some(value) = state
                .raster()
                .attribute_point_size()
                .value()
                .copied()
                .filter(|value| value.enabled())
            {
                return Err(
                    MaxwellLoweringError::UnsupportedAttributePointSizeSemantics {
                        slot: value.slot(),
                    },
                );
            }
            if state.raster().point_sprite_enable().value() == Some(&true) {
                return Err(MaxwellLoweringError::UnsupportedPointSpriteSemantics);
            }
            if state.raster().anti_aliased_point_enable().value() == Some(&true) {
                return Err(MaxwellLoweringError::UnsupportedAntiAliasedPointSemantics);
            }
            if let Some(select) = state
                .raster()
                .point_sprite_select()
                .value()
                .copied()
                .filter(|select| select.affects_point_coordinates())
            {
                return Err(
                    MaxwellLoweringError::UnsupportedPointSpriteCoordinatesSemantics(select),
                );
            }
            if let Some(mode) = state.raster().point_center_mode().value().copied() {
                return Err(MaxwellLoweringError::UnsupportedPointCenterSemantics(mode));
            }
        }
        validate_direct_line_rasterization_state(state)?;
        raster_state = Some(raster::draw_state(state)?);
    }
    Ok((tessellation, raster_state))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn draw_fragment_bounds_compose_surface_origin_extent_and_scissor() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        use nixe_gpu::ScissorRect;
        let mut channel = three_d_channel();
        assert_eq!(
            super::draw_scissor_region(channel.three_d(), 64, 32).unwrap(),
            ScissorRect {
                x: 0,
                y: 0,
                width: 64,
                height: 32
            }
        );
        program_three_d(&mut channel, 0x0ff4, (40 << 16) | 5);
        assert!(matches!(
            super::draw_scissor_region(channel.three_d(), 64, 32),
            Err(MaxwellLoweringError::IncompleteDraw(_))
        ));
        program_three_d(&mut channel, 0x0ff8, (20 << 16) | 7);
        assert_eq!(
            super::draw_scissor_region(channel.three_d(), 64, 32).unwrap(),
            ScissorRect {
                x: 5,
                y: 7,
                width: 40,
                height: 20
            }
        );
        program_three_d(&mut channel, 0x0e00, 1);
        assert!(matches!(
            super::draw_scissor_region(channel.three_d(), 64, 32),
            Err(MaxwellLoweringError::IncompleteDraw(_))
        ));
        program_three_d(&mut channel, 0x0e04, (50 << 16) | 12);
        program_three_d(&mut channel, 0x0e08, (23 << 16) | 2);
        assert_eq!(
            super::draw_surface_region(channel.three_d(), 64, 32)
                .unwrap()
                .scissor(64, 32),
            ScissorRect {
                x: 5,
                y: 7,
                width: 40,
                height: 20
            },
            "dynamic scissor must not change attachment load/store bounds"
        );
        assert_eq!(
            super::draw_scissor_region(channel.three_d(), 64, 32).unwrap(),
            ScissorRect {
                x: 12,
                y: 7,
                width: 33,
                height: 16
            }
        );
        program_three_d(&mut channel, 0x0e04, (80 << 16) | 70);
        assert_eq!(
            super::draw_scissor_region(channel.three_d(), 64, 32).unwrap(),
            ScissorRect {
                x: 64,
                y: 7,
                width: 0,
                height: 16
            }
        );
        program_three_d(&mut channel, 0x0e00, 0);
        assert_eq!(
            super::draw_scissor_region(channel.three_d(), 16, 12).unwrap(),
            ScissorRect {
                x: 5,
                y: 7,
                width: 11,
                height: 5
            }
        );
        program_three_d(&mut channel, 0x0ff4, 100);
        assert_eq!(
            super::draw_scissor_region(channel.three_d(), 64, 32).unwrap(),
            ScissorRect {
                x: 64,
                y: 7,
                width: 0,
                height: 20
            }
        );
    }

    #[test]
    fn snap_grid_precision_only_rejects_the_consumed_viewport() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        let mut channel = three_d_channel();
        program_three_d(&mut channel, 0x0a1c, 0);
        program_three_d(&mut channel, 0x0bfc, 0x1f1f);
        assert!(super::draw_viewport_transform(channel.three_d()).is_ok());
        let register = channel.three_d().fixed_function().viewport()[15].snap_grid_precision();
        assert_eq!(register.value(), Some(&[31, 31]));
        assert_eq!(
            register.source().unwrap().method(),
            nixe_gpu::GpuMethodId(0x0bfc)
        );
        program_three_d(&mut channel, 0x0a1c, 0x0302);
        assert!(matches!(
            super::draw_viewport_transform(channel.three_d()),
            Err(MaxwellLoweringError::UnsupportedViewportSnapGridPrecision {
                viewport: 0,
                precision: [2, 3]
            })
        ));
        program_three_d(&mut channel, 0x0a1c, 0);
        assert!(super::draw_viewport_transform(channel.three_d()).is_ok());
    }

    #[test]
    fn lower_left_origin_reflects_viewport_and_scissor_independently_of_facing() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        let mut channel = three_d_channel();
        for (method, value) in [
            (0x0a00, 32.0_f32.to_bits()),
            (0x0a04, 16.0_f32.to_bits()),
            (0x0a08, 1.0_f32.to_bits()),
            (0x0a0c, 37.0_f32.to_bits()),
            (0x0a10, 19.0_f32.to_bits()),
            (0x0a14, 0.0_f32.to_bits()),
            (0x193c, 1),
            (0x0d7c, 1),
            (0x192c, 1),
            (0x13ac, 1),
            (0x0ff4, 128 << 16),
            (0x0ff8, 64 << 16),
            (0x0e00, 1),
            (0x0e04, (100 << 16) | 8),
            (0x0e08, (20 << 16) | 4),
        ] {
            program_three_d(&mut channel, method, value);
        }
        for origin in [1, 0x11] {
            program_three_d(&mut channel, 0x13ac, origin);
            let viewport = super::draw_viewport_transform(channel.three_d())
                .unwrap()
                .unwrap();
            assert_eq!(viewport.scale(), [32.0, -16.0, 1.0]);
            assert_eq!(viewport.offset(), [37.0, 45.0, 0.0]);
            assert_eq!(
                super::draw_scissor_region(channel.three_d(), 128, 64).unwrap(),
                nixe_gpu::ScissorRect {
                    x: 8,
                    y: 44,
                    width: 92,
                    height: 16
                }
            );
        }
        program_three_d(&mut channel, 0x0a18, 0x6430);
        assert_eq!(
            super::draw_viewport_transform(channel.three_d())
                .unwrap()
                .unwrap()
                .scale()[1],
            16.0
        );
    }

    #[test]
    fn viewport_depth_mode_and_unbounded_pixel_limits_are_independent() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        let mut channel = three_d_channel();
        for (method, argument) in [
            (0x0a00, 640.0_f32.to_bits()),
            (0x0a04, 360.0_f32.to_bits()),
            (0x0a08, 0.5_f32.to_bits()),
            (0x0a0c, 640.0_f32.to_bits()),
            (0x0a10, 360.0_f32.to_bits()),
            (0x0a14, 0.5_f32.to_bits()),
            (0x0c08, f32::NEG_INFINITY.to_bits()),
            (0x0c0c, f32::INFINITY.to_bits()),
            (0x192c, 1),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        let initialized = super::draw_viewport_transform(channel.three_d())
            .unwrap()
            .unwrap();
        assert_eq!(initialized.depth_range(), [0.0, 1.0]);
        assert!(initialized.depth_clip_negative_one_to_one());
        program_three_d(&mut channel, 0x193c, 0x281c);
        let gl = super::draw_viewport_transform(channel.three_d())
            .unwrap()
            .unwrap();
        assert_eq!(gl.depth_range(), [0.0, 1.0]);
        assert!(gl.depth_clip_negative_one_to_one());
        program_three_d(&mut channel, 0x193c, 0x281d);
        let independent = super::draw_viewport_transform(channel.three_d())
            .unwrap()
            .unwrap();
        assert_eq!(independent.depth_range(), [0.0, 1.0]);
        assert!(independent.depth_clip_negative_one_to_one());
        program_three_d(&mut channel, 0x0d7c, 1);
        let zero = super::draw_viewport_transform(channel.three_d())
            .unwrap()
            .unwrap();
        assert_eq!(zero.depth_range(), [0.5, 1.0]);
        assert!(!zero.depth_clip_negative_one_to_one());
        program_three_d(&mut channel, 0x0c08, 0.75_f32.to_bits());
        assert!(matches!(
            super::draw_viewport_transform(channel.three_d()),
            Err(MaxwellLoweringError::UnsupportedViewportPixelDepthBounds)
        ));
    }

    #[test]
    fn viewport_y_swizzle_composes_with_both_scale_signs_and_preserves_offsets() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        let mut channel = three_d_channel();
        program_three_d(&mut channel, 0x193c, 0);
        for (method, argument) in [
            (0x0a00, 32.0_f32.to_bits()),
            (0x0a08, 0.5_f32.to_bits()),
            (0x0a0c, 37.0_f32.to_bits()),
            (0x0a10, 19.0_f32.to_bits()),
            (0x0a14, 0.5_f32.to_bits()),
            (0x0c08, 0.0_f32.to_bits()),
            (0x0c0c, 1.0_f32.to_bits()),
            (0x192c, 1),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        for scale in [-16.0_f32, 16.0] {
            program_three_d(&mut channel, 0x0a04, scale.to_bits());
            for (swizzle, expected_y) in [(0x6420, scale), (0x6430, -scale)] {
                program_three_d(&mut channel, 0x0a18, swizzle);
                // Unconsumed viewport state must not reject viewport zero.
                program_three_d(&mut channel, 0x0a38, 0x7654);
                let transform = super::draw_viewport_transform(channel.three_d())
                    .unwrap()
                    .unwrap();
                assert_eq!(transform.scale(), [32.0, expected_y, 0.5]);
                assert_eq!(transform.offset(), [37.0, 19.0, 0.5]);
                assert_eq!(transform.depth_range(), [0.0, 1.0]);
            }
        }
        program_three_d(&mut channel, 0x0a18, 0x6421);
        assert!(matches!(
            super::draw_viewport_transform(channel.three_d()),
            Err(
                MaxwellLoweringError::UnsupportedViewportCoordinateSwizzleSemantics {
                    viewport: 0,
                    ..
                }
            )
        ));
        program_three_d(&mut channel, 0x0a18, 0x6430);
        program_three_d(&mut channel, 0x192c, 0);
        assert!(matches!(
            super::draw_viewport_transform(channel.three_d()),
            Err(
                MaxwellLoweringError::UnsupportedViewportCoordinateSwizzleSemantics {
                    viewport: 0,
                    ..
                }
            )
        ));
    }
}
