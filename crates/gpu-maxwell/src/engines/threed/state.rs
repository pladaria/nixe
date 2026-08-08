//! Typed, source-preserving `MAXWELL_B` register state.
//!
//! Register writes remain separate from their later draw-time interpretation.
//! In particular, an undocumented hardware reset must stay `Unset`: zero is
//! not a reset value unless a pinned public source establishes that fact.

use crate::MaxwellMethodSource;

use super::{
    MaxwellThreeDCoverageState, MaxwellThreeDCoverageStateWrite, MaxwellThreeDFixedFunctionState,
    MaxwellThreeDFixedFunctionWrite, MaxwellThreeDLineState, MaxwellThreeDLineStateWrite,
    MaxwellThreeDRenderEnableState, MaxwellThreeDRenderEnableStateWrite,
    MaxwellThreeDRenderTargetState, MaxwellThreeDRenderTargetWrite,
    MaxwellThreeDShaderBindingState, MaxwellThreeDShaderBindingWrite,
    MaxwellThreeDShaderExecutionState, MaxwellThreeDShaderExecutionStateWrite,
    MaxwellThreeDVertexInputState, MaxwellThreeDVertexInputWrite, MaxwellThreeDZCullState,
    MaxwellThreeDZCullStateWrite,
};

/// How a modeled Maxwell register acquired its current value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDRegisterOrigin {
    /// No verified reset or method write establishes a value.
    Unset,
    /// A pinned public source establishes the hardware reset value.
    VerifiedReset,
    /// A validated guest method programmed the register.
    Programmed,
}

/// One typed register with explicit validity and optional write provenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDRegister<T> {
    origin: MaxwellThreeDRegisterOrigin,
    raw: Option<u32>,
    value: Option<T>,
    source: Option<MaxwellMethodSource>,
}

impl<T> MaxwellThreeDRegister<T> {
    /// Returns whether the value is absent, sourced from a verified reset, or
    /// explicitly programmed. Callers must not treat `Unset` as zero.
    #[must_use]
    pub const fn origin(&self) -> MaxwellThreeDRegisterOrigin {
        self.origin
    }

    /// Exact method/reset bits retained before later semantic interpretation.
    #[must_use]
    pub const fn raw(&self) -> Option<u32> {
        self.raw
    }

    /// Typed value, available only when the register has a valid origin.
    #[must_use]
    pub const fn value(&self) -> Option<&T> {
        self.value.as_ref()
    }

    /// Source of a programmed value. Reset and unset states have no method.
    #[must_use]
    pub const fn source(&self) -> Option<MaxwellMethodSource> {
        self.source
    }
}

impl<T> Default for MaxwellThreeDRegister<T> {
    fn default() -> Self {
        Self {
            origin: MaxwellThreeDRegisterOrigin::Unset,
            raw: None,
            value: None,
            source: None,
        }
    }
}

impl<T> MaxwellThreeDRegister<T> {
    pub(super) const fn programmed(raw: u32, value: T, source: MaxwellMethodSource) -> Self {
        Self {
            origin: MaxwellThreeDRegisterOrigin::Programmed,
            raw: Some(raw),
            value: Some(value),
            source: Some(source),
        }
    }
}

/// Exact IEEE-754 bits written to `SET_POINT_SIZE`.
///
/// Draw-time validation deliberately happens in the later semantic snapshot;
/// preserving the register bits here does not claim that every bit pattern is
/// a usable rasterizer point size.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDPointSize(u32);

impl MaxwellThreeDPointSize {
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// Raw rasterization registers whose derived combinations are validated later.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDRasterState {
    point_size: MaxwellThreeDRegister<MaxwellThreeDPointSize>,
}

impl MaxwellThreeDRasterState {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            point_size: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
        }
    }

    #[must_use]
    pub const fn point_size(&self) -> &MaxwellThreeDRegister<MaxwellThreeDPointSize> {
        &self.point_size
    }
}

/// Clip-space Z range selected before viewport transformation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDViewportZClipRange {
    NegativeWToPositiveW = 0,
    ZeroToPositiveW = 1,
}

impl MaxwellThreeDViewportZClipRange {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::NegativeWToPositiveW),
            1 => Some(Self::ZeroToPositiveW),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Raw viewport registers whose complete combinations are validated later.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDViewportState {
    z_clip_range: MaxwellThreeDRegister<MaxwellThreeDViewportZClipRange>,
}

impl MaxwellThreeDViewportState {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            z_clip_range: MaxwellThreeDRegister {
                origin: MaxwellThreeDRegisterOrigin::Unset,
                raw: None,
                value: None,
                source: None,
            },
        }
    }

    #[must_use]
    pub const fn z_clip_range(&self) -> &MaxwellThreeDRegister<MaxwellThreeDViewportZClipRange> {
        &self.z_clip_range
    }
}

/// Complete currently modeled state of one channel's `MAXWELL_B` engine.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDState {
    render_targets: Box<MaxwellThreeDRenderTargetState>,
    fixed_function: Box<MaxwellThreeDFixedFunctionState>,
    vertex_input: Box<MaxwellThreeDVertexInputState>,
    shader_bindings: Box<MaxwellThreeDShaderBindingState>,
    raster: MaxwellThreeDRasterState,
    viewport: MaxwellThreeDViewportState,
    render_enable: MaxwellThreeDRenderEnableState,
    shader_execution: MaxwellThreeDShaderExecutionState,
    coverage: MaxwellThreeDCoverageState,
    line: MaxwellThreeDLineState,
    zcull: MaxwellThreeDZCullState,
}

impl MaxwellThreeDState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn render_targets(&self) -> &MaxwellThreeDRenderTargetState {
        &self.render_targets
    }

    #[must_use]
    pub fn fixed_function(&self) -> &MaxwellThreeDFixedFunctionState {
        &self.fixed_function
    }

    #[must_use]
    pub fn vertex_input(&self) -> &MaxwellThreeDVertexInputState {
        &self.vertex_input
    }

    #[must_use]
    pub fn shader_bindings(&self) -> &MaxwellThreeDShaderBindingState {
        &self.shader_bindings
    }

    #[must_use]
    pub const fn raster(&self) -> &MaxwellThreeDRasterState {
        &self.raster
    }

    #[must_use]
    pub const fn viewport(&self) -> &MaxwellThreeDViewportState {
        &self.viewport
    }

    #[must_use]
    pub const fn render_enable(&self) -> &MaxwellThreeDRenderEnableState {
        &self.render_enable
    }

    #[must_use]
    pub const fn shader_execution(&self) -> &MaxwellThreeDShaderExecutionState {
        &self.shader_execution
    }

    #[must_use]
    pub const fn coverage(&self) -> &MaxwellThreeDCoverageState {
        &self.coverage
    }

    #[must_use]
    pub const fn line(&self) -> &MaxwellThreeDLineState {
        &self.line
    }

    #[must_use]
    pub const fn zcull(&self) -> &MaxwellThreeDZCullState {
        &self.zcull
    }

    pub(crate) fn pipeline_dependencies(&self, active_color_targets: &[u8]) -> Box<[Option<u32>]> {
        let mut dependencies = Vec::new();
        self.fixed_function
            .append_pipeline_dependencies(&mut dependencies, active_color_targets);
        self.vertex_input
            .append_pipeline_dependencies(&mut dependencies);
        self.shader_bindings
            .append_pipeline_dependencies(&mut dependencies);
        dependencies.push(self.raster.point_size.raw());
        dependencies.push(self.viewport.z_clip_range.raw());
        dependencies.push(self.render_targets.color_target_selection().raw());
        dependencies.push(self.coverage.csaa_enable().raw());
        dependencies.push(self.line.aliased_line_width_enable().raw());
        dependencies.into_boxed_slice()
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDStateWrite) {
        match write {
            MaxwellThreeDStateWrite::PointSize { value, source } => {
                self.raster.point_size =
                    MaxwellThreeDRegister::programmed(value.bits(), value, source);
            }
            MaxwellThreeDStateWrite::ViewportZClip { value, source } => {
                self.viewport.z_clip_range =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDStateWrite::RenderTarget(write) => self.render_targets.apply(write),
            MaxwellThreeDStateWrite::FixedFunction(write) => self.fixed_function.apply(write),
            MaxwellThreeDStateWrite::VertexInput(write) => self.vertex_input.apply(write),
            MaxwellThreeDStateWrite::ShaderBinding(write) => self.shader_bindings.apply(write),
            MaxwellThreeDStateWrite::RenderEnable(write) => self.render_enable.apply(write),
            MaxwellThreeDStateWrite::ShaderExecution(write) => self.shader_execution.apply(write),
            MaxwellThreeDStateWrite::Coverage(write) => self.coverage.apply(write),
            MaxwellThreeDStateWrite::Line(write) => self.line.apply(write),
            MaxwellThreeDStateWrite::ZCull(write) => self.zcull.apply(write),
        }
    }

    pub(in crate::engines) fn validate_cross_registers(
        &self,
    ) -> Result<(), MaxwellThreeDStateValidationError> {
        for target in self.render_targets.color() {
            if target.kind().value() == Some(&super::MaxwellThreeDImageKind::ThreeDimensional)
                && target.layer().value().is_some_and(|layer| *layer != 0)
            {
                return Err(MaxwellThreeDStateValidationError {
                    source: target.layer().source().or_else(|| target.kind().source()),
                    reason: "a three-dimensional color target cannot select an array layer",
                });
            }
        }
        for stream in self.vertex_input.streams() {
            if let (Some(address), Some(limit)) = (stream.address(), stream.limit())
                && address.get() > limit.get()
            {
                return Err(MaxwellThreeDStateValidationError {
                    source: stream
                        .limit_lower()
                        .source()
                        .or_else(|| stream.limit_upper().source()),
                    reason: "a vertex stream limit precedes its start address",
                });
            }
            if stream.instanced().value() == Some(&true) && stream.frequency().value() == Some(&0) {
                return Err(MaxwellThreeDStateValidationError {
                    source: stream
                        .frequency()
                        .source()
                        .or_else(|| stream.instanced().source()),
                    reason: "an instanced vertex stream cannot have zero frequency",
                });
            }
        }
        for attribute in self.vertex_input.attributes() {
            let Some(format) = attribute.value().filter(|format| format.enabled()) else {
                continue;
            };
            if self.vertex_input.streams()[format.stream() as usize]
                .format()
                .value()
                .is_some_and(|stream| !stream.enabled())
            {
                return Err(MaxwellThreeDStateValidationError {
                    source: attribute.source(),
                    reason: "an enabled vertex attribute references an explicitly disabled stream",
                });
            }
            let stream = &self.vertex_input.streams()[format.stream() as usize];
            if let (Some(address), Some(limit), Some(component_widths)) =
                (stream.address(), stream.limit(), format.component_widths())
            {
                let required =
                    u64::from(format.offset()).checked_add(u64::from(component_widths.byte_size()));
                let available = limit
                    .get()
                    .checked_sub(address.get())
                    .and_then(|distance| distance.checked_add(1));
                if required.is_none() || required > available {
                    return Err(MaxwellThreeDStateValidationError {
                        source: attribute.source(),
                        reason: "a vertex attribute format exceeds its stream range",
                    });
                }
            }
        }
        let index = self.vertex_input.index();
        if let (Some(upper), Some(lower), Some(limit_upper), Some(limit_lower)) = (
            index.address_upper().value(),
            index.address_lower().value(),
            index.limit_upper().value(),
            index.limit_lower().value(),
        ) {
            let address = (u64::from(*upper) << 32) | u64::from(*lower);
            let limit = (u64::from(*limit_upper) << 32) | u64::from(*limit_lower);
            if address > limit {
                return Err(MaxwellThreeDStateValidationError {
                    source: index.limit_lower().source(),
                    reason: "the index-buffer limit precedes its start address",
                });
            }
            if let Some(size) = index.element_size().value()
                && (address % size.bytes() != 0
                    || limit
                        .checked_add(1)
                        .is_some_and(|end| end % size.bytes() != 0))
            {
                return Err(MaxwellThreeDStateValidationError {
                    source: index.element_size().source(),
                    reason: "the index-buffer range is not aligned to its element size",
                });
            }
            if let (Some(size), Some(first)) = (index.element_size().value(), index.first().value())
            {
                let first_address = u64::from(*first)
                    .checked_mul(size.bytes())
                    .and_then(|offset| address.checked_add(offset));
                if first_address.is_none_or(|first_address| first_address > limit) {
                    return Err(MaxwellThreeDStateValidationError {
                        source: index.first().source(),
                        reason: "the first index lies outside the index-buffer range",
                    });
                }
            }
        }
        let bindings = self.shader_bindings();
        let mut stages = [false; 6];
        for pipeline in bindings.pipeline() {
            if pipeline.enabled().value() != Some(&true) {
                continue;
            }
            let Some(stage) = pipeline.stage().value() else {
                continue;
            };
            let stage_index = *stage as usize;
            if stages[stage_index] {
                return Err(MaxwellThreeDStateValidationError {
                    source: pipeline.stage().source(),
                    reason: "two enabled pipeline slots expose the same shader stage",
                });
            }
            stages[stage_index] = true;
        }
        for pool in [bindings.texture_headers(), bindings.samplers()] {
            if let (Some(address), Some(maximum_index)) =
                (pool.address(), pool.maximum_index().value())
            {
                let byte_count = u64::from(*maximum_index)
                    .checked_add(1)
                    .and_then(|count| count.checked_mul(32));
                if address.get() & 31 != 0
                    || byte_count
                        .and_then(|size| address.get().checked_add(size))
                        .is_none_or(|end| end > (1_u64 << 40))
                {
                    return Err(MaxwellThreeDStateValidationError {
                        source: pool
                            .maximum_index()
                            .source()
                            .or_else(|| pool.address_lower().source()),
                        reason: "a descriptor pool address/range is misaligned or overflows",
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::engines) struct MaxwellThreeDStateValidationError {
    pub source: Option<MaxwellMethodSource>,
    pub reason: &'static str,
}

/// One checked `MAXWELL_B` register transition ready for candidate state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDStateWrite {
    PointSize {
        value: MaxwellThreeDPointSize,
        source: MaxwellMethodSource,
    },
    ViewportZClip {
        value: MaxwellThreeDViewportZClipRange,
        source: MaxwellMethodSource,
    },
    RenderTarget(MaxwellThreeDRenderTargetWrite),
    FixedFunction(MaxwellThreeDFixedFunctionWrite),
    VertexInput(MaxwellThreeDVertexInputWrite),
    ShaderBinding(MaxwellThreeDShaderBindingWrite),
    RenderEnable(MaxwellThreeDRenderEnableStateWrite),
    ShaderExecution(MaxwellThreeDShaderExecutionStateWrite),
    Coverage(MaxwellThreeDCoverageStateWrite),
    Line(MaxwellThreeDLineStateWrite),
    ZCull(MaxwellThreeDZCullStateWrite),
}
