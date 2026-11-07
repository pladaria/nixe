use super::*;
use crate::{
    MaxwellThreeDFixedFunctionRegister as R, MaxwellThreeDFixedFunctionValue as V,
    MaxwellThreeDSamplerBindingMode, MaxwellThreeDState, MaxwellThreeDViewportScaleOffsetEnable,
};

/// Shader selection facts, without capturing unused constant-buffer bindings.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphicsSource {
    pub stage: Stage,
    pub address: u64,
    pub group: Option<u8>,
    start_address: u32,
    texture_cbuf: u32,
    viewport_transform: Option<u32>,
    geometry_passthrough_mask: Option<[u32; 8]>,
}

pub fn graphics_source(
    state: &MaxwellThreeDState,
    pipeline_index: usize,
) -> Result<GraphicsSource> {
    let bindings = state.shader_bindings();
    if pipeline_index >= crate::MAXWELL_PIPELINE_SHADER_COUNT {
        return Err(invalid("Graphics pipeline slot is out of range"));
    }
    let pipeline = bindings.pipeline(pipeline_index);
    if pipeline.enabled().value() != Some(&true) {
        return Err(invalid("Graphics pipeline slot is disabled"));
    }
    let stage = match pipeline
        .stage()
        .value()
        .ok_or_else(|| invalid("Graphics stage selector is missing"))?
    {
        MaxwellShaderStage::VertexCullBeforeFetch => Stage::VertexA,
        MaxwellShaderStage::Vertex => Stage::VertexB,
        MaxwellShaderStage::TessellationInit => Stage::TessellationControl,
        MaxwellShaderStage::Tessellation => Stage::TessellationEval,
        MaxwellShaderStage::Geometry => Stage::Geometry,
        MaxwellShaderStage::Pixel => Stage::Fragment,
        MaxwellShaderStage::Compute => {
            return Err(invalid("Compute cannot occupy a graphics pipeline slot"));
        }
    };
    let base = bindings
        .program_region()
        .address()
        .ok_or_else(|| invalid("Graphics program region is missing"))?
        .get();
    let offset = *pipeline
        .program_offset()
        .value()
        .ok_or_else(|| invalid("Graphics program offset is missing"))?;
    let address = base
        .checked_add(u64::from(offset))
        .ok_or_else(|| invalid("Graphics program address overflows"))?;
    let texture_cbuf = bindings
        .bindless_texture_constant_buffer_slot()
        .value()
        .copied()
        .unwrap_or(0);
    Ok(GraphicsSource {
        stage,
        address,
        group: pipeline.effective_group(),
        start_address: offset & 31,
        texture_cbuf: u32::from(texture_cbuf),
        viewport_transform: viewport_transform(state),
        geometry_passthrough_mask: (stage == Stage::Geometry)
            .then(|| bindings.geometry_passthrough_mask())
            .flatten(),
    })
}

pub fn graphics_request(
    state: &MaxwellThreeDState,
    pipeline_index: usize,
) -> Result<CaptureRequest> {
    let source = graphics_source(state, pipeline_index)?;
    let bindings = state.shader_bindings();
    let mut capture_bindings = CaptureBindings {
        texture_pool: texture_pool(state),
        viewport_transform: source.viewport_transform,
        geometry_passthrough_mask: source.geometry_passthrough_mask,
        ..Default::default()
    };
    if let Some(group) = source.group {
        let group = bindings
            .groups()
            .get(group as usize)
            .ok_or_else(|| invalid("Graphics constant buffer group is out of range"))?;
        for index in 0..group.constant_buffers().len() {
            capture_bindings.constant_buffers[index] = constant_buffer(state, source.group, index);
        }
    }

    Ok(CaptureRequest {
        address: source.address,
        metadata: EnvironmentMetadata {
            stage: source.stage,
            start_address: ProgramOffset(source.start_address),
            texture_bound_buffer: CbufIndex(source.texture_cbuf),
            is_proprietary_driver: source.texture_cbuf == 2,
            ..Default::default()
        },
        bindings: capture_bindings,
    })
}

fn texture_pool(state: &MaxwellThreeDState) -> Option<TexturePool> {
    let bindings = state.shader_bindings();
    let pool = bindings.texture_headers();
    pool.address()
        .zip(pool.maximum_index().value().copied())
        .zip(bindings.sampler_binding().value())
        .map(|((address, maximum_index), mode)| TexturePool {
            address: address.get(),
            maximum_index,
            via_header_index: *mode == MaxwellThreeDSamplerBindingMode::ViaTextureHeader,
        })
}
fn viewport_transform(state: &MaxwellThreeDState) -> Option<u32> {
    match state
        .fixed_function()
        .register(R::ViewportScaleOffsetEnable)
        .value()
    {
        Some(V::ViewportScaleOffsetEnable(value)) => Some(u32::from(
            *value == MaxwellThreeDViewportScaleOffsetEnable::Enabled,
        )),
        _ => None,
    }
}
fn constant_buffer(
    state: &MaxwellThreeDState,
    group: Option<u8>,
    index: usize,
) -> Option<BufferBinding> {
    let binding = group
        .and_then(|group| state.shader_bindings().groups().get(group as usize))
        .and_then(|group| group.constant_buffers().get(index))
        .copied()
        .flatten()?;
    Some(BufferBinding {
        address: binding.address().get(),
        size: u64::from(binding.size()),
    })
}
