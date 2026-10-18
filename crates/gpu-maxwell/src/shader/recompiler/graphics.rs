use super::*;
use crate::{
    MaxwellThreeDFixedFunctionRegister as R, MaxwellThreeDFixedFunctionValue as V,
    MaxwellThreeDSamplerBindingMode, MaxwellThreeDState, MaxwellThreeDViewportScaleOffsetEnable,
};

pub fn graphics_request(
    state: &MaxwellThreeDState,
    pipeline_index: usize,
) -> Result<CaptureRequest> {
    let bindings = state.shader_bindings();
    let pipeline = bindings
        .pipeline()
        .get(pipeline_index)
        .ok_or_else(|| invalid("Graphics pipeline slot is out of range"))?;
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
    let pool = bindings.texture_headers();
    let texture_pool = pool
        .address()
        .zip(pool.maximum_index().value().copied())
        .zip(bindings.sampler_binding().value())
        .map(|((address, maximum_index), mode)| TexturePool {
            address: address.get(),
            maximum_index,
            via_header_index: *mode == MaxwellThreeDSamplerBindingMode::ViaTextureHeader,
        });
    let viewport_transform = match state
        .fixed_function()
        .register(R::ViewportScaleOffsetEnable)
        .value()
    {
        Some(V::ViewportScaleOffsetEnable(value)) => Some(u32::from(
            *value == MaxwellThreeDViewportScaleOffsetEnable::Enabled,
        )),
        _ => None,
    };
    let mut capture_bindings = CaptureBindings {
        texture_pool,
        viewport_transform,
        geometry_passthrough_mask: bindings.geometry_passthrough_mask(),
        ..Default::default()
    };
    if let Some(group) = pipeline.effective_group() {
        let group = bindings
            .groups()
            .get(group as usize)
            .ok_or_else(|| invalid("Graphics constant buffer group is out of range"))?;
        for (index, binding) in group.constant_buffers().iter().enumerate() {
            if let Some(binding) = binding.filter(|binding| binding.enabled()) {
                match binding.address().zip(binding.size()) {
                    Some((address, size)) => {
                        capture_bindings.constant_buffers[index] = Some(BufferBinding {
                            address: address.get(),
                            size: u64::from(size),
                        })
                    }
                    None => {
                        capture_bindings.invalid_constant_buffers[index] =
                            Some("enabled graphics binding has incomplete address or size")
                    }
                }
            }
        }
    }
    Ok(CaptureRequest {
        address,
        metadata: EnvironmentMetadata {
            stage,
            start_address: ProgramOffset(offset & 31),
            texture_bound_buffer: CbufIndex(u32::from(texture_cbuf)),
            is_proprietary_driver: texture_cbuf == 2,
            ..Default::default()
        },
        bindings: capture_bindings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn graphics_state_captures_header_and_tracks_passthrough_register_changes() {
        let (allocation, space, base) = crate::shader::test_support::mapped_memory();
        allocation
            .write(0x10 + 0x58, &0xe300_0000_0007_000fu64.to_le_bytes())
            .unwrap();
        let mut channel = crate::engines::tests::three_d_channel();
        for (method, value) in [
            (0x1608, (base >> 32) as u32),
            (0x160c, base as u32),
            (0x2000, 0x11),
            (0x2004, 0x10),
            (0x200c, 4),
        ] {
            crate::engines::tests::program_three_d(&mut channel, method, value);
        }
        for index in 0..8 {
            crate::engines::tests::program_three_d(&mut channel, 0x1240 + index * 4, index + 1);
        }
        let request = graphics_request(channel.three_d(), 0).unwrap();
        assert_eq!(
            request.bindings.geometry_passthrough_mask,
            Some([1, 2, 3, 4, 5, 6, 7, 8])
        );
        assert_eq!(request.metadata.start_address, ProgramOffset(0x10));
        let mut compiler = Compiler::new().unwrap();
        let (captured, _) = capture(
            &mut compiler,
            &space,
            request,
            &HostTranslateInfo::default(),
            &CompilerSettings::default(),
            &mut MemoryCache::default(),
        )
        .unwrap();
        assert_eq!(
            captured
                .environment
                .observations()
                .instructions
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            vec![ProgramOffset(0x68)]
        );
        crate::engines::tests::program_three_d(&mut channel, 0x1240, 99);
        assert_eq!(
            graphics_request(channel.three_d(), 0)
                .unwrap()
                .bindings
                .geometry_passthrough_mask
                .unwrap()[0],
            99
        );
    }
}
