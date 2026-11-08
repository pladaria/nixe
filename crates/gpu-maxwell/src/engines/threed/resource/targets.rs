//! Render-target register descriptions precede VM/resource consumption.
//! Fixed pipeline state and the texture cache share the same format/MSAA decoding.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/fixed_pipeline_state.cpp
use super::*;
pub(super) fn color(
    index: u8,
    target: &MaxwellThreeDColorTargetState,
) -> Result<(ImageDescription, MaxwellThreeDGuestImageFormat), MaxwellThreeDResourceError> {
    let role = MaxwellThreeDResourceRole::ColorTarget(index);
    let width = *target
        .width()
        .value()
        .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
    let height = *target
        .height()
        .value()
        .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
    let third = *target
        .third_dimension()
        .value()
        .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
    let kind = *target
        .kind()
        .value()
        .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
    let guest_format = target
        .format()
        .value()
        .copied()
        .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
    let format = color_image_format(guest_format.raw(), role)?;
    // GM20B color surface formats explicitly distinguish UNORM and sRGB.
    // deko3d programs RGBA8Srgb/BGRA8Srgb directly without SET_SRGB_WRITE;
    // the latter's zero reset must not strip the attachment's transfer function.
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/format_traits.inc#L46-L47
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/gpu_3d_base.cpp#L50-L55
    // Linear color WIDTH is a byte pitch, while block-linear WIDTH counts
    // texels. Eden ImageInfo also gives linear surfaces one layer.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/image_info.cpp
    let linear = target.layout().value() == Some(&MaxwellThreeDImageLayout::PitchLinear);
    let width = if linear {
        width / u32::from(format.plane_bytes_per_texel(0).unwrap())
    } else {
        width
    };
    let (dimension, depth, layers) = if linear {
        (ImageDimension::Two, 1, 1)
    } else {
        match kind {
            MaxwellThreeDImageKind::Array => {
                let layers = u16::try_from(third)
                    .map_err(|_| MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
                (ImageDimension::Two, 1, layers)
            }
            MaxwellThreeDImageKind::ThreeDimensional => (ImageDimension::Three, third, 1),
        }
    };
    let description = image_description(MaxwellImageDescriptionRequest {
        dimension,
        extent: ImageExtent {
            width,
            height,
            depth,
        },
        format,
        kind: ImageKind::Color,
        layers,
        role,
    })?;
    Ok((
        description,
        MaxwellThreeDGuestImageFormat::Color(guest_format),
    ))
}
pub(super) fn depth(
    target: &MaxwellThreeDDepthStencilTargetState,
) -> Result<(ImageDescription, MaxwellThreeDGuestImageFormat), MaxwellThreeDResourceError> {
    let role = MaxwellThreeDResourceRole::DepthStencilTarget;
    if target.sparse().value().is_some_and(|value| value & 1 != 0) {
        return Err(MaxwellThreeDResourceError::UnsupportedSparseDepth {
            value: *target.sparse().value().unwrap(),
        });
    }
    let guest_format = *target
        .format()
        .value()
        .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
    let format = match guest_format {
        MaxwellThreeDDepthStencilFormat::Z16 => ImageFormat::Depth16Unorm,
        MaxwellThreeDDepthStencilFormat::Z24Stencil8
        | MaxwellThreeDDepthStencilFormat::Stencil8Z24 => ImageFormat::Depth24UnormStencil8Uint,
        MaxwellThreeDDepthStencilFormat::ZFloat32 => ImageFormat::Depth32Float,
        MaxwellThreeDDepthStencilFormat::ZFloat32X24Stencil8 => {
            ImageFormat::Depth32FloatStencil8Uint
        }
        format => {
            return Err(MaxwellThreeDResourceError::UnsupportedDepthFormat {
                role,
                format: format as u32,
            });
        }
    };
    let width = *target
        .width()
        .value()
        .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
    let height = *target
        .height()
        .value()
        .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
    let control = *target
        .array_control()
        .value()
        .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
    let layers = if target.layout().value() == Some(&MaxwellThreeDImageLayout::PitchLinear) {
        1
    } else {
        match control {
            MaxwellThreeDDepthArrayControl::ThirdDimensionDefinesArraySize => *target
                .third_dimension()
                .value()
                .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?,
            MaxwellThreeDDepthArrayControl::ArraySizeIsOne => 1,
        }
    };
    let description = image_description(MaxwellImageDescriptionRequest {
        dimension: ImageDimension::Two,
        extent: ImageExtent {
            width,
            height,
            depth: 1,
        },
        format,
        kind: ImageKind::DepthStencil,
        layers,
        role,
    })?;
    Ok((
        description,
        MaxwellThreeDGuestImageFormat::DepthStencil(guest_format),
    ))
}
pub(super) fn sampled(
    description: ImageDescription,
    layout: MaxwellThreeDImageLayout,
    guest_format: MaxwellThreeDGuestImageFormat,
    sample_mode: Option<MaxwellThreeDSampleMode>,
    role: MaxwellThreeDResourceRole,
) -> Result<ImageDescription, MaxwellThreeDResourceError> {
    let samples = match sample_mode.ok_or(MaxwellThreeDResourceError::IncompleteState { role })? {
        MaxwellThreeDSampleMode::Samples1x1 => SampleCount::One,
        MaxwellThreeDSampleMode::Samples2x2 => SampleCount::Four,
        mode => return Err(MaxwellThreeDResourceError::UnsupportedSampleMode { role, mode }),
    };
    // Target WIDTH/HEIGHT count stored samples, not raster pixels. Keep
    // their full footprint for mappings/pitches while the neutral image
    // carries pixel dimensions and a separate sample count.
    // https://github.com/devkitPro/deko3d/blob/master/source/dk_image.cpp
    let extent = description.extent();
    let description = if samples == SampleCount::Four {
        if !extent.width.is_multiple_of(2)
        || !extent.height.is_multiple_of(2)
        // Only the 32-bit color MS4 storage family is decoded below.
        // Wider single-sample formats must not inherit its PTE kind.
        || (matches!(guest_format, MaxwellThreeDGuestImageFormat::Color(_))
            && description.format().plane_bytes_per_texel(0) != Some(4))
        || description.dimension() != ImageDimension::Two
        || !matches!(
            layout,
            MaxwellThreeDImageLayout::BlockLinear {
            block_depth_log2: 0,
            ..
            }
        )
        || !matches!(
            guest_format,
            MaxwellThreeDGuestImageFormat::Color(_)
            | MaxwellThreeDGuestImageFormat::DepthStencil(
                MaxwellThreeDDepthStencilFormat::Stencil8Z24
                | MaxwellThreeDDepthStencilFormat::Z24Stencil8
            )
        ) {
            return Err(MaxwellThreeDResourceError::UnsupportedImageLayout { role });
        }
        ImageDescription::new(
            description.dimension(),
            ImageExtent {
                width: extent.width / 2,
                height: extent.height / 2,
                depth: 1,
            },
            description.format(),
            description.kind(),
            1,
            description.array_layers(),
            samples,
        )
        .map_err(|_| MaxwellThreeDResourceError::ContradictoryState { role })?
    } else {
        description
    };
    Ok(description)
}
