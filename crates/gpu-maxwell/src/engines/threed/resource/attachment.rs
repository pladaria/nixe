//! Stateless render-target and 2D image descriptions. Current native views and
//! their DAddr page index belong to the texture cache, as in Eden ImageInfo.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/image_info.cpp
use super::*;
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AttachmentInfo {
    pub address: MaxwellThreeDUnresolvedAddress,
    pub description: ImageDescription,
    pub layout: MaxwellThreeDImageLayout,
    pub array_pitch_dwords: Option<u32>,
    pub selected_layer: u16,
    pub guest_format: MaxwellThreeDGuestImageFormat,
    pub row_pitch_bytes: Option<u32>,
    pub sample_mode: Option<MaxwellThreeDSampleMode>,
}
impl AttachmentInfo {
    pub fn image_info(
        self,
        role: MaxwellThreeDResourceRole,
    ) -> Result<ImageInfo, MaxwellThreeDResourceError> {
        let Self {
            address,
            description,
            layout,
            array_pitch_dwords,
            selected_layer,
            guest_format,
            row_pitch_bytes,
            sample_mode,
        } = self;
        let extent = description.extent();
        let description = targets::sampled(description, layout, guest_format, sample_mode, role)?;
        let samples = description.samples();
        let bpp = u64::from(
            description
                .format()
                .plane_bytes_per_texel(0)
                .ok_or(MaxwellThreeDResourceError::UnsupportedImageLayout { role })?,
        );
        let compact_row = u64::from(extent.width)
            .checked_mul(bpp)
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        let compact_layer = compact_row
            .checked_mul(u64::from(extent.height))
            .and_then(|size| size.checked_mul(u64::from(extent.depth)))
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        // Maxwell render-target and zeta ARRAY_PITCH registers count dwords,
        // not bytes. This is visible in deko3d's 1280x720 target, whose
        // captured 0x000f0000 value denotes a 0x003c0000-byte layer stride.
        // Keep the conversion at the guest/neutral boundary so the neutral
        // image layout remains byte-addressed.
        // https://github.com/yuzu-emu-mirror/yuzu-mainline/blob/310c1f50beb77fc5c6f9075029973161d4e51a4a/src/video_core/texture_cache/image_info.cpp#L126-L177
        let array_pitch = array_pitch_dwords.map(|pitch| u64::from(pitch) * 4);
        let (neutral_layout, layer_stride) = match layout {
            MaxwellThreeDImageLayout::PitchLinear => {
                let row_pitch = row_pitch_bytes.map(u64::from).unwrap_or(compact_row);
                if row_pitch < compact_row {
                    return Err(MaxwellThreeDResourceError::ContradictoryState { role });
                }
                let stride = array_pitch
                    .filter(|value| *value != 0)
                    .unwrap_or(row_pitch * u64::from(extent.height));
                (
                    ImageMemoryLayout::PitchLinear {
                        row_pitch,
                        layer_stride: stride,
                    },
                    stride,
                )
            }
            MaxwellThreeDImageLayout::BlockLinear {
                block_height_log2,
                block_depth_log2,
            } => {
                let row = align_up(compact_row, 64, role)?;
                let rows = align_up(u64::from(extent.height), 8_u64 << block_height_log2, role)?;
                let depths = align_up(u64::from(extent.depth), 1_u64 << block_depth_log2, role)?;
                let minimum = row
                    .checked_mul(rows)
                    .and_then(|size| size.checked_mul(depths))
                    .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
                let stride = array_pitch.filter(|value| *value != 0).unwrap_or(minimum);
                if samples == SampleCount::Four && stride < minimum {
                    return Err(MaxwellThreeDResourceError::ContradictoryState { role });
                }
                (
                    ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                        block_width_log2: 0,
                        block_height_log2,
                        block_depth_log2,
                        layer_stride: stride,
                    }),
                    stride,
                )
            }
        };
        if layer_stride < compact_layer {
            return Err(MaxwellThreeDResourceError::ContradictoryState { role });
        }
        let size = layer_stride
            .checked_mul(u64::from(description.array_layers()))
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        Ok(ImageInfo {
            address: address.get(),
            description,
            swizzle: Swizzle::IDENTITY,
            layout: neutral_layout,
            mip_layouts: Arc::from([(0, neutral_layout)]),
            layer_stride: array_pitch.unwrap_or(0),
            maybe_unaligned_layer_stride: array_pitch.unwrap_or(0),
            size,
            selected_layer,
            guest_format,
        })
    }
}

pub fn render_target_info(
    state: &MaxwellThreeDState,
    role: MaxwellThreeDResourceRole,
) -> Result<Option<AttachmentInfo>, MaxwellThreeDResourceError> {
    let (address, description, selected_layer, guest_format, layout, array_pitch_dwords) =
        match role {
            MaxwellThreeDResourceRole::ColorTarget(index) => {
                let target = state.render_targets().color(index as usize);
                match target.readiness(true) {
                    MaxwellThreeDAttachmentReadiness::Unprogrammed
                    | MaxwellThreeDAttachmentReadiness::Disabled => return Ok(None),
                    MaxwellThreeDAttachmentReadiness::Ready => (),
                    _ => return Err(MaxwellThreeDResourceError::IncompleteState { role }),
                }
                let (description, selected_layer, guest_format) = targets::color(index, &target)?;
                (
                    unresolved(
                        target.address_upper().value(),
                        target.address_lower().value(),
                    ),
                    description,
                    selected_layer,
                    guest_format,
                    target.layout().value().copied(),
                    target.array_pitch().value().copied(),
                )
            }
            MaxwellThreeDResourceRole::DepthStencilTarget => {
                let target = state.render_targets().depth_stencil();
                let (description, selected_layer, guest_format) = targets::depth(&target)?;
                (
                    unresolved(
                        target.address_upper().value(),
                        target.address_lower().value(),
                    ),
                    description,
                    selected_layer,
                    guest_format,
                    target.layout().value().copied(),
                    target.array_pitch().value().copied(),
                )
            }
            _ => return Err(MaxwellThreeDResourceError::IncompleteState { role }),
        };
    let sample_mode = state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::SampleMode)
        .value()
        .and_then(|value| match value {
            MaxwellThreeDFixedFunctionValue::SampleMode(mode) => Some(*mode),
            _ => None,
        });
    Ok(Some(AttachmentInfo {
        address: address.ok_or(MaxwellThreeDResourceError::IncompleteState { role })?,
        description,
        selected_layer,
        guest_format,
        layout: layout.ok_or(MaxwellThreeDResourceError::IncompleteState { role })?,
        array_pitch_dwords,
        row_pitch_bytes: None,
        sample_mode,
    }))
}

pub fn solid_image_info(
    request: &crate::engines::twod::MaxwellTwoDSolidOperation,
    space: &MaxwellGpuMemory,
) -> Result<AttachmentInfo, MaxwellThreeDResourceError> {
    let role = if request.zeta {
        MaxwellThreeDResourceRole::DepthStencilTarget
    } else {
        MaxwellThreeDResourceRole::BlitDestination
    };
    surface_info(
        request.destination,
        space,
        role,
        request.zeta,
        request.zeta,
        false,
    )
}
pub fn blit_image_infos(
    request: &crate::engines::twod::MaxwellTwoDBlitOperation,
    space: &MaxwellGpuMemory,
) -> Result<[AttachmentInfo; 2], MaxwellThreeDResourceError> {
    let resolve = matches!(
        request.kind,
        crate::engines::twod::blit::BlitKind::Downsample2x { .. }
    );
    Ok([
        surface_info(
            request.images[0],
            space,
            MaxwellThreeDResourceRole::BlitSource,
            false,
            true,
            resolve,
        )?,
        surface_info(
            request.images[1],
            space,
            MaxwellThreeDResourceRole::BlitDestination,
            false,
            true,
            false,
        )?,
    ])
}
fn surface_info(
    surface: crate::engines::twod::blit::BlitSurface,
    space: &MaxwellGpuMemory,
    role: MaxwellThreeDResourceRole,
    zeta: bool,
    packed_copy: bool,
    resolve: bool,
) -> Result<AttachmentInfo, MaxwellThreeDResourceError> {
    use crate::engines::twod::blit::BlitSurfaceLayout;
    let address = space.address(surface.address).map_err(|_| {
        MaxwellThreeDResourceError::AddressOutOfRange {
            role,
            address: surface.address,
        }
    })?;
    let pte_kind = space
        .page_kind(address)
        .map_err(|error| MaxwellThreeDResourceError::Resolution { role, error })?;
    let (format, kind, guest_format) = match pte_kind {
        MAXWELL_S8Z24_2CZ_KIND | MAXWELL_Z24S8_2CZ_KIND if packed_copy => (
            ImageFormat::Depth24UnormStencil8Uint,
            ImageKind::DepthStencil,
            MaxwellThreeDGuestImageFormat::DepthStencil(if pte_kind == MAXWELL_S8Z24_2CZ_KIND {
                MaxwellThreeDDepthStencilFormat::Stencil8Z24
            } else {
                MaxwellThreeDDepthStencilFormat::Z24Stencil8
            }),
        ),
        _ if zeta => {
            return Err(MaxwellThreeDResourceError::UnsupportedKind {
                role,
                expected: MAXWELL_Z24S8_2CZ_KIND,
                actual: pte_kind,
            });
        }
        _ => (
            color_image_format(surface.format, role)?,
            ImageKind::Color,
            MaxwellThreeDGuestImageFormat::Color(MaxwellThreeDColorTargetFormat::Color(
                surface.format,
            )),
        ),
    };
    let description = image_description(MaxwellImageDescriptionRequest {
        dimension: ImageDimension::Two,
        extent: ImageExtent {
            width: surface.width,
            height: surface.height,
            depth: 1,
        },
        format,
        kind,
        layers: 1,
        role,
    })?;
    Ok(AttachmentInfo {
        address: MaxwellThreeDUnresolvedAddress::new(
            (surface.address >> 32) as u8,
            surface.address as u32,
        ),
        description,
        layout: match surface.layout {
            BlitSurfaceLayout::Pitch(_) => MaxwellThreeDImageLayout::PitchLinear,
            BlitSurfaceLayout::BlockLinear(block_height_log2) => {
                MaxwellThreeDImageLayout::BlockLinear {
                    block_height_log2,
                    block_depth_log2: 0,
                }
            }
        },
        array_pitch_dwords: None,
        selected_layer: 0,
        guest_format,
        row_pitch_bytes: match surface.layout {
            BlitSurfaceLayout::Pitch(pitch) => Some(pitch),
            _ => None,
        },
        sample_mode: Some(if resolve && pte_kind == MAXWELL_C32_MS4_2CBR_KIND {
            MaxwellThreeDSampleMode::Samples2x2
        } else {
            MaxwellThreeDSampleMode::Samples1x1
        }),
    })
}

pub fn attachment_description(
    info: AttachmentInfo,
    role: MaxwellThreeDResourceRole,
) -> Result<ImageDescription, MaxwellThreeDResourceError> {
    targets::sampled(
        info.description,
        info.layout,
        info.guest_format,
        info.sample_mode,
        role,
    )
}
