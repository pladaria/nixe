//! Consumed color state, routed from physical Maxwell targets to fragment slots.
//! Register encodings:
//! https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1980-L2165
use super::super::MaxwellThreeDBlendOp;
use super::*;
use MaxwellThreeDFixedFunctionRegister as R;
use MaxwellThreeDFixedFunctionValue as V;
use nixe_gpu::{
    BlendComponent, BlendFactor as F, BlendOperation as O, ColorBlendState, ColorOutputState,
    ColorWriteMask,
};

pub(super) fn draw_color_outputs(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    attachments: &DrawAttachmentSelection,
) -> Result<[ColorOutputState; 8], MaxwellThreeDLoweringError> {
    let mut outputs = [ColorOutputState::REPLACE; 8];
    if attachments.colors.is_empty() {
        return Ok(outputs);
    }
    let fixed = state.fixed_function();
    let per_target = boolean(required(
        fixed.register(R::BlendPerTargetEnable).value().copied(),
        None,
        "SET_BLEND_STATE_PER_TARGET",
    )?)?;
    for (slot, target) in attachments.color_targets().enumerate() {
        let selected = per_target.then_some(target);
        let enabled = if per_target {
            fixed.blend_enable()[usize::from(target)]
                .value()
                .copied()
                .ok_or(MaxwellThreeDLoweringError::IncompleteBlendState {
                    target: selected,
                    field: "SET_BLEND(i)",
                })?
        } else {
            *fixed.blend_enable_common().value().ok_or(
                MaxwellThreeDLoweringError::IncompleteBlendState {
                    target: None,
                    field: "SET_BLEND_ENABLE_COMMON",
                },
            )? == MaxwellThreeDBlendEnableCommon::Enabled
        };
        if enabled {
            // Float and 16-bit normalized formats consume additional Maxwell
            // arithmetic controls; do not inherit host defaults for those.
            let format = resolved_image(resources, attachments.colors[slot].1)?
                .description()
                .format();
            if !matches!(
                format,
                nixe_gpu::ImageFormat::Rgba8Unorm
                    | nixe_gpu::ImageFormat::Bgra8Unorm
                    | nixe_gpu::ImageFormat::Rgba8Srgb
                    | nixe_gpu::ImageFormat::Bgra8Srgb
            ) {
                return Err(MaxwellThreeDLoweringError::UnsupportedBlendFormat { target, format });
            }
            let read = |index: usize| {
                let (register, common, per) = [
                    (
                        R::BlendSeparateAlpha,
                        "SET_BLEND_SEPARATE_FOR_ALPHA",
                        "SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA",
                    ),
                    (
                        R::BlendColorOp,
                        "SET_BLEND_OP_COLOR",
                        "SET_BLEND_PER_TARGET_OP_COLOR",
                    ),
                    (
                        R::BlendColorSource,
                        "SET_BLEND_COEFF_SOURCE_COLOR",
                        "SET_BLEND_PER_TARGET_COEFF_SOURCE_COLOR",
                    ),
                    (
                        R::BlendColorDestination,
                        "SET_BLEND_COEFF_DESTINATION_COLOR",
                        "SET_BLEND_PER_TARGET_COEFF_DESTINATION_COLOR",
                    ),
                    (
                        R::BlendAlphaOp,
                        "SET_BLEND_OP_ALPHA",
                        "SET_BLEND_PER_TARGET_OP_ALPHA",
                    ),
                    (
                        R::BlendAlphaSource,
                        "SET_BLEND_COEFF_SOURCE_ALPHA",
                        "SET_BLEND_PER_TARGET_COEFF_SOURCE_ALPHA",
                    ),
                    (
                        R::BlendAlphaDestination,
                        "SET_BLEND_COEFF_DESTINATION_ALPHA",
                        "SET_BLEND_PER_TARGET_COEFF_DESTINATION_ALPHA",
                    ),
                ][index];
                required(
                    if per_target {
                        fixed.per_target_blend()[usize::from(target)][index]
                            .value()
                            .copied()
                    } else {
                        fixed.register(register).value().copied()
                    },
                    selected,
                    if per_target { per } else { common },
                )
            };
            let separate = boolean(read(0)?)?;
            let component = |index| -> Result<BlendComponent, MaxwellThreeDLoweringError> {
                let V::BlendOp(operation) = read(index)? else {
                    return Err(wrong_type());
                };
                let operation = match operation {
                    MaxwellThreeDBlendOp::Add => O::Add,
                    MaxwellThreeDBlendOp::Subtract => O::Subtract,
                    MaxwellThreeDBlendOp::ReverseSubtract => O::ReverseSubtract,
                    MaxwellThreeDBlendOp::Min => O::Min,
                    MaxwellThreeDBlendOp::Max => O::Max,
                };
                // These equations do not consume factor state.
                let (source, destination) = if matches!(operation, O::Min | O::Max) {
                    (F::One, F::One)
                } else {
                    (
                        factor(read(index + 1)?, selected)?,
                        factor(read(index + 2)?, selected)?,
                    )
                };
                Ok(BlendComponent {
                    operation,
                    source,
                    destination,
                })
            };
            let color = component(1)?;
            let alpha = if separate { component(4)? } else { color };
            outputs[slot].blend = Some(ColorBlendState { color, alpha });
        }
        if let Some(V::Boolean(single)) = fixed.register(R::SingleColorTargetWriteControl).value() {
            let mask_register = if *single { 0 } else { target };
            let mask = fixed.color_mask()[usize::from(mask_register)]
                .value()
                .ok_or(MaxwellThreeDLoweringError::IncompleteColorWriteState {
                    target,
                    mask_register,
                })?;
            outputs[slot].write_mask =
                ColorWriteMask::new(mask.red, mask.green, mask.blue, mask.alpha);
        }
    }
    Ok(outputs)
}

fn required(
    value: Option<V>,
    target: Option<u8>,
    field: &'static str,
) -> Result<V, MaxwellThreeDLoweringError> {
    value.ok_or(MaxwellThreeDLoweringError::IncompleteBlendState { target, field })
}
fn wrong_type() -> MaxwellThreeDLoweringError {
    MaxwellThreeDLoweringError::ContradictoryState {
        reason: "color state register has the wrong typed value",
    }
}
fn boolean(value: V) -> Result<bool, MaxwellThreeDLoweringError> {
    if let V::Boolean(value) = value {
        Ok(value)
    } else {
        Err(wrong_type())
    }
}
fn factor(value: V, target: Option<u8>) -> Result<F, MaxwellThreeDLoweringError> {
    let V::BlendFactor(value) = value else {
        return Err(wrong_type());
    };
    Ok(match value.raw() {
        1 | 0x4000 => F::Zero,
        2 | 0x4001 => F::One,
        3 | 0x4300 => F::SourceColor,
        4 | 0x4301 => F::OneMinusSourceColor,
        5 | 0x4302 => F::SourceAlpha,
        6 | 0x4303 => F::OneMinusSourceAlpha,
        7 | 0x4304 => F::DestinationAlpha,
        8 | 0x4305 => F::OneMinusDestinationAlpha,
        9 | 0x4306 => F::DestinationColor,
        10 | 0x4307 => F::OneMinusDestinationColor,
        11 | 0x4308 => F::SourceAlphaSaturated,
        value => return Err(MaxwellThreeDLoweringError::UnsupportedBlendFactor { target, value }),
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::MaxwellThreeDBlendFactor;
    use super::*;

    #[test]
    fn d3d_and_ogl_factors_select_identical_neutral_arithmetic() {
        for (d3d, ogl, expected) in [
            (1, 0x4000, F::Zero),
            (2, 0x4001, F::One),
            (3, 0x4300, F::SourceColor),
            (4, 0x4301, F::OneMinusSourceColor),
            (5, 0x4302, F::SourceAlpha),
            (6, 0x4303, F::OneMinusSourceAlpha),
            (7, 0x4304, F::DestinationAlpha),
            (8, 0x4305, F::OneMinusDestinationAlpha),
            (9, 0x4306, F::DestinationColor),
            (10, 0x4307, F::OneMinusDestinationColor),
            (11, 0x4308, F::SourceAlphaSaturated),
        ] {
            for raw in [d3d, ogl] {
                assert_eq!(
                    factor(
                        V::BlendFactor(MaxwellThreeDBlendFactor::parse(raw).unwrap()),
                        None
                    )
                    .unwrap(),
                    expected
                );
            }
        }
    }
}
