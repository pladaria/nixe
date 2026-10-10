//! Consumed color state, routed from physical Maxwell targets to fragment slots.
//! Register encodings:
//! https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1980-L2165
use super::super::threed::MaxwellThreeDBlendOp;
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
) -> Result<[ColorOutputState; 8], MaxwellLoweringError> {
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
    // Equation selection (0x12e4) and enable selection (0x135c) are independent.
    // SINGLE_ROP_CONTROL broadcasts SET_BLEND(0); otherwise SET_BLEND(target)
    // still controls each target even when all targets use common equations.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2446-L2454
    for (slot, target) in attachments.color_targets().enumerate() {
        let selected = per_target.then_some(target);
        // Both selector values choose SET_BLEND(0) for target zero. The selector
        // is only consumed when a different physical target participates.
        let enable_target = if target == 0 {
            0
        } else {
            match fixed.single_rop_control().value().ok_or(
                MaxwellLoweringError::IncompleteBlendState {
                    target: None,
                    field: "SET_SINGLE_ROP_CONTROL",
                },
            )? {
                MaxwellThreeDSingleRopControl::Enabled => 0,
                MaxwellThreeDSingleRopControl::Disabled => target,
            }
        };
        let enabled = fixed.blend_enable()[usize::from(enable_target)]
            .value()
            .copied()
            .ok_or(MaxwellLoweringError::IncompleteBlendState {
                target: Some(enable_target),
                field: "SET_BLEND(i)",
            })?;
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
                return Err(MaxwellLoweringError::UnsupportedBlendFormat { target, format });
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
            let component = |index| -> Result<BlendComponent, MaxwellLoweringError> {
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
                .ok_or(MaxwellLoweringError::IncompleteColorWriteState {
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
) -> Result<V, MaxwellLoweringError> {
    value.ok_or(MaxwellLoweringError::IncompleteBlendState { target, field })
}
fn wrong_type() -> MaxwellLoweringError {
    MaxwellLoweringError::ContradictoryState {
        reason: "color state register has the wrong typed value",
    }
}
fn boolean(value: V) -> Result<bool, MaxwellLoweringError> {
    if let V::Boolean(value) = value {
        Ok(value)
    } else {
        Err(wrong_type())
    }
}
fn factor(value: V, target: Option<u8>) -> Result<F, MaxwellLoweringError> {
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
        value => return Err(MaxwellLoweringError::UnsupportedBlendFactor { target, value }),
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::threed::MaxwellThreeDBlendFactor;
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
