//! Maxwell stage interfaces, attribute transfers, and interpolation.

use super::binary::MaxwellShaderProgramHeader;
use super::decode::{allocate_shader_temporary, validate_register_range};
use super::error::{MaxwellShaderTranslationError, malformed};
use super::tessellation;
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderFloatControl, ShaderInstruction, ShaderInterfaceElement, ShaderInterpolation,
    ShaderIoLocation, ShaderNanMode, ShaderOperation, ShaderPredicate, ShaderRegister,
    ShaderRoundingMode, ShaderScalarType, ShaderSourceLocation, ShaderStage,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn neutral_stage(stage: MaxwellShaderStage) -> ShaderStage {
    match stage {
        MaxwellShaderStage::Vertex | MaxwellShaderStage::VertexCullBeforeFetch => {
            ShaderStage::Vertex
        }
        MaxwellShaderStage::TessellationInit => ShaderStage::TessellationControl,
        MaxwellShaderStage::Tessellation => ShaderStage::TessellationEvaluation,
        MaxwellShaderStage::Geometry => ShaderStage::Geometry,
        MaxwellShaderStage::Pixel => ShaderStage::Fragment,
        MaxwellShaderStage::Compute => ShaderStage::Compute,
    }
}

pub(super) fn decode_header_inputs(
    header: MaxwellShaderProgramHeader,
    vertex_input_types: &BTreeMap<ShaderIoLocation, ShaderScalarType>,
) -> Result<Vec<ShaderInterfaceElement>, MaxwellShaderTranslationError> {
    // Legacy vertex colors have their own map, separate from generic varyings.
    // Do not silently drop them (including their fixed-function color clamp).
    // https://download.nvidia.com/open-gpu-doc/Shader-Program-Header/1/Shader-Program-Header.html#ImapColor
    let color_map_bit = if header.stage == MaxwellShaderStage::Pixel {
        448
    } else {
        320
    };
    if header.bits(color_map_bit, 16) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedHeaderFeature {
            stage: header.stage,
            feature: "legacy vertex color input attributes",
        });
    }
    let mut inputs = Vec::new();
    if header.stage == MaxwellShaderStage::Pixel {
        for component in 0..4_u8 {
            if header.bit(188 + component as usize) {
                inputs.push(interface_element(
                    ShaderIoLocation::Position,
                    component,
                    None,
                ));
            }
        }
        for generic in 0..32_u8 {
            for component in 0..4_u8 {
                let raw = header.bits(192 + generic as usize * 8 + component as usize * 2, 2) as u8;
                let interpolation = match raw {
                    0 => continue,
                    1 => ShaderInterpolation::Constant,
                    2 => ShaderInterpolation::Perspective,
                    3 => ShaderInterpolation::ScreenLinear,
                    _ => unreachable!("two-bit interpolation"),
                };
                inputs.push(interface_element(
                    ShaderIoLocation::Generic(generic),
                    component,
                    Some(interpolation),
                ));
            }
        }
    } else {
        tessellation::header_inputs(header, &mut inputs);
        for component in 0..4_u8 {
            if header.bit(188 + component as usize) {
                inputs.push(interface_element(
                    ShaderIoLocation::Position,
                    component,
                    None,
                ));
            }
        }
        for generic in 0..32_u8 {
            for component in 0..4_u8 {
                if header.bit(192 + generic as usize * 4 + component as usize) {
                    let location = ShaderIoLocation::Generic(generic);
                    inputs.push(
                        ShaderInterfaceElement::new(
                            location,
                            component,
                            vertex_input_types
                                .get(&location)
                                .copied()
                                .unwrap_or(ShaderScalarType::Float32),
                            None,
                        )
                        .expect("decoded SPH component is bounded"),
                    );
                }
            }
        }
    }
    Ok(inputs)
}

pub(super) fn decode_header_outputs(
    header: MaxwellShaderProgramHeader,
) -> Result<Vec<ShaderInterfaceElement>, MaxwellShaderTranslationError> {
    if header.stage != MaxwellShaderStage::Pixel && header.bits(560, 16) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedHeaderFeature {
            stage: header.stage,
            feature: "legacy vertex color output attributes",
        });
    }
    let mut outputs = Vec::new();
    if header.stage == MaxwellShaderStage::Pixel {
        for target in 0..8_u8 {
            for component in 0..4_u8 {
                if header.bit(576 + target as usize * 4 + component as usize) {
                    outputs.push(interface_element(
                        ShaderIoLocation::Color(target),
                        component,
                        None,
                    ));
                }
            }
        }
        if header.bit(608) {
            outputs.push(interface_element(ShaderIoLocation::SampleMask, 0, None));
        }
        if header.bit(609) {
            outputs.push(interface_element(ShaderIoLocation::FragmentDepth, 0, None));
        }
    } else {
        tessellation::header_outputs(header, &mut outputs);
        if header.bit(427) {
            outputs.push(interface_element(ShaderIoLocation::PointSize, 0, None));
        }
        for component in 0..4_u8 {
            if header.bit(428 + component as usize) {
                outputs.push(interface_element(
                    ShaderIoLocation::Position,
                    component,
                    None,
                ));
            }
        }
        for generic in 0..32_u8 {
            for component in 0..4_u8 {
                if header.bit(432 + generic as usize * 4 + component as usize) {
                    outputs.push(interface_element(
                        ShaderIoLocation::Generic(generic),
                        component,
                        None,
                    ));
                }
            }
        }
    }
    Ok(outputs)
}

pub(super) fn interface_element(
    location: ShaderIoLocation,
    component: u8,
    interpolation: Option<ShaderInterpolation>,
) -> ShaderInterfaceElement {
    ShaderInterfaceElement::new(
        location,
        component,
        ShaderScalarType::Float32,
        interpolation,
    )
    .expect("decoded SPH component is bounded")
}

pub(super) fn preload_vertex_inputs(
    stage: ShaderStage,
    inputs: &[ShaderInterfaceElement],
) -> Vec<ShaderInstruction> {
    if stage != ShaderStage::Vertex {
        return Vec::new();
    }

    // NVIDIA's public SPH specification makes the enabled generic input
    // components explicit in ImapGenericVector:
    // https://download.nvidia.com/open-gpu-doc/Shader-Program-Header/1/Shader-Program-Header.html#ImapVector
    // The VTG launch contract used by the captured program exposes generic
    // input vector zero in r0-r3; later attributes are fetched explicitly with
    // ALD. Keep this deliberately narrow ABI bridge here, and assert its
    // decoded IR shape in the captured-program test, rather than teaching the
    // platform-independent IR about Maxwell launch registers.
    inputs
        .iter()
        .filter(|input| input.location() == ShaderIoLocation::Generic(0))
        .map(|input| {
            ShaderInstruction::new(
                ShaderSourceLocation::new(0),
                ShaderPredicate::Always,
                ShaderOperation::LoadInput {
                    destinations: vec![ShaderRegister::new(u16::from(input.component()))]
                        .into_boxed_slice(),
                    location: input.location(),
                    first_component: input.component(),
                    scalar_type: input.scalar_type(),
                },
            )
        })
        .collect()
}

pub(super) fn append_implicit_outputs(
    stage: ShaderStage,
    source: ShaderSourceLocation,
    outputs: &[ShaderInterfaceElement],
    explicitly_stored: &BTreeSet<(ShaderIoLocation, u8)>,
    instructions: &mut Vec<ShaderInstruction>,
) -> Result<(), MaxwellShaderTranslationError> {
    let mut next_undefined_register = 255_u16;
    // TCS outputs are shared; synthetic undefined stores could erase writes
    // from other invocations, including their per-patch tessellation levels.
    if stage == ShaderStage::TessellationControl {
        return Ok(());
    }
    for output in outputs {
        if explicitly_stored.contains(&(output.location(), output.component())) {
            continue;
        }
        if stage != ShaderStage::Fragment {
            // The SPH output map allocates interface locations; actual VTG
            // writes are explicit AST operations. Mesa records the map and
            // store requests independently:
            // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sph.rs#L476-494
            // A declared component without a reachable AST is undefined, not
            // an implicit mapping to r0-r3.
            let register = ShaderRegister::new(next_undefined_register);
            next_undefined_register = next_undefined_register.saturating_sub(1);
            instructions.push(ShaderInstruction::new(
                source,
                ShaderPredicate::Always,
                ShaderOperation::Undefined32 {
                    destination: register,
                },
            ));
            instructions.push(ShaderInstruction::new(
                source,
                ShaderPredicate::Always,
                ShaderOperation::StoreOutput {
                    sources: vec![register].into_boxed_slice(),
                    location: output.location(),
                    first_component: output.component(),
                    scalar_type: output.scalar_type(),
                },
            ));
            continue;
        }
        // Maxwell fragment outputs are assigned consecutively to GPRs before
        // EXIT. Mesa's pinned register allocator materializes `OpRegOut`
        // sources at r0, r1, ...:
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/assign_regs.rs#L1235-1255
        let register = match output.location() {
            ShaderIoLocation::Position | ShaderIoLocation::Generic(_) => unreachable!(),
            ShaderIoLocation::Color(target) => {
                u16::from(target) * 4 + u16::from(output.component())
            }
            ShaderIoLocation::FragmentDepth | ShaderIoLocation::SampleMask => {
                return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    stage: MaxwellShaderStage::Pixel,
                    instruction_offset: source.byte_offset(),
                    encoding: 0,
                    detail: "implicit depth or sample-mask output register mapping",
                });
            }
            _ => {
                return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    stage: MaxwellShaderStage::Pixel,
                    instruction_offset: source.byte_offset(),
                    encoding: 0,
                    detail: "implicit system-value output register mapping",
                });
            }
        };
        instructions.push(ShaderInstruction::new(
            source,
            ShaderPredicate::Always,
            ShaderOperation::StoreOutput {
                sources: vec![ShaderRegister::new(register)].into_boxed_slice(),
                location: output.location(),
                first_component: output.component(),
                scalar_type: output.scalar_type(),
            },
        ));
    }
    Ok(())
}

pub(super) const fn is_attribute_load(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfffe == 0xefd8
}

pub(super) const fn is_attribute_store(encoding: u64) -> bool {
    ((encoding >> 48) as u16) & 0xfffe == 0xeff0
}

pub(super) const fn is_interpolate(encoding: u64) -> bool {
    encoding >> 56 == 0xe0
}

pub(super) fn decode_attribute_load(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    vertex_input_types: &BTreeMap<ShaderIoLocation, ShaderScalarType>,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    if matches!(
        stage,
        MaxwellShaderStage::TessellationInit | MaxwellShaderStage::Tessellation
    ) {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "tessellation ALD requires ISBE handle/output/patch address lowering",
        });
    }
    let destination = (encoding & 0xff) as u8;
    let components = (((encoding >> 47) & 0x3) + 1) as u8;
    validate_register_range(
        stage,
        offset,
        encoding,
        destination,
        components,
        register_count,
    )?;
    if ((encoding >> 8) & 0xff) != 0xff || ((encoding >> 39) & 0xff) != 0xff {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "indexed ALD is not encoded with RZ operands",
        ));
    }
    // ALD addresses a contiguous sequence of 32-bit attribute slots. A
    // vector load may therefore cross an attribute-vector boundary; the
    // captured deko3d vertex shader uses the two adjacent ABI slots at 0x2f8
    // and 0x2fc to load InstanceId and VertexId together. Mesa NAK preserves
    // this component count in bits 47..49:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L2855-L2876
    // Split the contiguous hardware transfer at neutral-IR location and type
    // boundaries instead of pretending that it belongs to one vec4 input.
    let first_address = ((encoding >> 20) & 0x3ff) as u16;
    let mut operations = Vec::new();
    for component in 0..components {
        let address = first_address
            .checked_add(u16::from(component) * 4)
            .ok_or_else(|| malformed(stage, offset, encoding, "ALD attribute address overflows"))?;
        let (location, first_component, default_scalar_type, _) =
            input_attribute_location(stage, offset, encoding, address)?;
        let scalar_type = vertex_input_types
            .get(&location)
            .copied()
            .unwrap_or(default_scalar_type);
        let destination = ShaderRegister::new(u16::from(destination + component));
        if let Some(ShaderOperation::LoadInput {
            destinations,
            location: previous_location,
            first_component: previous_first_component,
            scalar_type: previous_scalar_type,
        }) = operations.last_mut()
            && *previous_location == location
            && *previous_scalar_type == scalar_type
            && *previous_first_component + destinations.len() as u8 == first_component
        {
            let mut grouped = destinations.to_vec();
            grouped.push(destination);
            *destinations = grouped.into_boxed_slice();
        } else {
            operations.push(ShaderOperation::LoadInput {
                destinations: vec![destination].into_boxed_slice(),
                location,
                first_component,
                scalar_type,
            });
        }
    }
    Ok(operations)
}

fn input_attribute_location(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    address: u16,
) -> Result<(ShaderIoLocation, u8, ShaderScalarType, u8), MaxwellShaderTranslationError> {
    // Mesa's pinned Maxwell ABI identifies these adjacent scalar system-value
    // attributes explicitly:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak_private.h#L81-L82
    let system_value = match address {
        0x2f8 => Some(ShaderIoLocation::InstanceId),
        0x2fc => Some(ShaderIoLocation::VertexId),
        _ => None,
    };
    if let Some(location) = system_value {
        if !matches!(
            stage,
            MaxwellShaderStage::Vertex | MaxwellShaderStage::VertexCullBeforeFetch
        ) {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "vertex system-value ALD is used outside a vertex shader",
            ));
        }
        return Ok((location, 0, ShaderScalarType::Unsigned32, 1));
    }

    let (location, first_component) = attribute_location(stage, offset, encoding, address)?;
    Ok((location, first_component, ShaderScalarType::Float32, 4))
}

pub(super) fn decode_attribute_store(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    if encoding & (1 << 31) != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "per-patch AST requires a tessellation control shader",
        ));
    }
    let source = (encoding & 0xff) as u8;
    let components = (((encoding >> 47) & 0x3) + 1) as u8;
    validate_register_range(stage, offset, encoding, source, components, register_count)?;
    if ((encoding >> 8) & 0xff) != 0xff || ((encoding >> 39) & 0xff) != 0xff {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "indexed AST is not encoded with RZ operands",
        ));
    }
    let address = ((encoding >> 20) & 0x3ff) as u16;
    let (location, first_component) = attribute_location(stage, offset, encoding, address)?;
    let available_components = if location == ShaderIoLocation::PointSize {
        1
    } else {
        4
    };
    if first_component + components > available_components {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "AST crosses an attribute vector boundary",
        ));
    }
    Ok(ShaderOperation::StoreOutput {
        sources: (0..components)
            .map(|component| ShaderRegister::new(u16::from(source + component)))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        location,
        first_component,
        scalar_type: ShaderScalarType::Float32,
    })
}

pub(super) fn decode_interpolate(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    inputs: &mut Vec<ShaderInterfaceElement>,
    next_temporary: &mut u16,
    normalized_multiplier: bool,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    if stage != MaxwellShaderStage::Pixel {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "IPA is only allocated for pixel shaders",
        ));
    }
    let destination = (encoding & 0xff) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    if encoding & (1_u64 << 38) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "indexed IPA attribute addressing",
        });
    }
    if encoding & (1_u64 << 51) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "saturated IPA result",
        });
    }
    let sample_mode = ((encoding >> 52) & 0x3) as u8;
    if sample_mode != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "IPA centroid/offset sample mode",
        });
    }
    let address = ((encoding >> 28) & 0x3ff) as u16;
    let (location, component) = attribute_location(stage, offset, encoding, address)?;
    let interpolation_mode = ((encoding >> 54) & 0x3) as u8;
    let interpolation = inputs
        .iter()
        .find(|input| input.location() == location && input.component() == component)
        .and_then(|input| input.interpolation());
    match interpolation_mode {
        0 if interpolation == Some(ShaderInterpolation::Perspective) => {
            interpolate_perspective_numerator(
                stage,
                offset,
                encoding,
                destination,
                location,
                component,
                None,
                inputs,
                next_temporary,
            )
        }
        0 => Ok(vec![ShaderOperation::LoadInput {
            destinations: vec![ShaderRegister::new(u16::from(destination))].into_boxed_slice(),
            location,
            first_component: component,
            scalar_type: ShaderScalarType::Float32,
        }]),
        1 => {
            let interpolation = interpolation.ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "IPA.PASS_MUL_W references a non-interpolated input",
                )
            })?;
            if interpolation != ShaderInterpolation::Perspective {
                return Err(malformed(
                    stage,
                    offset,
                    encoding,
                    "IPA.PASS_MUL_W requires a perspective input",
                ));
            }
            let reciprocal = ((encoding >> 20) & 0xff) as u8;
            validate_register_range(stage, offset, encoding, reciprocal, 1, register_count)?;
            // When the producer is the canonical reciprocal of FragCoord.w,
            // the host interpolator already performs precisely this operation.
            // The caller proves that producer from register definitions; never
            // infer it just from the operand register number.
            if normalized_multiplier {
                return Ok(vec![ShaderOperation::InterpolateInput {
                    destination: ShaderRegister::new(u16::from(destination)),
                    location,
                    component,
                    interpolation,
                }]);
            }
            interpolate_perspective_numerator(
                stage,
                offset,
                encoding,
                destination,
                location,
                component,
                Some(ShaderRegister::new(u16::from(reciprocal))),
                inputs,
                next_temporary,
            )
        }
        2 => {
            if interpolation != Some(ShaderInterpolation::Constant) {
                return Err(malformed(
                    stage,
                    offset,
                    encoding,
                    "IPA.CONSTANT requires a flat input declared by the shader header",
                ));
            }
            Ok(vec![ShaderOperation::InterpolateInput {
                destination: ShaderRegister::new(u16::from(destination)),
                location,
                component,
                interpolation: ShaderInterpolation::Constant,
            }])
        }
        3 => {
            let interpolation = interpolation.ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "IPA.SC references a non-interpolated input",
                )
            })?;
            Ok(vec![ShaderOperation::InterpolateInput {
                destination: ShaderRegister::new(u16::from(destination)),
                location,
                component,
                interpolation,
            }])
        }
        _ => unreachable!("two-bit IPA interpolation mode"),
    }
}

pub(super) fn is_reciprocal_fragment_w(
    instructions: &[ShaderInstruction],
    register: ShaderRegister,
) -> bool {
    fn definition(instructions: &[ShaderInstruction], register: ShaderRegister) -> Option<usize> {
        instructions.iter().rposition(|instruction| {
            let mut writes = false;
            instruction
                .operation()
                .visit_destination_registers(|destination| writes |= destination == register);
            writes
        })
    }
    let Some(index) = definition(instructions, register) else {
        return false;
    };
    let instruction = &instructions[index];
    if instruction.predicate() != ShaderPredicate::Always {
        return false;
    }
    let ShaderOperation::Reciprocal32 {
        source,
        float_control: ShaderFloatControl::PRECISE,
        ..
    } = instruction.operation()
    else {
        return false;
    };
    let preceding = &instructions[..index];
    let Some(index) = definition(preceding, *source) else {
        return false;
    };
    let instruction = &preceding[index];
    if instruction.predicate() != ShaderPredicate::Always {
        return false;
    }
    matches!(instruction.operation(), ShaderOperation::LoadInput { destinations, location: ShaderIoLocation::Position, first_component, scalar_type: ShaderScalarType::Float32 } if destinations.iter().enumerate().any(|(index, destination)| destination == source && usize::from(*first_component) + index == 3))
}

#[allow(clippy::too_many_arguments)]
fn interpolate_perspective_numerator(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    destination: u8,
    location: ShaderIoLocation,
    component: u8,
    multiplier: Option<ShaderRegister>,
    inputs: &mut Vec<ShaderInterfaceElement>,
    next_temporary: &mut u16,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    // IPA.PASS exposes sum(lambda_i * attribute_i / clip_w_i), whereas
    // host perspective inputs expose that numerator divided by FragCoord.w.
    // Undo host normalization before PASS's explicit shader multiply or
    // PASS_MUL_W's register operand. The operand need not be exactly 1/w.
    // https://github.com/kotx/Ryujinx/blob/master/src/Ryujinx.Graphics.Shader/Instructions/InstEmitAttribute.cs#L146-L213
    let mut temporary = || {
        allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "IPA temporary-register space exhausted",
            next_temporary,
        )
    };
    let attribute = temporary()?;
    let reciprocal_w = temporary()?;
    let destination = ShaderRegister::new(u16::from(destination));
    let numerator = if multiplier.is_some() {
        temporary()?
    } else {
        destination
    };
    if !inputs
        .iter()
        .any(|input| input.location() == ShaderIoLocation::Position && input.component() == 3)
    {
        inputs.push(interface_element(ShaderIoLocation::Position, 3, None));
    }
    let float_control = ShaderFloatControl::new(
        ShaderRoundingMode::NearestEven,
        ShaderNanMode::Propagate,
        true,
        true,
        false,
    );
    let mut operations = vec![
        ShaderOperation::LoadInput {
            destinations: vec![attribute].into_boxed_slice(),
            location,
            first_component: component,
            scalar_type: ShaderScalarType::Float32,
        },
        ShaderOperation::LoadInput {
            destinations: vec![reciprocal_w].into_boxed_slice(),
            location: ShaderIoLocation::Position,
            first_component: 3,
            scalar_type: ShaderScalarType::Float32,
        },
        ShaderOperation::Multiply32 {
            destination: numerator,
            left: attribute,
            right: reciprocal_w,
            scalar_type: ShaderScalarType::Float32,
            float_control,
        },
    ];
    if let Some(multiplier) = multiplier {
        operations.push(ShaderOperation::Multiply32 {
            destination,
            left: numerator,
            right: multiplier,
            scalar_type: ShaderScalarType::Float32,
            float_control,
        });
    }
    Ok(operations)
}

pub(super) fn attribute_location(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    address: u16,
) -> Result<(ShaderIoLocation, u8), MaxwellShaderTranslationError> {
    // Scalar point-size slot in the public Maxwell shader ABI.
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak_private.h#L48
    if address == 0x6c {
        return Ok((ShaderIoLocation::PointSize, 0));
    }
    if (0x70..=0x7c).contains(&address) && address.is_multiple_of(4) {
        return Ok((ShaderIoLocation::Position, ((address - 0x70) / 4) as u8));
    }
    if (0x80..0x280).contains(&address) && address.is_multiple_of(4) {
        let relative = address - 0x80;
        return Ok((
            ShaderIoLocation::Generic((relative / 0x10) as u8),
            ((relative % 0x10) / 4) as u8,
        ));
    }
    Err(malformed(
        stage,
        offset,
        encoding,
        "attribute address is unsupported or misaligned",
    ))
}

#[cfg(test)]
mod tests;
