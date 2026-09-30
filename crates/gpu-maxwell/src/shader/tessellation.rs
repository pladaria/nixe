//! SPH and direct patch operations. ISBE addressing is a separate SASS boundary:
//! an ALD vertex operand is not automatically a neutral control-point index.
//!
//! Public producer ABI and instruction fields:
//! https://github.com/devkitPro/uam/blob/master/source/compiler_iface.cpp#L476-L487
//! https://github.com/devkitPro/uam/blob/master/source/nv_attributes.h
//! https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L2487-L2506
use super::*;

#[cfg(test)]
#[path = "patch_order_tests.rs"]
mod patch_order_tests;

pub(super) fn header_inputs(
    header: MaxwellShaderProgramHeader,
    inputs: &mut Vec<ShaderInterfaceElement>,
) {
    if !matches!(
        header.stage,
        MaxwellThreeDShaderStage::TessellationInit | MaxwellThreeDShaderStage::Tessellation
    ) {
        return;
    }
    if header.bit(184) {
        declare_input(inputs, ShaderIoLocation::PrimitiveId);
    }
    if header.bit(187) {
        inputs.push(interface_element(ShaderIoLocation::PointSize, 0, None));
    }
    if header.stage == MaxwellThreeDShaderStage::Tessellation {
        for index in 0..6 {
            if header.bit(164 + index) {
                let (location, component) = patch_location((index * 4) as u16).unwrap();
                inputs.push(interface_element(location, component, None));
            }
        }
    }
}

pub(super) fn header_outputs(
    header: MaxwellShaderProgramHeader,
    outputs: &mut Vec<ShaderInterfaceElement>,
) {
    if header.stage == MaxwellThreeDShaderStage::TessellationInit {
        // SPH allocates scalar slots, not vec4s. Slots 6/7 are padding before
        // user patch data, while 0..5 hold the six tessellator levels.
        for slot in 0..header.bits(56, 8) {
            if let Some((location, component)) = patch_location((slot * 4) as u16) {
                outputs.push(interface_element(location, component, None));
            }
        }
    }
    if matches!(
        header.stage,
        MaxwellThreeDShaderStage::TessellationInit | MaxwellThreeDShaderStage::Tessellation
    ) && header.bit(427)
    {
        outputs.push(interface_element(ShaderIoLocation::PointSize, 0, None));
    }
}

pub(super) fn patch_location(address: u16) -> Option<(ShaderIoLocation, u8)> {
    if !address.is_multiple_of(4) {
        return None;
    }
    match address {
        0..=12 => Some((ShaderIoLocation::TessLevelOuter, (address / 4) as u8)),
        16..=20 => Some((ShaderIoLocation::TessLevelInner, ((address - 16) / 4) as u8)),
        32..=1020 => Some((
            ShaderIoLocation::Patch(((address - 32) / 16) as u8),
            ((address - 32) % 16 / 4) as u8,
        )),
        _ => None,
    }
}

/// Preserve Maxwell's warp-synchronous patch-output accesses on hosts that do
/// not execute the whole patch in lockstep. UAM removes TCS BAR.SYNC entirely:
/// https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_lowering_nvc0.cpp#L831-L835
/// RAW, WAR and WAW conflicts need a rendezvous; independent slots and repeated
/// reads do not. One barrier orders all earlier accesses. This runs only during
/// shader translation, with constant-time scalar-slot membership tests.
pub(super) fn order_patch_outputs(instructions: &mut Vec<ShaderInstruction>) {
    let mut reads = [0_u64; 4];
    let mut writes = [0_u64; 4];
    let mut barriers = Vec::new();
    for (index, instruction) in instructions.iter().enumerate() {
        let (location, component, count, write) = match instruction.operation() {
            ShaderOperation::LoadPatchOutput {
                location,
                component,
                ..
            } => (*location, *component, 1, false),
            ShaderOperation::StoreOutput {
                location,
                first_component,
                sources,
                ..
            } => (*location, *first_component, sources.len(), true),
            _ => continue,
        };
        let base = match location {
            ShaderIoLocation::TessLevelOuter => 0,
            ShaderIoLocation::TessLevelInner => 4,
            ShaderIoLocation::Patch(n) => 8 + usize::from(n) * 4,
            _ => continue,
        } + usize::from(component);
        let conflict = (base..base + count).any(|slot| {
            let mask = 1 << (slot % 64);
            writes[slot / 64] & mask != 0 || (write && reads[slot / 64] & mask != 0)
        });
        if conflict {
            barriers.push(index);
            reads.fill(0);
            writes.fill(0);
        }
        let accessed = if write { &mut writes } else { &mut reads };
        for slot in base..base + count {
            accessed[slot / 64] |= 1 << (slot % 64);
        }
    }
    if barriers.is_empty() {
        return;
    }
    let original = std::mem::take(instructions);
    instructions.reserve(original.len() + barriers.len());
    let mut barriers = barriers.into_iter().peekable();
    for (index, instruction) in original.into_iter().enumerate() {
        if barriers.peek() == Some(&index) {
            barriers.next();
            // Predicated stores still need every invocation to rendezvous.
            // The neutral verifier rejects branches/early exits for which
            // reaching this point uniformly has not been proven.
            instructions.push(ShaderInstruction::new(
                instruction.source(),
                ShaderPredicate::Always,
                ShaderOperation::PatchBarrier,
            ));
        }
        instructions.push(instruction);
    }
}

fn declare_input(inputs: &mut Vec<ShaderInterfaceElement>, location: ShaderIoLocation) {
    if !inputs.iter().any(|input| input.location() == location) {
        inputs.push(
            ShaderInterfaceElement::new(location, 0, ShaderScalarType::Unsigned32, None).unwrap(),
        );
    }
}

pub(super) const fn is_system_register_read(encoding: u64) -> bool {
    encoding >> 48 == 0xf0c8
}

pub(super) fn decode_system_register(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    inputs: &mut Vec<ShaderInterfaceElement>,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    if encoding & !0xffff_0000_0fff_00ff != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "S2R reserved field is nonzero",
        ));
    }
    // S2R selector: public GM107 emitSYS/emitS2R, not a shader fingerprint.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L267-L274
    let location = match ((encoding >> 20) & 0xff, stage) {
        (0x11, MaxwellThreeDShaderStage::TessellationInit) => ShaderIoLocation::InvocationId,
        (
            0x10,
            MaxwellThreeDShaderStage::TessellationInit | MaxwellThreeDShaderStage::Tessellation,
        ) => ShaderIoLocation::PatchVertices,
        _ => {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail: "S2R system value requires unsupported stage or invocation/warp ABI semantics",
            });
        }
    };
    let destination = encoding as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    declare_input(inputs, location);
    Ok(ShaderOperation::LoadInput {
        destinations: vec![ShaderRegister::new(u16::from(destination))].into_boxed_slice(),
        location,
        first_component: 0,
        scalar_type: ShaderScalarType::Unsigned32,
    })
}

pub(super) fn decode_control_store(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
    inputs: &mut Vec<ShaderInterfaceElement>,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    let source = encoding as u8;
    let count = (((encoding >> 47) & 3) + 1) as u8;
    validate_register_range(stage, offset, encoding, source, count, register_count)?;
    if ((encoding >> 8) & 0xff) != 0xff || ((encoding >> 39) & 0xff) != 0xff {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "indexed AST requires attribute/ISBE address lowering",
        });
    }
    let first_address = ((encoding >> 20) & 0x3ff) as u16;
    let patch = encoding & (1 << 31) != 0;
    let mut operations = Vec::new();
    let vertex = if patch {
        None
    } else {
        let vertex = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "control-point index temporary exceeds register bank",
            next_temporary,
        )?;
        declare_input(inputs, ShaderIoLocation::InvocationId);
        operations.push(ShaderOperation::LoadInput {
            destinations: vec![vertex].into_boxed_slice(),
            location: ShaderIoLocation::InvocationId,
            first_component: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        Some(vertex)
    };
    for lane in 0..count {
        let address = first_address + u16::from(lane) * 4;
        let (location, component) = if patch {
            patch_location(address).ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "AST.P reserved or misaligned patch attribute",
                )
            })?
        } else {
            attribute_location(stage, offset, encoding, address)?
        };
        let register = ShaderRegister::new(u16::from(source + lane));
        operations.push(if let Some(vertex) = vertex {
            ShaderOperation::StoreControlPoint {
                source: register,
                vertex,
                location,
                component,
            }
        } else {
            ShaderOperation::StoreOutput {
                sources: vec![register].into_boxed_slice(),
                location,
                first_component: component,
                scalar_type: ShaderScalarType::Float32,
            }
        });
    }
    Ok(operations)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(
        stage: MaxwellThreeDShaderStage,
        control_points: u8,
        patch_slots: u8,
    ) -> MaxwellShaderProgramHeader {
        let mut words = [0_u32; 20];
        words[0] = 0x60061
            | if stage == MaxwellThreeDShaderStage::TessellationInit {
                2 << 10
            } else {
                3 << 10
            };
        words[1] = u32::from(patch_slots) << 24;
        words[2] = u32::from(control_points) << 24;
        decode_program_header(
            &words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    #[test]
    fn control_level_prefix_executes_only_for_invocation_zero() {
        // Public deko_examples TCS prefix through the four level stores.
        // https://github.com/switchbrew/switch-examples/blob/master/graphics/deko3d/deko_examples/source/tess_simple_tcsh.glsl
        let code = [
            [
                0xf0c8_0000_0117_0000,
                0x5b64_0380_0ff7_0007,
                0x0104_0a00_0000_f000,
            ],
            [
                0xeff0_7f80_8100_ff00,
                0x0104_0000_0000_f001,
                0xeff0_7f80_8080_ff00,
            ],
            [
                0xeff0_7f80_8000_ff01,
                0x0104_0400_0000_f001,
                0xeff0_7f80_8040_ff01,
            ],
            [0xe300_0000_0007_000f, 0, 0],
        ];
        let binary = MaxwellShaderBinary {
            address: 0,
            header: header(MaxwellThreeDShaderStage::TessellationInit, 3, 6),
            bundles: code
                .into_iter()
                .enumerate()
                .map(|(index, instructions)| MaxwellShaderInstructionBundle {
                    offset: index as u32 * 32,
                    control: 0,
                    instructions,
                })
                .collect(),
            source_cpu_writes: Box::new([]),
            source_mappings: Box::new([]),
        };
        let translated = translate_shader_binary(&binary, 5, &BTreeMap::new()).unwrap();
        let ir =
            finalize_shader_ir(translated.ir, binary.header.stage, &BTreeMap::new(), &[]).unwrap();
        for invocation in 0..3 {
            let inputs = nixe_gpu::ShaderEvaluationInputs::default().with_interface_bits(
                ShaderIoLocation::InvocationId,
                0,
                invocation,
            );
            let result = nixe_gpu::evaluate_shader_ir(&ir, &inputs, 64).unwrap();
            for (location, component, value) in [
                (ShaderIoLocation::TessLevelOuter, 0, 2.0_f32),
                (ShaderIoLocation::TessLevelOuter, 1, 3.0),
                (ShaderIoLocation::TessLevelOuter, 2, 5.0),
                (ShaderIoLocation::TessLevelInner, 0, 5.0),
            ] {
                assert_eq!(
                    result.output_bits(location, component),
                    (invocation == 0).then_some(value.to_bits())
                );
            }
            assert_eq!(
                result.output_bits(ShaderIoLocation::TessLevelOuter, 3),
                None
            );
            assert_eq!(
                result.output_bits(ShaderIoLocation::TessLevelInner, 1),
                None
            );
        }
    }

    #[test]
    fn control_sph_and_patch_store_translate_without_wgsl() {
        let binary = MaxwellShaderBinary {
            address: 0,
            header: header(MaxwellThreeDShaderStage::TessellationInit, 5, 12),
            bundles: vec![MaxwellShaderInstructionBundle {
                offset: 0,
                control: 0,
                instructions: [
                    0x0104_0a00_0007_f000, // MOV32I R0, 5.0
                    0xeff0_7f80_8107_ff00, // AST.P a[0x10], R0
                    0xe300_0000_0007_000f,
                ],
            }]
            .into_boxed_slice(),
            source_cpu_writes: Box::new([]),
            source_mappings: Box::new([]),
        };
        let translated = translate_shader_binary(&binary, 4, &BTreeMap::new()).unwrap();
        let ir =
            finalize_shader_ir(translated.ir, binary.header.stage, &BTreeMap::new(), &[]).unwrap();
        assert_eq!(ir.ir().tessellation_control_points(), Some(5));
        assert!(
            ir.ir()
                .outputs()
                .iter()
                .any(|element| element.location() == ShaderIoLocation::Patch(0)
                    && element.component() == 3)
        );
        assert!(matches!(
            ir.ir().instructions()[1].operation(),
            ShaderOperation::StoreOutput {
                location: ShaderIoLocation::TessLevelInner,
                first_component: 0,
                ..
            }
        ));
        assert_eq!(
            ir.ir().instructions().len(),
            3,
            "unwritten shared outputs must not be overwritten at EXIT"
        );
        assert_eq!(
            ShaderBackendModule::new(ir).stage(),
            ShaderStage::TessellationControl
        );
    }

    #[test]
    fn direct_control_stores_use_invocation_index_not_input_patch_size() {
        let mut inputs = vec![];
        let operations = decode_control_store(
            MaxwellThreeDShaderStage::TessellationInit,
            8,
            0xeff1_ff80_0707_ff00,
            4,
            &mut 4,
            &mut inputs,
        )
        .unwrap();
        assert!(matches!(
            operations[0],
            ShaderOperation::LoadInput {
                location: ShaderIoLocation::InvocationId,
                ..
            }
        ));
        for (lane, operation) in operations[1..].iter().enumerate() {
            assert!(
                matches!(operation, ShaderOperation::StoreControlPoint { source, vertex, location: ShaderIoLocation::Position, component } if source.index() == lane as u16 && vertex.index() == 4 && *component == lane as u8)
            );
        }
        let operation = decode_system_register(
            MaxwellThreeDShaderStage::TessellationInit,
            16,
            0xf0c8_0000_0117_0002,
            4,
            &mut inputs,
        )
        .unwrap();
        assert!(matches!(
            operation,
            ShaderOperation::LoadInput {
                location: ShaderIoLocation::InvocationId,
                ..
            }
        ));
        assert_eq!(inputs.len(), 1);
        assert!(matches!(
            decode_system_register(
                MaxwellThreeDShaderStage::TessellationInit,
                24,
                0xf0c8_0000_01d7_0001,
                4,
                &mut inputs
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                instruction_offset: 24,
                ..
            })
        ));
    }

    fn program(
        stage: ShaderStage,
        inputs: Vec<(ShaderIoLocation, u8)>,
        outputs: Vec<(ShaderIoLocation, u8)>,
    ) -> TranslatedShaderIr {
        let elements = |list: Vec<(ShaderIoLocation, u8)>| {
            list.into_iter()
                .map(|(location, component)| interface_element(location, component, None))
                .collect()
        };
        TranslatedShaderIr {
            ir: ShaderIr::new(stage, elements(inputs), elements(outputs), vec![], vec![]),
            texture_bindings: Box::new([]),
        }
    }

    #[test]
    fn links_every_stage_component_instead_of_bypassing_control_and_evaluation() {
        use ShaderIoLocation::{Generic as G, Patch as P};
        let mut programs = vec![
            program(ShaderStage::Vertex, vec![], vec![(G(0), 2)]),
            program(
                ShaderStage::TessellationControl,
                vec![(G(0), 2)],
                vec![(G(1), 1), (P(0), 3)],
            ),
            program(
                ShaderStage::TessellationEvaluation,
                vec![(G(1), 1), (P(0), 3)],
                vec![(G(2), 0)],
            ),
            program(ShaderStage::Fragment, vec![(G(2), 0)], vec![]),
        ];
        validate_graphics_stage_interfaces(&programs).unwrap();
        programs[2] = program(
            ShaderStage::TessellationEvaluation,
            vec![(P(0), 2)],
            vec![(G(2), 0)],
        );
        assert!(matches!(
            validate_graphics_stage_interfaces(&programs),
            Err(MaxwellShaderTranslationError::StageInterfaceMismatch {
                location: P(0),
                component: 2,
                ..
            })
        ));
        programs.remove(1);
        assert!(matches!(
            validate_graphics_stage_interfaces(&programs),
            Err(MaxwellShaderTranslationError::StageInterfaceMismatch { location: P(0), .. })
        ));
    }

    #[test]
    fn fragment_interpolation_links_per_component_only_to_final_producer() {
        let stages = [
            MaxwellThreeDShaderStage::Vertex,
            MaxwellThreeDShaderStage::TessellationInit,
            MaxwellThreeDShaderStage::Tessellation,
            MaxwellThreeDShaderStage::Pixel,
        ];
        let programs: Box<[_]> = stages
            .into_iter()
            .enumerate()
            .map(|(index, stage)| {
                let mut words = [0_u32; 20];
                words[0] = match stage {
                    MaxwellThreeDShaderStage::Vertex => 0x0006_0461,
                    MaxwellThreeDShaderStage::TessellationInit => 0x0006_0861,
                    MaxwellThreeDShaderStage::Tessellation => 0x0006_0c61,
                    MaxwellThreeDShaderStage::Pixel => 0x0006_1462,
                    _ => unreachable!(),
                };
                match index {
                    0 => words[13] = 3 << 16, // VS: generic 0.xy
                    1 => {
                        words[2] = 5 << 24;
                        words[6] = 3;
                        words[13] = 3 << 20;
                    }
                    2 => {
                        words[6] = 3 << 4;
                        words[13] = 3 << 24;
                    }
                    3 => words[6] = 0b1101 << 16, // FS generic 2.x constant, 2.y linear
                    _ => unreachable!(),
                }
                let header = decode_program_header(
                    &words
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let binary = MaxwellShaderBinary {
                    address: 0,
                    header,
                    bundles: vec![MaxwellShaderInstructionBundle {
                        offset: 0,
                        control: 0,
                        instructions: [0xe300_0000_0007_000f; 3],
                    }]
                    .into_boxed_slice(),
                    source_cpu_writes: Box::new([]),
                    source_mappings: Box::new([]),
                };
                Arc::new(MaxwellShaderProgramTranslationInput {
                    pipeline: index as u8,
                    register_count: 4,
                    effective_group: None,
                    texture_constant_buffer_slot: None,
                    vertex_input_types: Box::new([]),
                    binary,
                })
            })
            .collect();
        let inputs = MaxwellShaderTranslationInputs {
            fingerprint: nixe_gpu::cache_fingerprint(&programs),
            programs,
        };
        let translated = translate_prepared_maxwell_shader_programs(&inputs).unwrap();
        for program in &translated[..2] {
            assert!(
                program
                    .module
                    .ir()
                    .ir()
                    .outputs()
                    .iter()
                    .all(|output| output.interpolation().is_none())
            );
        }
        assert_eq!(
            translated[1].module.ir().ir().tessellation_control_points(),
            Some(5)
        );
        let outputs = translated[2].module.ir().ir().outputs();
        assert_eq!(
            outputs[0].interpolation(),
            Some(ShaderInterpolation::Constant)
        );
        assert_eq!(
            outputs[1].interpolation(),
            Some(ShaderInterpolation::ScreenLinear)
        );
    }
}
