//! Physical quad oracle with distinct words per pixel and NaN-like payloads.
use super::*;
use nixe_gpu::{
    ShaderBitwiseOperation as Bits, ShaderFloatControl, ShaderQuadOperation as Quad,
    ShaderRoundingMode,
};

#[test]
fn fragment_quad_exchange_preserves_neighbor_pixels_and_raw_payloads() {
    quad_exchange(None, None);
}

#[test]
fn swizzled_quad_add_preserves_derivatives_and_all_four_lane_selectors() {
    for swizzle in [0x99, 0xa5, 0xe4] {
        quad_exchange(Some(swizzle), None);
    }
}

#[test]
fn fragment_front_facing_preserves_full_word_masks_and_programmed_orientation() {
    for face in [
        nixe_gpu::FrontFace::CounterClockwise,
        nixe_gpu::FrontFace::Clockwise,
    ] {
        quad_exchange(None, Some(face));
    }
}

fn quad_exchange(swizzle: Option<u8>, front_face: Option<nixe_gpu::FrontFace>) {
    let _guard = accelerated_test_guard();
    let Some(initialized) = initialize_backend(
        BackendInstanceId::new(760),
        NonCpuDeviceId::new(760),
        WgpuBackendConfiguration::default(),
    ) else {
        return;
    };
    assert!(
        front_face.is_some()
            || initialized
                .presentation_context()
                .device()
                .features()
                .contains(wgpu::Features::SUBGROUP),
        "physical quad test needs subgroups"
    );
    let runtime = RuntimeOwner::new(initialized.into_runtime());
    let (mut creations, output, image, subresources) =
        backed_color_image(ImageFormat::Rgba8Unorm, 4, 4, &[initialized_page(&[0; 64])]);
    let vertices = [
        [-1_f32, -1., 0., 1., 1., 1., 1.],
        [3., -1., 0., 1., 1., 1., 1.],
        [-1., 3., 0., 1., 1., 1., 1.],
    ]
    .into_iter()
    .flatten()
    .flat_map(f32::to_le_bytes)
    .collect::<Vec<_>>();
    let allocation = GpuAllocationId::new(760);
    let allocation_description = GpuAllocationDescription::new(vertices.len() as u64, 4).unwrap();
    let buffer = BufferId::new(760);
    let description = BufferDescription::new(vertices.len() as u64).unwrap();
    creations.extend([
        BackendResourceCreateInfo::Allocation {
            id: allocation,
            description: allocation_description,
        },
        BackendResourceCreateInfo::Buffer {
            id: buffer,
            description,
            view: Some(
                BufferView::new(
                    buffer,
                    description,
                    0,
                    backing(
                        allocation,
                        allocation_description,
                        &initialized_page(&vertices),
                    ),
                )
                .unwrap(),
            ),
        },
        BackendResourceCreateInfo::Shader {
            id: ShaderId::new(760),
            description: ShaderDescription {
                stage: ShaderStage::Vertex,
            },
            module: interpolated_vertex_module(ShaderInterpolation::Perspective, false),
        },
        BackendResourceCreateInfo::Pipeline {
            id: PipelineId::new(760),
            description: PipelineDescription {
                kind: PipelineKind::Graphics,
            },
        },
    ]);
    let pass = RenderPassId::new(760);
    let pass_description = RenderPassDescription::new(vec![RenderPassAttachmentDescription {
        kind: ImageKind::Color,
        format: ImageFormat::Rgba8Unorm,
        samples: SampleCount::One,
    }])
    .unwrap();
    creations.push(BackendResourceCreateInfo::RenderPass {
        id: pass,
        description: pass_description.clone(),
    });
    let quads = if front_face.is_some() {
        vec![Quad::Broadcast(0)]
    } else {
        match swizzle {
            Some(0x99) => vec![Quad::SwapHorizontal],
            Some(0xa5) => vec![Quad::SwapVertical],
            Some(_) => vec![Quad::Broadcast(0)],
            None => vec![
                Quad::Broadcast(0),
                Quad::Broadcast(1),
                Quad::Broadcast(2),
                Quad::Broadcast(3),
                Quad::SwapHorizontal,
                Quad::SwapVertical,
                Quad::SwapDiagonal,
            ],
        }
    };
    for (index, quad) in quads.into_iter().enumerate() {
        let fragment = ShaderId::new(761 + index as u64);
        creations.push(BackendResourceCreateInfo::Shader {
            id: fragment,
            description: ShaderDescription {
                stage: ShaderStage::Fragment,
            },
            module: if front_face.is_some() {
                front_facing_shader()
            } else {
                fragment_shader(quad, swizzle)
            },
        });
        let mut prepared = PreparedDraw::new(
            nixe_gpu::PreparedStateId::new(1, 1),
            PipelineId::new(760),
            pass,
            PrimitiveTopology::Triangles,
            vec![],
            vec![
                VertexBufferLayout::new(
                    BufferRegion {
                        buffer,
                        range: BufferRange::new(0, vertices.len() as u64).unwrap(),
                    },
                    28,
                    VertexStepMode::Vertex,
                    vec![
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 16,
                            shader_location: 1,
                        },
                    ],
                )
                .unwrap(),
            ],
            None,
        )
        .unwrap();
        if let Some(face) = front_face {
            prepared.edit_state().front_face = face;
        }
        let begin = RenderPassOperation::begin(
            pass,
            pass_description.clone(),
            vec![RenderAttachment {
                image,
                subresources,
                kind: ImageKind::Color,
                format: ImageFormat::Rgba8Unorm,
                samples: SampleCount::One,
                load: AttachmentLoad::Clear(ClearValue::Color([0.; 4])),
                store: AttachmentStore::Store,
            }],
        )
        .unwrap();
        let operation =
            |command| GpuOperation::new(command, [], [], CapabilityRequirements::none());
        let draw = GpuOperation::new(
            GpuCommand::Draw(
                DrawOperation::new(
                    Arc::new(prepared),
                    DrawArguments::NonIndexed {
                        first_vertex: 0,
                        vertex_count: 3,
                        first_instance: 0,
                        instance_count: 1,
                    },
                )
                .unwrap(),
            ),
            [],
            [
                ResourceDependency::Buffer(buffer),
                ResourceDependency::Shader(ShaderId::new(760)),
                ResourceDependency::Shader(fragment),
            ],
            CapabilityRequirements::none(),
        );
        runtime
            .runtime()
            .submit(
                &creations,
                &[],
                &OperationSubmission::new(
                    FrontendSubmissionId::new(index as u64 + 1),
                    vec![],
                    vec![
                        operation(GpuCommand::RenderPass(begin)),
                        draw,
                        operation(GpuCommand::RenderPass(RenderPassOperation::end(pass))),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        creations.clear();
        let mut pixels = [0; 64];
        output.range().read(0, &mut pixels).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                if let Some(face) = front_face {
                    let actual = &pixels[(y * 4 + x) * 4..(y * 4 + x + 1) * 4];
                    let expected = if face == nixe_gpu::FrontFace::CounterClockwise {
                        [255, 0, 0, 255]
                    } else {
                        [0, 255, 0, 255]
                    };
                    assert_eq!(actual, expected, "face={face:?}");
                    continue;
                }
                let (sx, sy) = match quad {
                    Quad::Broadcast(lane) => (
                        x / 2 * 2 + usize::from(lane & 1),
                        y / 2 * 2 + usize::from(lane >> 1),
                    ),
                    Quad::SwapHorizontal => (x ^ 1, y),
                    Quad::SwapVertical => (x, y ^ 1),
                    Quad::SwapDiagonal => (x ^ 1, y ^ 1),
                };
                let expected = match swizzle {
                    Some(0x99) => 16,
                    Some(0xa5) => 64,
                    Some(_) => [48, 112, 143, 80][(y & 1) * 2 + (x & 1)],
                    None => (((sx + sy * 4 + 1) as f32 / 16.) * 255.).round() as u8,
                };
                let actual = &pixels[(y * 4 + x) * 4..(y * 4 + x + 1) * 4];
                assert!(
                    actual[0].abs_diff(expected) <= 1 && actual[1..] == [0, 0, 255],
                    "quad={quad:?} x={x} y={y} actual={actual:?} expected={expected}"
                );
            }
        }
    }
}

fn fragment_shader(quad: Quad, swizzle: Option<u8>) -> nixe_gpu::ShaderBackendModule {
    let r = ShaderRegister::new;
    let mut code = vec![];
    let mut emit = |operation| {
        let offset = code.len() as u32 * 8;
        code.push(ShaderInstruction::new(
            ShaderSourceLocation::new(offset),
            ShaderPredicate::Always,
            operation,
        ));
    };
    emit(ShaderOperation::LoadInput {
        destinations: vec![r(0), r(1)].into_boxed_slice(),
        location: ShaderIoLocation::Position,
        first_component: 0,
        scalar_type: ShaderScalarType::Float32,
    });
    for register in [0, 1] {
        emit(ShaderOperation::ConvertFloat32ToInteger {
            destination: r(register),
            source: r(register),
            destination_bits: 32,
            flush_denormals_to_zero: false,
            destination_type: ShaderScalarType::Unsigned32,
            rounding: ShaderRoundingMode::TowardZero,
        });
    }
    for (reg, bits) in [
        (2, 4),
        (4, 0x7fa54300),
        (6, 1),
        (9, 0),
        (10, 1_f32.to_bits()),
        (11, (1_f32 / 16.).to_bits()),
    ] {
        emit(ShaderOperation::MoveImmediate32 {
            destination: r(reg),
            bits,
            scalar_type: ShaderScalarType::Unsigned32,
        });
    }
    emit(ShaderOperation::Multiply32 {
        destination: r(1),
        left: r(1),
        right: r(2),
        scalar_type: ShaderScalarType::Unsigned32,
        float_control: ShaderFloatControl::PRECISE,
    });
    emit(ShaderOperation::Add32 {
        destination: r(0),
        left: r(0),
        right: r(1),
        scalar_type: ShaderScalarType::Unsigned32,
        float_control: ShaderFloatControl::PRECISE,
    });
    if let Some(swizzle) = swizzle {
        if swizzle == 0xe4 {
            emit(ShaderOperation::MoveImmediate32 {
                destination: r(2),
                bits: (1_f32 / 8.).to_bits(),
                scalar_type: ShaderScalarType::Float32,
            });
            emit(ShaderOperation::MoveImmediate32 {
                destination: r(0),
                bits: (3_f32 / 16.).to_bits(),
                scalar_type: ShaderScalarType::Float32,
            });
        } else {
            emit(ShaderOperation::ConvertIntegerToFloat32 {
                destination: r(0),
                source: r(0),
                source_type: ShaderScalarType::Unsigned32,
                rounding: ShaderRoundingMode::NearestEven,
            });
            emit(ShaderOperation::QuadShuffle32 {
                destination: r(2),
                source: r(0),
                operation: quad,
            });
        }
        emit(ShaderOperation::FloatQuadSwizzleAdd32 {
            destination: r(0),
            left: r(2),
            right: r(0),
            swizzle,
            float_control: ShaderFloatControl::PRECISE,
        });
        if swizzle == 0xe4 {
            emit(ShaderOperation::MoveImmediate32 {
                destination: r(6),
                bits: 0.5_f32.to_bits(),
                scalar_type: ShaderScalarType::Float32,
            });
            emit(ShaderOperation::Add32 {
                destination: r(0),
                left: r(0),
                right: r(6),
                scalar_type: ShaderScalarType::Float32,
                float_control: ShaderFloatControl::PRECISE,
            });
        } else {
            emit(ShaderOperation::Multiply32 {
                destination: r(0),
                left: r(0),
                right: r(11),
                scalar_type: ShaderScalarType::Float32,
                float_control: ShaderFloatControl::PRECISE,
            });
        }
    } else {
        emit(ShaderOperation::Bitwise32 {
            destination: r(0),
            left: r(0),
            right: r(4),
            operation: Bits::Xor,
        });
        emit(ShaderOperation::QuadShuffle32 {
            destination: r(0),
            source: r(0),
            operation: quad,
        });
        emit(ShaderOperation::Bitwise32 {
            destination: r(0),
            left: r(0),
            right: r(4),
            operation: Bits::Xor,
        });
        emit(ShaderOperation::Add32 {
            destination: r(0),
            left: r(0),
            right: r(6),
            scalar_type: ShaderScalarType::Unsigned32,
            float_control: ShaderFloatControl::PRECISE,
        });
        emit(ShaderOperation::ConvertIntegerToFloat32 {
            destination: r(0),
            source: r(0),
            source_type: ShaderScalarType::Unsigned32,
            rounding: ShaderRoundingMode::NearestEven,
        });
        emit(ShaderOperation::Multiply32 {
            destination: r(0),
            left: r(0),
            right: r(11),
            scalar_type: ShaderScalarType::Float32,
            float_control: ShaderFloatControl::PRECISE,
        });
    }
    emit(ShaderOperation::StoreOutput {
        sources: vec![r(0), r(9), r(9), r(10)].into_boxed_slice(),
        location: ShaderIoLocation::Color(0),
        first_component: 0,
        scalar_type: ShaderScalarType::Float32,
    });
    emit(ShaderOperation::Exit);
    nixe_gpu::ShaderBackendModule::new(
        VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            (0..2)
                .map(|c| {
                    ShaderInterfaceElement::new(
                        ShaderIoLocation::Position,
                        c,
                        ShaderScalarType::Float32,
                        None,
                    )
                    .unwrap()
                })
                .collect(),
            (0..4)
                .map(|c| {
                    ShaderInterfaceElement::new(
                        ShaderIoLocation::Color(0),
                        c,
                        ShaderScalarType::Float32,
                        None,
                    )
                    .unwrap()
                })
                .collect(),
            vec![],
            code,
        ))
        .unwrap(),
    )
}

fn front_facing_shader() -> nixe_gpu::ShaderBackendModule {
    use nixe_gpu::{ShaderIntegerComparison as C, ShaderPredicateSetOperation as P};
    let r = ShaderRegister::new;
    let mut ops = vec![ShaderOperation::LoadInput {
        destinations: vec![r(0)].into_boxed_slice(),
        location: ShaderIoLocation::FrontFacing,
        first_component: 0,
        scalar_type: ShaderScalarType::Unsigned32,
    }];
    for (register, bits) in [(1, u32::MAX), (2, 0), (3, 1_f32.to_bits())] {
        ops.push(ShaderOperation::MoveImmediate32 {
            destination: r(register),
            bits,
            scalar_type: ShaderScalarType::Unsigned32,
        });
    }
    for (destination, right) in [(4, 1), (5, 2)] {
        ops.push(ShaderOperation::CompareInteger32 {
            destination: r(destination),
            float_result: true,
            left: r(0),
            right: r(right),
            signed: false,
            comparison: C::Equal,
            accumulator: ShaderPredicate::Always,
            set_operation: P::And,
        });
    }
    ops.push(ShaderOperation::StoreOutput {
        sources: vec![r(4), r(5), r(2), r(3)].into_boxed_slice(),
        location: ShaderIoLocation::Color(0),
        first_component: 0,
        scalar_type: ShaderScalarType::Float32,
    });
    ops.push(ShaderOperation::Exit);
    let code = ops
        .into_iter()
        .enumerate()
        .map(|(i, op)| {
            ShaderInstruction::new(
                ShaderSourceLocation::new(i as u32 * 8),
                ShaderPredicate::Always,
                op,
            )
        })
        .collect();
    nixe_gpu::ShaderBackendModule::new(
        VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            vec![
                ShaderInterfaceElement::new(
                    ShaderIoLocation::FrontFacing,
                    0,
                    ShaderScalarType::Unsigned32,
                    None,
                )
                .unwrap(),
            ],
            (0..4)
                .map(|c| {
                    ShaderInterfaceElement::new(
                        ShaderIoLocation::Color(0),
                        c,
                        ShaderScalarType::Float32,
                        None,
                    )
                    .unwrap()
                })
                .collect(),
            vec![],
            code,
        ))
        .unwrap(),
    )
}
