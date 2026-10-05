//! Execute neutral IR output, not GLSL replacements for guest shaders. This
//! isolates patch I/O/barriers from the still-pending float arithmetic contract.
use super::{SIZE, device::Context, native, normal};
use nixe_gpu::*;

type Op = ShaderOperation;
type Loc = ShaderIoLocation;
type Ty = ShaderScalarType;
use ShaderRegister as R;

pub(super) fn interface(location: Loc, component: u8, ty: Ty) -> ShaderInterfaceElement {
    ShaderInterfaceElement::new(location, component, ty, None).unwrap()
}
pub(super) fn immediate(register: u16, bits: u32) -> Op {
    Op::MoveImmediate32 {
        destination: R::new(register),
        bits,
        scalar_type: Ty::Unsigned32,
    }
}
pub(super) fn load(register: u16, location: Loc, component: u8, scalar_type: Ty) -> Op {
    Op::LoadInput {
        destinations: vec![R::new(register)].into(),
        location,
        first_component: component,
        scalar_type,
    }
}
pub(super) fn store(register: u16, location: Loc, component: u8) -> Op {
    Op::StoreOutput {
        sources: vec![R::new(register)].into(),
        location,
        first_component: component,
        scalar_type: Ty::Float32,
    }
}
pub(super) fn ir(
    stage: ShaderStage,
    inputs: Vec<ShaderInterfaceElement>,
    outputs: Vec<ShaderInterfaceElement>,
    code: Vec<Op>,
) -> ShaderIr {
    let instructions = code
        .into_iter()
        .enumerate()
        .map(|(index, op)| {
            ShaderInstruction::new(
                ShaderSourceLocation::new(index as u32 * 8),
                ShaderPredicate::Always,
                op,
            )
        })
        .collect();
    ShaderIr::new(stage, inputs, outputs, vec![], instructions)
}

pub(super) fn shaders() -> [VerifiedShaderIr; 4] {
    let vs = ir(
        ShaderStage::Vertex,
        vec![],
        vec![interface(Loc::Generic(0), 0, Ty::Float32)],
        vec![
            immediate(0, 1_f32.to_bits()),
            store(0, Loc::Generic(0), 0),
            Op::Exit,
        ],
    );
    let mut outputs: Vec<_> = (0..3)
        .map(|c| interface(Loc::Generic(0), c, Ty::Float32))
        .collect();
    outputs.extend((0..3).map(|c| interface(Loc::TessLevelOuter, c, Ty::Float32)));
    outputs.push(interface(Loc::TessLevelInner, 0, Ty::Float32));
    let mut tc = ir(
        ShaderStage::TessellationControl,
        vec![
            interface(Loc::InvocationId, 0, Ty::Unsigned32),
            interface(Loc::Generic(0), 0, Ty::Float32),
        ],
        outputs,
        vec![
            load(0, Loc::InvocationId, 0, Ty::Unsigned32),
            immediate(1, 2),
            immediate(2, 4),
            Op::Multiply32 {
                destination: R::new(2),
                left: R::new(0),
                right: R::new(2),
                scalar_type: Ty::Unsigned32,
                float_control: ShaderFloatControl::PRECISE,
            },
            Op::LoadConstantBufferIndexed32 {
                destination: R::new(3),
                binding: 0,
                base_byte_offset: 0,
                dynamic_byte_offset: R::new(2),
                scalar_type: Ty::Float32,
            },
            Op::StoreControlPoint {
                source: R::new(3),
                vertex: R::new(0),
                location: Loc::Generic(0),
                component: 0,
            },
            Op::LoadControlPoint {
                destination: R::new(4),
                vertex: R::new(0),
                output: false,
                location: Loc::Generic(0),
                component: 0,
            },
            Op::StoreControlPoint {
                source: R::new(4),
                vertex: R::new(0),
                location: Loc::Generic(0),
                component: 1,
            },
            Op::PatchBarrier,
            // Every invocation reads invocation two's buffer-derived output.
            Op::LoadControlPoint {
                destination: R::new(5),
                vertex: R::new(1),
                output: true,
                location: Loc::Generic(0),
                component: 0,
            },
            Op::StoreControlPoint {
                source: R::new(5),
                vertex: R::new(0),
                location: Loc::Generic(0),
                component: 2,
            },
            immediate(6, 0),
            Op::SetPredicateInteger32 {
                destinations: [Some(0), None],
                left: R::new(0),
                right: R::new(6),
                signed: false,
                comparison: ShaderIntegerComparison::Equal,
                accumulator: ShaderPredicate::Always,
                set_operation: ShaderPredicateSetOperation::And,
            },
            immediate(7, 1_f32.to_bits()),
            store(7, Loc::TessLevelOuter, 0),
            store(7, Loc::TessLevelOuter, 1),
            store(7, Loc::TessLevelOuter, 2),
            store(7, Loc::TessLevelInner, 0),
            Op::Exit,
        ],
    )
    .with_tessellation_control_points(Some(3));
    // Build via the public IR constructor; only invocation zero writes patch levels.
    let mut code = tc.instructions().to_vec();
    for instruction in code.iter_mut().filter(|i| {
        matches!(
            i.operation(),
            Op::StoreOutput {
                location: Loc::TessLevelOuter | Loc::TessLevelInner,
                ..
            }
        )
    }) {
        *instruction = ShaderInstruction::new(
            instruction.source(),
            ShaderPredicate::Register {
                register: 0,
                inverted: false,
            },
            instruction.operation().clone(),
        );
    }
    tc = ShaderIr::new(
        tc.stage(),
        tc.inputs().to_vec(),
        tc.outputs().to_vec(),
        vec![
            ShaderResourceAccess::new(0, ShaderResourceKind::ConstantBuffer, true, false).unwrap(),
        ],
        code,
    )
    .with_tessellation_control_points(Some(3));
    let mut inputs: Vec<_> = (0..3)
        .map(|c| interface(Loc::Generic(0), c, Ty::Float32))
        .collect();
    inputs.extend((0..2).map(|c| interface(Loc::TessCoord, c, Ty::Float32)));
    let mut outputs: Vec<_> = (0..4)
        .map(|c| interface(Loc::Position, c, Ty::Float32))
        .collect();
    outputs.extend((0..3).map(|c| {
        ShaderInterfaceElement::new(
            Loc::Generic(0),
            c,
            Ty::Float32,
            Some(ShaderInterpolation::Perspective),
        )
        .unwrap()
    }));
    let mut code = vec![
        load(0, Loc::TessCoord, 0, Ty::Float32),
        load(1, Loc::TessCoord, 1, Ty::Float32),
        immediate(2, 0.25_f32.to_bits()),
        immediate(3, 1_f32.to_bits()),
    ];
    code.extend((0..4).map(|c| store(c, Loc::Position, c as u8)));
    code.push(immediate(4, 0));
    for component in 0..3 {
        code.push(Op::LoadControlPoint {
            destination: R::new(5),
            vertex: R::new(4),
            output: false,
            location: Loc::Generic(0),
            component,
        });
        code.push(store(5, Loc::Generic(0), component));
    }
    code.push(Op::Exit);
    let te = ir(ShaderStage::TessellationEvaluation, inputs, outputs, code);
    let inputs = (0..3)
        .map(|c| {
            ShaderInterfaceElement::new(
                Loc::Generic(0),
                c,
                Ty::Float32,
                Some(ShaderInterpolation::Perspective),
            )
            .unwrap()
        })
        .collect();
    let mut code = Vec::new();
    for component in 0..3 {
        code.push(Op::InterpolateInput {
            destination: R::new(u16::from(component)),
            location: Loc::Generic(0),
            component,
            interpolation: ShaderInterpolation::Perspective,
        });
        code.push(store(u16::from(component), Loc::Color(0), component));
    }
    code.extend([
        immediate(3, 1_f32.to_bits()),
        store(3, Loc::Color(0), 3),
        Op::Exit,
    ]);
    let fs = ir(
        ShaderStage::Fragment,
        inputs,
        (0..4)
            .map(|c| interface(Loc::Color(0), c, Ty::Float32))
            .collect(),
        code,
    );
    [vs, tc, te, fs].map(|ir| VerifiedShaderIr::verify(ir).unwrap())
}

pub fn check(ctx: &Context) {
    check_bindings(ctx, false);
    check_bindings(ctx, true);
}

pub(super) fn shared_buffer_shaders() -> [VerifiedShaderIr; 4] {
    let mut stages = shaders();
    // One descriptor is consumed by all four stages. Each extra read reaches
    // a distinct checked output; unused declarations must not create bindings.
    for (index, offset) in [(0, 4), (2, 12), (3, 0)] {
        let ir = stages[index].ir();
        let code = ir
            .instructions()
            .iter()
            .map(|instruction| {
                let register = match (index, instruction.operation()) {
                    (0, Op::MoveImmediate32 { destination, .. }) if destination.index() == 0 => {
                        Some(*destination)
                    }
                    (2, Op::MoveImmediate32 { destination, .. }) if destination.index() == 2 => {
                        Some(*destination)
                    }
                    (
                        3,
                        Op::InterpolateInput {
                            destination,
                            component: 0,
                            ..
                        },
                    ) => Some(*destination),
                    _ => None,
                };
                let operation = register.map_or_else(
                    || instruction.operation().clone(),
                    |destination| Op::LoadConstantBuffer32 {
                        destination,
                        binding: 0,
                        byte_offset: offset,
                        scalar_type: Ty::Float32,
                    },
                );
                ShaderInstruction::new(instruction.source(), instruction.predicate(), operation)
            })
            .collect();
        stages[index] = VerifiedShaderIr::verify(ShaderIr::new(
            ir.stage(),
            ir.inputs().to_vec(),
            ir.outputs().to_vec(),
            [0, 200]
                .map(|binding| {
                    ShaderResourceAccess::new(
                        binding,
                        ShaderResourceKind::ConstantBuffer,
                        true,
                        false,
                    )
                    .unwrap()
                })
                .into(),
            code,
        ))
        .unwrap();
    }
    stages
}

fn check_bindings(ctx: &Context, shared: bool) {
    let caps = ctx.capabilities.unwrap();
    let [vs, tc, te, fs] = if shared {
        shared_buffer_shaders()
    } else {
        shaders()
    };
    let shaders = lower_tessellation_shaders_to_spirv(
        &vs,
        Some(&tc),
        &te,
        &fs,
        SpirvTessellationOptions {
            depth_clip_negative_one_to_one: false,
            input_control_points: 3,
            mode: TessellationMode {
                domain: TessellationDomain::Triangles,
                spacing: TessellationSpacing::Equal,
                output: TessellationOutput::Triangles(TessellationWinding::CounterClockwise),
            },
            float32: caps.float32,
            float64: caps.float64,
        },
    )
    .unwrap();
    assert_eq!(shaders.bindings().len(), 1);
    assert_eq!(
        shaders.bindings()[0].stages,
        if shared {
            PipelineStages::TESSELLATION_CONTROL_SHADER
                .union(PipelineStages::VERTEX_SHADER)
                .union(PipelineStages::TESSELLATION_EVALUATION_SHADER)
                .union(PipelineStages::FRAGMENT_SHADER)
        } else {
            PipelineStages::TESSELLATION_CONTROL_SHADER
        }
    );
    for module in shaders.modules() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            temp.path(),
            module
                .words()
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let validation =
            std::process::Command::new(std::env::var_os("SPIRV_VAL").unwrap_or("spirv-val".into()))
                .args(["--target-env", "vulkan1.1"])
                .arg(temp.path())
                .output()
                .unwrap();
        assert!(
            validation.status.success(),
            "{}",
            String::from_utf8_lossy(&validation.stderr)
        );
    }
    let pipeline = native::Pipeline::from_tessellation(ctx, &shaders);
    let normal = normal::NormalPipelines::new(ctx);
    let cases = if shared {
        [
            ([0.25, 1.0, 1.0, 0.25], [64, 255, 255, 64]),
            // Keep interpolated colors away from UNORM halfway ties (0.5),
            // where permitted interpolation error changes the final byte.
            ([0.75, 0.25, 0.0, 0.75], [191, 64, 0, 191]),
        ]
    } else {
        [
            ([0.0, 1.0, 1.0, 0.25], [0, 255, 255, 64]),
            ([0.0, 1.0, 0.0, 0.25], [0, 255, 0, 64]),
        ]
    };
    for (words, color) in cases {
        let r = normal::Resources::new(ctx);
        let bindings = native::Bindings::new(
            pipeline.clone(),
            &r.buffer,
            [&r.color, &r.depth, &r.image],
            [&r.color_views[1], &r.depth_views[1], &r.image_view],
        );
        let mut before = ctx.device.create_command_encoder(&Default::default());
        r.initialize(&mut before);
        r.upload(ctx, &mut before);
        // Initialize all four words through the normal queue; the sample's other
        // fixture upload is deliberately superseded in this normal segment.
        use wgpu::util::DeviceExt;
        let bytes: Vec<_> = words.into_iter().flat_map(f32::to_le_bytes).collect();
        let upload = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("native IR oracle inputs"),
                contents: &bytes,
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        before.copy_buffer_to_buffer(&upload, 0, &r.buffer, 0, 16);
        r.handoff(&mut before);
        let draw = bindings.encode(ctx, 1);
        let mut after = ctx.device.create_command_encoder(&Default::default());
        let pixels = normal.sample(ctx, &r, &mut after, 1);
        ctx.queue.submit([before.finish(), draw, after.finish()]);
        let pixels = normal::read_pixels(ctx, pixels);
        // Interior/exterior samples isolate shader/patch semantics from edge-fill
        // conventions. The direct TessCoord position occupies one NDC quadrant.
        for (x, y) in [(18, 18), (20, 20), (17, 24)] {
            assert_eq!(
                pixels[(y * SIZE + x) as usize],
                color,
                "emitted shader pixel ({x}, {y})"
            );
        }
        assert_eq!(pixels[(4 * SIZE + 4) as usize], [0, 0, 0, 255]);
    }
}
