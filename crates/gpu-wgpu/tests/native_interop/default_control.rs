//! Execute the generated semantic control adapter with dynamic raw-bit levels.
use super::{SIZE, device::Context, emitted::*, native, normal};
use nixe_gpu::*;
type Op = ShaderOperation;
type Loc = ShaderIoLocation;
type Ty = ShaderScalarType;
use ShaderRegister as R;

pub(super) fn shaders(points: u8) -> [VerifiedShaderIr; 3] {
    let vs = ir(
        ShaderStage::Vertex,
        vec![interface(Loc::VertexId, 0, Ty::Unsigned32)],
        vec![interface(Loc::Generic(0), 1, Ty::Float32)],
        vec![
            load(0, Loc::VertexId, 0, Ty::Unsigned32),
            immediate(1, u32::from(points - 1)),
            immediate(2, 0),
            Op::SetPredicateInteger32 {
                destinations: [Some(0), None],
                left: R::new(0),
                right: R::new(1),
                signed: false,
                comparison: ShaderIntegerComparison::Equal,
                accumulator: ShaderPredicate::Always,
                set_operation: ShaderPredicateSetOperation::And,
            },
            immediate(2, 1_f32.to_bits()),
            store(2, Loc::Generic(0), 1),
            Op::Exit,
        ],
    );
    let mut code = vs.instructions().to_vec();
    code[4] = ShaderInstruction::new(
        code[4].source(),
        ShaderPredicate::Register {
            register: 0,
            inverted: false,
        },
        code[4].operation().clone(),
    );
    let vs = ShaderIr::new(
        vs.stage(),
        vs.inputs().to_vec(),
        vs.outputs().to_vec(),
        vec![],
        code,
    );
    let inputs = vec![
        interface(Loc::Generic(0), 1, Ty::Float32),
        interface(Loc::TessCoord, 0, Ty::Float32),
        interface(Loc::TessCoord, 1, Ty::Float32),
        interface(Loc::TessLevelInner, 0, Ty::Float32),
        interface(Loc::TessLevelOuter, 3, Ty::Float32),
    ];
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
    code.extend([
        // The LAST input point must survive the adapter unchanged for every
        // cardinality (not only triangles or control point zero).
        immediate(4, u32::from(points - 1)),
        Op::LoadControlPoint {
            destination: R::new(5),
            vertex: R::new(4),
            output: false,
            location: Loc::Generic(0),
            component: 1,
        },
        store(5, Loc::Generic(0), 1),
        load(6, Loc::TessLevelInner, 0, Ty::Float32),
        store(6, Loc::Generic(0), 0),
        // This lane isn't consumed by triangle subdivision, but TES reads it.
        load(7, Loc::TessLevelOuter, 3, Ty::Float32),
        store(7, Loc::Generic(0), 2),
        Op::Exit,
    ]);
    let te = ir(ShaderStage::TessellationEvaluation, inputs, outputs, code);
    let [_, _, _, fs] = super::emitted::shaders();
    [
        VerifiedShaderIr::verify(vs).unwrap(),
        VerifiedShaderIr::verify(te).unwrap(),
        fs,
    ]
}

fn validate(module: &SpirvShaderModule) {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        file.path(),
        module
            .words()
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let result = std::process::Command::new(std::env::var_os("SPIRV_VAL").expect("set SPIRV_VAL"))
        .args(["--target-env", "vulkan1.1"])
        .arg(file.path())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

pub fn check(ctx: &Context) {
    let caps = ctx.capabilities.unwrap();
    for points in [1, 4, 5, 32] {
        let [vs, te, fs] = shaders(points);
        let shaders = lower_tessellation_shaders_to_spirv(
            &vs,
            None,
            &te,
            &fs,
            SpirvTessellationOptions {
                depth_clip_negative_one_to_one: false,
                input_control_points: points,
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
        assert!(shaders.bindings().is_empty());
        for module in shaders.modules() {
            validate(module);
        }
        // Exactly one native pipeline for all level changes below.
        let pipeline = native::Pipeline::from_tessellation(ctx, &shaders);
        let normal = normal::NormalPipelines::new(ctx);
        for (outer, inner, unused, expected) in [
            (1_f32, 0.25_f32, 0.5_f32, [64, 255, 128, 64]),
            (4_f32, 0.75_f32, 0.25_f32, [191, 255, 64, 64]),
            (0_f32, 0.75_f32, 0.25_f32, [0, 0, 0, 255]),
            (2_f32, 0.25_f32, 0.5_f32, [64, 255, 128, 64]),
        ] {
            let words = shaders
                .parameters(TessellationControl::DefaultLevels {
                    outer: [
                        outer.to_bits(),
                        outer.to_bits(),
                        outer.to_bits(),
                        unused.to_bits(),
                    ],
                    inner: [inner.to_bits(), 0x7fc0_1234],
                    defined: 0b01_1111,
                })
                .unwrap()
                .unwrap();
            let bytes: Vec<_> = words.into_iter().flat_map(u32::to_ne_bytes).collect();
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
            r.handoff(&mut before);
            let draw = bindings.encode_parameters(ctx, 1, &bytes);
            let mut after = ctx.device.create_command_encoder(&Default::default());
            let pixels = normal.sample(ctx, &r, &mut after, 1);
            ctx.queue.submit([before.finish(), draw, after.finish()]);
            let pixels = normal::read_pixels(ctx, pixels);
            for (x, y) in [(18, 18), (20, 20), (17, 24)] {
                let actual = pixels[(y * SIZE + x) as usize];
                // UNORM conversion/interpolation may differ by one unit across
                // drivers. Patch suppression and forwarded green remain exact.
                for channel in 0..4 {
                    assert!(
                        actual[channel].abs_diff(expected[channel]) <= 1,
                        "points={points}, outer={outer}, pixel=({x},{y}), actual={actual:?}, expected={expected:?}"
                    );
                }
            }
            assert_eq!(pixels[(4 * SIZE + 4) as usize], [0, 0, 0, 255]);
        }
    }
}
