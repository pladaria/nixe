//! GPU numeric oracle for native SPIR-V arithmetic and portable float conversions.
//! Finite/Inf/zero results are bit-exact; propagated NaN payloads are unspecified.
use super::*;
use nixe_gpu::*;
use wgpu::util::DeviceExt;

const SIDE: u32 = 256;
type Op = ShaderOperation;
type Ty = ShaderScalarType;
type Loc = ShaderIoLocation;
use ShaderRegister as R;

fn shader(operator: u8) -> VerifiedShaderIr {
    let control = ShaderFloatControl::new(
        ShaderRoundingMode::NearestEven,
        if operator == 5 {
            ShaderNanMode::Propagate
        } else {
            ShaderNanMode::Canonicalize
        },
        true,
        true,
        false,
    );
    let mut code = vec![
        Op::LoadInput {
            destinations: vec![R::new(0)].into(),
            location: Loc::Generic(0),
            first_component: 0,
            scalar_type: Ty::Unsigned32,
        },
        Op::MoveImmediate32 {
            destination: R::new(1),
            bits: 12,
            scalar_type: Ty::Unsigned32,
        },
        Op::Multiply32 {
            destination: R::new(0),
            left: R::new(0),
            right: R::new(1),
            scalar_type: Ty::Unsigned32,
            float_control: ShaderFloatControl::PRECISE,
        },
    ];
    for i in 0..3 {
        code.push(Op::LoadConstantBufferIndexed32 {
            destination: R::new(2 + i),
            binding: 0,
            base_byte_offset: i32::from(i) * 4,
            dynamic_byte_offset: R::new(0),
            scalar_type: Ty::Float32,
        });
    }
    code.push(match operator {
        0 => Op::Add32 {
            destination: R::new(5),
            left: R::new(2),
            right: R::new(3),
            scalar_type: Ty::Float32,
            float_control: control,
        },
        1 => Op::Multiply32 {
            destination: R::new(5),
            left: R::new(2),
            right: R::new(3),
            scalar_type: Ty::Float32,
            float_control: control,
        },
        2 => Op::FusedMultiplyAdd32 {
            destination: R::new(5),
            left: R::new(2),
            right: R::new(3),
            addend: R::new(4),
            float_control: control,
        },
        3 | 4 => Op::ConvertIntegerToFloat32 {
            destination: R::new(5),
            source: R::new(2),
            source_type: if operator == 3 {
                Ty::Signed32
            } else {
                Ty::Unsigned32
            },
        },
        5 => Op::FloatMultiplyZero32 {
            destination: R::new(5),
            left: R::new(2),
            right: R::new(3),
            float_control: control,
        },
        6 | 7 => Op::UnpackHalf32 {
            destination: R::new(5),
            source: R::new(2),
            high: operator == 7,
        },
        8 => Op::PackHalf32 {
            destination: R::new(5),
            source: R::new(2),
        },
        _ => unreachable!(),
    });
    code.push(Op::StoreOutput {
        sources: vec![R::new(5)].into(),
        location: Loc::Color(0),
        first_component: 0,
        scalar_type: Ty::Unsigned32,
    });
    code.push(Op::Exit);
    VerifiedShaderIr::verify(ShaderIr::new(
        ShaderStage::Fragment,
        vec![
            ShaderInterfaceElement::new(
                Loc::Generic(0),
                0,
                Ty::Unsigned32,
                Some(ShaderInterpolation::Constant),
            )
            .unwrap(),
        ],
        vec![ShaderInterfaceElement::new(Loc::Color(0), 0, Ty::Unsigned32, None).unwrap()],
        vec![
            ShaderResourceAccess::new(0, ShaderResourceKind::ConstantBuffer, true, false).unwrap(),
        ],
        code.into_iter()
            .enumerate()
            .map(|(i, op)| {
                ShaderInstruction::new(
                    ShaderSourceLocation::new(i as u32 * 8),
                    ShaderPredicate::Always,
                    op,
                )
            })
            .collect(),
    ))
    .unwrap()
}

fn cases() -> Vec<[u32; 3]> {
    let mut cases = Vec::new();
    // Integer conversion: signed boundaries and both nearest-even halfway ties.
    for bits in [
        0,
        1,
        u32::MAX,
        i32::MIN as u32,
        i32::MAX as u32,
        0x0100_0001,
        0x0100_0003,
        0xfeff_ffff,
        0xfeff_fffd,
    ] {
        cases.push([bits, 0, 0]);
    }
    // Signed zeros, subnormal inputs, cancellation, rounding boundary, Inf/NaN.
    let special = [
        0,
        0x8000_0000,
        1,
        0x807f_ffff,
        0x0080_0000,
        0x0080_0001,
        0x8080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3f7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0001,
        0x7f80_0001,
        0x7f7f_ffff,
    ];
    for a in special {
        for b in special {
            for c in [0, 0x8000_0000, 0x0080_0000, 0x8080_0000] {
                cases.push([a, b, c]);
            }
        }
    }
    // Midpoint multiplication and FMA with a low product term lost by naive
    // float64 summation. The latter must not be implemented by double rounding.
    for sign in [0, 0x8000_0000] {
        cases.push([0x0080_0000 ^ sign, 0x3f7f_ffff, 0]);
        let a = (2_f32.powi(-75) * (1.0 + 2_f32.powi(-23))).to_bits();
        let b = (2_f32.powi(-75) * (1.0 - 2_f32.powi(-23))).to_bits();
        cases.push([a ^ sign, b, 0x8080_0000 ^ sign]);
        // (2^23+2048)*(2^24-4095) = 2^47+2048. The product
        // is just ABOVE half a subnormal ULP; adding -MIN_POSITIVE gives
        // a magnitude just BELOW the midpoint. Naive FP64 -> FP32 rounds
        // twice and returns MIN_POSITIVE instead of the required signed zero.
        cases.push([0x1a00_0800 ^ sign, 0x19ff_f001, 0x8080_0000 ^ sign]);
    }
    let mut random = 0x65b4_ae37_u32;
    let mut next = || {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        random
    };
    while cases.len() < (SIDE * SIDE) as usize {
        let i = cases.len() as u32;
        let a = next();
        let b = next();
        let c = next();
        cases.push(match i % 4 {
            0 => [a, b, c],
            // Product near float32 underflow, both signs and varying mantissas.
            1 => [
                (a & 0x807f_ffff) | (63 << 23),
                (b & 0x807f_ffff) | ((63 + (i / 4) % 4) << 23),
                c & 0x80ff_ffff,
            ],
            // Cancellation around MIN_POSITIVE and the exponent-sum guard.
            2 => {
                let a = (a & 0x807f_ffff) | ((1 + (i / 4) % 254) << 23);
                [a, 0xbf80_0000, a]
            }
            _ => [
                (a & 0x807f_ffff) | (52 << 23),
                (b & 0x807f_ffff) | (52 << 23),
                0x8080_0000 | (c & 0x007f_ffff),
            ],
        });
    }
    cases
}

#[test]
fn underflow_fixture_exposes_double_rounding() {
    let a = f32::from_bits(0x1a00_0800);
    let b = f32::from_bits(0x19ff_f001);
    let c = -f32::MIN_POSITIVE;
    assert_eq!(a.mul_add(b, c).to_bits(), 0x807f_ffff);
    let double_rounded = (f64::from(a) * f64::from(b) + f64::from(c)) as f32;
    assert_eq!(double_rounded.to_bits(), 0x8080_0000);
    assert!(cases().contains(&[a.to_bits(), b.to_bits(), c.to_bits()]));
}

fn validate(module: &SpirvShaderModule) {
    let path = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        path.path(),
        module
            .words()
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let output =
        std::process::Command::new(std::env::var_os("NIXE_SPIRV_VAL").expect("set NIXE_SPIRV_VAL"))
            .args(["--target-env", "vulkan1.1"])
            .arg(path.path())
            .output()
            .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "requires Vulkan, float64 numerical guarantees and NIXE_SPIRV_VAL; FMA additionally needs VK_KHR_shader_fma"]
fn native_float_conversions_and_daz_ftz_match_ir_bits() {
    let _guard = NATIVE_DEVICE_TEST_LOCK.lock().unwrap();
    if !crate::test_hardware::available(wgpu::Backends::VULKAN) {
        return;
    }
    enable_validation();
    run_numeric_oracle();
    assert_eq!(
        VALIDATION_ERRORS.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "Vulkan validation errors, including teardown"
    );
}

struct ValidationLogger;
static VALIDATION_LOGGER: ValidationLogger = ValidationLogger;
static VALIDATION_ERRORS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
impl log::Log for ValidationLogger {
    fn enabled(&self, m: &log::Metadata<'_>) -> bool {
        m.level() <= log::Level::Warn
    }
    fn log(&self, r: &log::Record<'_>) {
        if self.enabled(r.metadata()) {
            eprintln!("{}: {}", r.level(), r.args());
            if r.level() == log::Level::Error {
                VALIDATION_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    fn flush(&self) {}
}

fn enable_validation() {
    assert!(
        wgpu::InstanceFlags::default()
            .contains(wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION),
        "requires debug test build"
    );
    log::set_logger(&VALIDATION_LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Warn);
    // SAFETY: read-only loader queries, no borrowed handles retained.
    unsafe {
        let loader = ash::Entry::load().unwrap();
        let layer = c"VK_LAYER_KHRONOS_validation";
        assert!(
            loader
                .enumerate_instance_layer_properties()
                .unwrap()
                .iter()
                .any(|p| std::ffi::CStr::from_ptr(p.layer_name.as_ptr()) == layer),
            "requires Khronos validation"
        );
        assert!(
            loader
                .enumerate_instance_extension_properties(Some(layer))
                .unwrap()
                .iter()
                .any(|p| std::ffi::CStr::from_ptr(p.extension_name.as_ptr())
                    == ash::ext::validation_features::NAME),
            "requires synchronization validation"
        );
    }
}

fn run_numeric_oracle() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let Some(adapter) = crate::test_hardware::adapter(&instance, wgpu::Backends::VULKAN) else {
        return;
    };
    let created = create_device(
        &instance,
        &adapter,
        &wgpu::DeviceDescriptor {
            required_features: wgpu::Features::PASSTHROUGH_SHADERS,
            ..Default::default()
        },
    )
    .unwrap()
    .unwrap();
    let device = &created.device;
    let queue = &created.queue;
    let caps = created.capabilities;
    eprintln!("numeric adapter: {:?}; caps: {caps:?}", adapter.get_info());
    let cases = cases();
    let data: Vec<_> = cases
        .iter()
        .flatten()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("float32 oracle inputs"),
        contents: &data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: input.as_entire_binding(),
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    let vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            r#"
struct Out { @builtin(position) position: vec4<f32>, @location(0) @interpolate(flat) index: u32 }
@vertex fn main(@location(0) xy: vec2<f32>, @location(1) index: u32) -> Out {
    var o: Out; o.position = vec4<f32>(xy, 0.0, 1.0); o.index = index; return o;
}"#
            .into(),
        ),
    });
    // The portable interface packs Generic locations into vec4, while native
    // SPIR-V declares only the live scalar component.
    let vs_wgsl = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(r#"
struct Out { @builtin(position) position: vec4<f32>, @location(0) @interpolate(flat) index: vec4<u32> }
@vertex fn main(@location(0) xy: vec2<f32>, @location(1) index: u32) -> Out {
    var o: Out; o.position = vec4<f32>(xy, 0.0, 1.0); o.index = vec4<u32>(index, 0u, 0u, 0u); return o;
}"#.into()),
    });
    // One pixel rectangle per input; the flat integer is independent of float
    // interpolation and reaches the emitted shader without conversions.
    let mut vertices = Vec::new();
    for i in 0..SIDE * SIDE {
        let x = (i % SIDE) as f32 * 2.0 / SIDE as f32 - 1.0;
        let y = 1.0 - (i / SIDE) as f32 * 2.0 / SIDE as f32;
        let d = 2.0 / SIDE as f32;
        for [x, y] in [
            [x, y],
            [x + d, y],
            [x, y - d],
            [x, y - d],
            [x + d, y],
            [x + d, y - d],
        ] {
            vertices.extend(x.to_le_bytes());
            vertices.extend(y.to_le_bytes());
            vertices.extend(i.to_le_bytes());
        }
    }
    let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &vertices,
        usage: wgpu::BufferUsages::VERTEX,
    });
    for (operator, wgsl) in
        (0..9)
            .map(|op| (op, false))
            .chain([(5, true), (6, true), (7, true), (8, true)])
    {
        let conversion_cases;
        let cases = if operator >= 6 {
            conversion_cases = if operator < 8 {
                (0..65536_u32)
                    .map(|i| [i | ((65535 - i) << 16), 0, 0])
                    .collect::<Vec<_>>()
            } else {
                let mut values = cases
                    .iter()
                    .enumerate()
                    .map(|(i, v)| [(v[0] & 0x807f_ffff) | ((100 + i as u32 % 48) << 23), 0, 0])
                    .collect::<Vec<_>>();
                for (i, bits) in [
                    0,
                    0x8000_0000,
                    0x7f80_0000,
                    0xff80_0000,
                    0x7f80_0001,
                    0xffcf_ffff,
                    0x32ff_ffff,
                    0x3300_0000,
                    0x3300_0001,
                    0x337f_ffff,
                    0x3380_0000,
                    0x3380_0001,
                    0x387f_ffff,
                    0x3880_0000,
                    0x477f_e000,
                    0x477f_efff,
                    0x477f_f000,
                    0x477f_f001,
                    0x3f80_1000,
                    0x3f80_3000,
                ]
                .into_iter()
                .enumerate()
                {
                    values[i] = [bits, 0, 0];
                }
                values
            };
            &conversion_cases
        } else {
            &cases
        };
        let data: Vec<_> = cases
            .iter()
            .flatten()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        queue.write_buffer(&input, 0, &data);
        if operator == 2 && !caps.float32.fused_multiply_add {
            eprintln!(
                "SKIP FMA GPU oracle: shaderFmaFloat32 unavailable (addition/multiplication still checked)"
            );
            continue;
        }
        let ir = shader(operator);
        let mut float32 = caps.float32;
        // Exercise the non-preserving lowering even on hosts with preservation.
        float32.denorm_preserve = false;
        let module = lower_shader_ir_to_spirv(
            &ir,
            SpirvShaderOptions {
                depth_clip_negative_one_to_one: false,
                input_control_points: 0,
                tessellation_mode: None,
                float32,
                float64: caps.float64,
            },
        )
        .unwrap();
        validate(&module);
        // SAFETY: validated SPIR-V; numerical features enabled on the imported
        // device, descriptors/interfaces explicitly match this test pipeline.
        let fs = if wgsl {
            let module = lower_shader_ir_to_wgsl(&ir).unwrap();
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("WGSL absorbing-zero multiply oracle"),
                source: wgpu::ShaderSource::Wgsl(module.source().into()),
            })
        } else {
            unsafe {
                device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some("emitted arithmetic oracle"),
                    spirv: Some(std::borrow::Cow::Borrowed(module.words())),
                    entry_points: std::borrow::Cow::Borrowed(&[
                        wgpu::PassthroughShaderEntryPoint {
                            name: "main".into(),
                            workgroup_size: (0, 0, 0),
                        },
                    ]),
                    ..Default::default()
                })
            }
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: if wgsl { &vs_wgsl } else { &vs },
                entry_point: Some("main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Uint32],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let extent = wgpu::Extent3d {
            width: SIDE,
            height: SIDE,
            depth_or_array_layers: 1,
        };
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = output.create_view(&Default::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(SIDE * SIDE * 4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.draw(0..SIDE * SIDE * 6, 0..1);
        }
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIDE * 4),
                    rows_per_image: None,
                },
            },
            extent,
        );
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = readback.get_mapped_range(..).unwrap();
        for (i, values) in cases.iter().enumerate() {
            let mut inputs =
                ShaderEvaluationInputs::default().with_interface_bits(Loc::Generic(0), 0, 0);
            for (word, bits) in values.iter().enumerate() {
                inputs = inputs.with_constant_buffer_bits(0, word as u32 * 4, *bits);
            }
            let expected = evaluate_shader_ir(&ir, &inputs, 32)
                .unwrap()
                .output_bits(Loc::Color(0), 0)
                .unwrap();
            let actual = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            if operator == 5 && f32::from_bits(expected).is_nan() {
                assert!(
                    f32::from_bits(actual).is_nan(),
                    "NaN propagation case={i}, wgsl={wgsl}"
                );
                continue;
            }
            assert_eq!(
                actual, expected,
                "operator={operator}, wgsl={wgsl}, case={i}, operands={values:08x?}, GPU={actual:08x}, IR={expected:08x}"
            );
        }
        eprintln!(
            "operator {operator}, wgsl={wgsl}: {} numeric pixels passed",
            cases.len()
        );
        drop(bytes);
        readback.unmap();
    }
}
