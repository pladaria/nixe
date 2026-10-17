use super::*;

#[test]
fn proven_half_products_skip_repairs_until_float32_underflow_becomes_possible() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let block = b.entry();
    let address = constant(&mut b, block, 0);
    let bits = b.emit(
        block,
        pc(),
        Op::Load {
            space: MemorySpace::Storage,
            resource: Some(ResourceId::new(42)),
            address_type: Type::U32,
            ty: Type::U32,
        },
        vec![address],
    )[0];
    let input = b.emit(block, pc(), Op::UnpackHalf, vec![bits.into()])[0];
    let input = half_convert(&mut b, block, input, Type::F16, Type::F32);
    let mut value = input;
    for power in 1..=6 {
        if power > 1 {
            value = b.emit(
                block,
                pc(),
                Op::Binary {
                    op: Binary::Multiply,
                    ty: Type::F32,
                    float_control: nixe_gpu::ShaderFloatControl::PRECISE,
                },
                vec![value.into(), input.into()],
            )[0];
        }
        let bits = b.emit(
            block,
            pc(),
            Op::Bitcast {
                from: Type::F32,
                to: Type::U32,
            },
            vec![value.into()],
        )[0];
        let address = constant(&mut b, block, 12 + power * 4);
        b.emit(
            block,
            pc(),
            Op::Store {
                space: MemorySpace::Storage,
                resource: Some(ResourceId::new(42)),
                address_type: Type::U32,
                ty: Type::U32,
            },
            vec![address, bits.into()],
        );
    }
    b.terminate(block, pc(), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.workgroup_size = Some([1, 1, 1]);
    let mut opts = options(&device);
    opts.shader.float32.denorm_preserve = false;
    let module = spirv::emit(&f, opts).unwrap();
    let parsed = rspirv::dr::load_words(&module.words).unwrap();
    let calls = parsed.functions[0]
        .blocks
        .iter()
        .flat_map(|b| &b.instructions)
        .filter(|i| i.class.opcode == rspirv::spirv::Op::FunctionCall)
        .count();
    assert_eq!(calls, 1, "only the sixth power can be subnormal FP32");
    let mut kernel = Kernel::new(&device, &module, 10);
    let mut cases = vec![
        0, 1, 2, 0x3ff, 0x400, 0x3800, 0x3c00, 0x7bff, 0x7c00, 0x7e00, 0x8000, 0x8001, 0xfc00,
    ];
    let mut random = 0x15a82f91u32;
    for _ in 0..256 {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        cases.push(random as u16);
    }
    for half in cases {
        let input = expand_half(half);
        let mut words = [0; 10];
        words[0] = u32::from(half);
        let actual = kernel.execute(&mut device, &words);
        let mut expected = input;
        for power in 1..=6 {
            if power > 1 {
                expected *= input;
            }
            let result = f32::from_bits(actual[3 + power]);
            if expected.is_nan() {
                assert!(result.is_nan());
            } else {
                assert_eq!(
                    result.to_bits(),
                    expected.to_bits(),
                    "half={half:04x} power={power}"
                );
            }
        }
    }
    drop(kernel);
    drop(device);
    monitor.check().unwrap();
}

#[test]
fn native_half_conversions_match_exact_bits_exhaustively_and_at_rounding_boundaries() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let opts = options(&device);
    if !opts.shader.native_half_conversions {
        eprintln!("SKIP native half conversions: host lacks independent preserving FP16 controls");
        return;
    }
    let monitor = device.validation_monitor();
    let mut pack_inputs = vec![
        0, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fffffff, 0xffffffff,
    ];
    for bits in [0x477fefff, 0x477ff000, 0x477ff001, 0x7f7fffff, 0x7fffff] {
        pack_inputs.extend([bits, bits | 0x80000000]);
    }
    // Values on either side of every positive half rounding midpoint, and their
    // negative counterparts, exercise overflow, tiny ties and even/odd rounding.
    for half in 0..0x7bffu16 {
        let midpoint =
            ((f64::from(expand_half(half)) + f64::from(expand_half(half + 1))) * 0.5) as f32;
        for bits in [
            midpoint.to_bits() - 1,
            midpoint.to_bits(),
            midpoint.to_bits() + 1,
        ] {
            pack_inputs.extend([bits, bits | 0x80000000]);
        }
    }
    for payload in [1, 0x1000, 0x2000, 0x400000, 0x7fffff] {
        pack_inputs.extend([0x7f800000 | payload, 0xff800000 | payload]);
    }
    pack_inputs.resize(pack_inputs.len().next_multiple_of(128), 0);
    for unpack in [true, false] {
        let inputs: Vec<u32> = if unpack {
            (0..65536).collect()
        } else {
            pack_inputs.clone()
        };
        let count = inputs.len() as u32;
        let mut b = FunctionBuilder::new(ShaderStage::Compute);
        let block = b.entry();
        let lane = b.emit(
            block,
            pc(),
            Op::Builtin {
                builtin: nixe_gpu::ShaderComputeBuiltin::GlobalInvocationId,
                component: 0,
            },
            vec![],
        )[0];
        let four = constant(&mut b, block, 4);
        let address = b.emit(
            block,
            pc(),
            Op::Binary {
                op: Binary::Multiply,
                ty: Type::U32,
                float_control: nixe_gpu::ShaderFloatControl::PRECISE,
            },
            vec![lane.into(), four],
        )[0];
        let value = b.emit(
            block,
            pc(),
            Op::Load {
                space: MemorySpace::Storage,
                resource: Some(ResourceId::new(42)),
                address_type: Type::U32,
                ty: Type::U32,
            },
            vec![address.into()],
        )[0];
        let result = if unpack {
            let f = b.emit(block, pc(), Op::UnpackHalf, vec![value.into()])[0];
            let f = half_convert(&mut b, block, f, Type::F16, Type::F32);
            b.emit(
                block,
                pc(),
                Op::Bitcast {
                    from: Type::F32,
                    to: Type::U32,
                },
                vec![f.into()],
            )[0]
        } else {
            let f = b.emit(
                block,
                pc(),
                Op::Bitcast {
                    from: Type::U32,
                    to: Type::F32,
                },
                vec![value.into()],
            )[0];
            let f = half_convert(&mut b, block, f, Type::F32, Type::F16);
            b.emit(block, pc(), Op::PackHalf, vec![f.into()])[0]
        };
        let offset = constant(&mut b, block, count * 4);
        let output = b.emit(
            block,
            pc(),
            Op::Binary {
                op: Binary::Add,
                ty: Type::U32,
                float_control: nixe_gpu::ShaderFloatControl::PRECISE,
            },
            vec![address.into(), offset],
        )[0];
        b.emit(
            block,
            pc(),
            Op::Store {
                space: MemorySpace::Storage,
                resource: Some(ResourceId::new(42)),
                address_type: Type::U32,
                ty: Type::U32,
            },
            vec![output.into(), result.into()],
        );
        b.terminate(block, pc(), Terminator::Return);
        let mut function = b.finish().unwrap();
        function.workgroup_size = Some([128, 1, 1]);
        let mut input = inputs.clone();
        input.resize(inputs.len() * 2, 0);
        let module = spirv::emit(&function, opts).unwrap();
        let mut native = Kernel::new(&device, &module, input.len());
        let actual = native.execute_groups(&mut device, &input, count / 128);
        let mut exact_opts = opts;
        exact_opts.shader.native_half_conversions = false;
        let module = spirv::emit(&function, exact_opts).unwrap();
        let mut exact = Kernel::new(&device, &module, input.len());
        let expected = exact.execute_groups(&mut device, &input, count / 128);
        for (index, ((actual, expected), input)) in actual[inputs.len()..]
            .iter()
            .zip(&expected[inputs.len()..])
            .zip(&inputs)
            .enumerate()
        {
            assert_eq!(
                actual, expected,
                "unpack={unpack} index={index} input={input:08x}"
            );
        }
    }
    drop(device);
    monitor.check().unwrap();
}

#[test]
fn native_half_fma_matches_single_rounding_and_saturation() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let opts = options(&device);
    if !opts.shader.native_half_conversions || !opts.shader.native_half_fma {
        eprintln!("SKIP native half FMA: host lacks preserving FP16 or shaderFmaFloat16");
        return;
    }
    let monitor = device.validation_monitor();
    for saturate in [false, true] {
        let mut b = FunctionBuilder::new(ShaderStage::Compute);
        let block = b.entry();
        let mut args = Vec::new();
        for index in 0..3 {
            let address = constant(&mut b, block, index * 4);
            let bits = b.emit(
                block,
                pc(),
                Op::Load {
                    space: MemorySpace::Storage,
                    resource: Some(ResourceId::new(42)),
                    address_type: Type::U32,
                    ty: Type::U32,
                },
                vec![address],
            )[0];
            args.push(b.emit(block, pc(), Op::UnpackHalf, vec![bits.into()])[0].into());
        }
        let result = b.emit(block, pc(), Op::HalfFma { saturate }, args)[0];
        let result = b.emit(block, pc(), Op::PackHalf, vec![result.into()])[0];
        let output = constant(&mut b, block, 16);
        b.emit(
            block,
            pc(),
            Op::Store {
                space: MemorySpace::Storage,
                resource: Some(ResourceId::new(42)),
                address_type: Type::U32,
                ty: Type::U32,
            },
            vec![output, result.into()],
        );
        b.terminate(block, pc(), Terminator::Return);
        let mut function = b.finish().unwrap();
        function.workgroup_size = Some([1, 1, 1]);
        let module = spirv::emit(&function, opts).unwrap();
        let parsed = rspirv::dr::load_words(&module.words).unwrap();
        let f16 = parsed
            .types_global_values
            .iter()
            .find(|i| {
                i.class.opcode == rspirv::spirv::Op::TypeFloat
                    && i.operands == [rspirv::dr::Operand::LiteralBit32(16)]
            })
            .unwrap()
            .result_id;
        assert!(
            parsed.functions[0]
                .blocks
                .iter()
                .flat_map(|b| &b.instructions)
                .any(|i| i.class.opcode == rspirv::spirv::Op::FmaKHR && i.result_type == f16)
        );
        let mut kernel = Kernel::new(&device, &module, 5);
        let mut cases = vec![
            (0x3e00, 0x3c01, 0x8001),
            (0x3e00, 0x3c03, 1),
            (0x4b80, 0x6c44, 0x8001),
            (0x4b80, 0x6c44, 1),
            (1, 0x3800, 0),
            (1, 0x3800, 1),
            (0x8000, 0x3c00, 0x8000),
            (0x7c00, 0, 0),
            (0x7bff, 0x3c00, 0xfbff),
        ];
        let mut random = 0x4561234u32;
        for _ in 0..1024 {
            let mut next = || {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                random as u16
            };
            cases.push((next(), next(), next()));
        }
        for (a, b, c) in cases {
            let actual = kernel.execute(
                &mut device,
                &[u32::from(a), u32::from(b), u32::from(c), 0, 0],
            )[4];
            let [a, b, c] = [a, b, c].map(expand_half);
            let result = f64::from(a) * f64::from(b) + f64::from(c);
            let expected = if saturate && (result.is_nan() || result.is_sign_negative()) {
                0
            } else if saturate && result >= 1.0 {
                0x3c00
            } else {
                u32::from(half_reference::round_half(result))
            };
            if expected & 0x7fff > 0x7c00 {
                assert!(actual & 0x7fff > 0x7c00);
            } else {
                assert_eq!(
                    actual, expected,
                    "saturate={saturate}: {a:?} * {b:?} + {c:?}"
                );
            }
        }
    }
    drop(device);
    monitor.check().unwrap();
}

#[test]
fn typed_half_binary_matches_single_rounding_on_native_and_integer_hosts() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let native = options(&device);
    let mut pairs = vec![
        (0x8000, 0x8000),
        (0, 0x8000),
        (1, 1),
        (0x3ff, 1),
        (0x3c00, 0x1000),
        (0x7bff, 0x3c01),
        (0x7c00, 0),
        (0x7c01, 0x3c00),
        (0xfc01, 0x8000),
    ];
    let mut random = 0x9e3779b9u32;
    for _ in 0..128 {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        let a = random as u16;
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        pairs.push((a, random as u16));
    }
    for op in [Binary::Add, Binary::Multiply] {
        for saturate in [false, true] {
            let control = nixe_gpu::ShaderFloatControl::new(
                nixe_gpu::ShaderRoundingMode::NearestEven,
                nixe_gpu::ShaderNanMode::Propagate,
                false,
                false,
                saturate,
            );
            let function = scalar_kernel(
                Op::Binary {
                    op,
                    ty: Type::F16,
                    float_control: control,
                },
                &[Type::F16; 2],
                &[Type::F16],
            );
            for preserving_half in [false, native.shader.native_half_conversions] {
                let mut opts = native;
                opts.shader.native_half_conversions = preserving_half;
                let module = spirv::emit(&function, opts).unwrap();
                let mut kernel = Kernel::new(&device, &module, 5);
                for &(a, b) in &pairs {
                    let af = f64::from(expand_half(a));
                    let bf = f64::from(expand_half(b));
                    let value = if op == Binary::Add { af + bf } else { af * bf };
                    let value = if saturate {
                        if value.is_nan() || value.is_sign_negative() {
                            0.0
                        } else {
                            value.min(1.0)
                        }
                    } else {
                        value
                    };
                    let expected = u32::from(half_reference::round_half(value));
                    let actual =
                        kernel.execute(&mut device, &[u32::from(a), u32::from(b), 0, 0, 0])[4];
                    if expected & 0x7fff > 0x7c00 {
                        assert!(actual & 0x7fff > 0x7c00);
                    } else {
                        assert_eq!(
                            actual, expected,
                            "op={op:?} sat={saturate} native={preserving_half} a={a:04x} b={b:04x}"
                        );
                    }
                    assert_eq!(actual >> 16, 0);
                }
            }
        }
    }
    drop(device);
    monitor.check().unwrap();
}

#[test]
fn selected_native_float32_modes_match_exact_lowering_at_underflow_and_overflow() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let opts = options(&device);
    let values = [
        0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000, 0x00800001, 0x80800001, 0x3f7fffff,
        0x3f800000, 0x3f800001, 0xbf800001, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
        0x7fc12345,
    ];
    for (op, rounding) in [
        (Binary::Add, nixe_gpu::ShaderRoundingMode::NearestEven),
        (Binary::Multiply, nixe_gpu::ShaderRoundingMode::NearestEven),
        (Binary::Multiply, nixe_gpu::ShaderRoundingMode::TowardZero),
    ] {
        let control = nixe_gpu::ShaderFloatControl::new(
            rounding,
            nixe_gpu::ShaderNanMode::Propagate,
            true,
            true,
            false,
        );
        let f = scalar_kernel(
            Op::Binary {
                op,
                ty: Type::F32,
                float_control: control,
            },
            &[Type::F32; 2],
            &[Type::F32],
        );
        let native_module = spirv::emit(&f, opts).unwrap();
        let mut exact_opts = opts;
        exact_opts.shader.float32.denorm_flush_to_zero = false;
        exact_opts.shader.float32.rounding_mode_rtz = false;
        let exact_module = spirv::emit(&f, exact_opts).unwrap();
        let mut native = Kernel::new(&device, &native_module, 5);
        let mut exact = Kernel::new(&device, &exact_module, 5);
        for a in values {
            for b in values {
                let input = [a, b, 0, 0, 0];
                let actual = native.execute(&mut device, &input)[4];
                let expected = exact.execute(&mut device, &input)[4];
                if expected & 0x7fffffff > 0x7f800000 {
                    assert!(actual & 0x7fffffff > 0x7f800000);
                } else {
                    assert_eq!(actual, expected, "{op:?} {rounding:?}: {a:08x} {b:08x}");
                }
            }
        }
    }
    drop(device);
    monitor.check().unwrap();
}
