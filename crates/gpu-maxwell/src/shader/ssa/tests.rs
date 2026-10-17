use super::super::test_support::mapped_memory;
use super::*;
use nixe_gpu::{
    ShaderEvaluationInputs, ShaderIntegerComparison, ShaderIoLocation, ShaderStage,
    VerifiedShaderIr, evaluate_shader_ir,
};

fn pc(index: usize) -> u32 {
    (index / 3 * 32 + index % 3 * 8 + 8) as u32
}
fn words(instructions: &[u64]) -> Vec<u64> {
    instructions
        .chunks(3)
        .flat_map(|chunk| {
            std::iter::once(0).chain((0..3).map(move |i| chunk.get(i).copied().unwrap_or(0)))
        })
        .collect()
}
fn mov(destination: u8, bits: u32) -> u64 {
    0x0100_0000_0007_f000 | u64::from(bits) << 20 | u64::from(destination)
}
fn bra(index: usize, target: usize, predicate: u8, inverted: bool) -> u64 {
    0xe240_0000_0000_000f
        | (((i64::from(pc(target)) - i64::from(pc(index)) - 8) as u64) & 0xff_ffff) << 20
        | u64::from(predicate) << 16
        | u64::from(inverted) << 19
}
const EXIT: u64 = 0xe300_0000_0007_000f;

#[test]
fn static_local_spills_share_words_and_promote_branch_and_predicated_writes_to_phis() {
    for branch in [false, true] {
        let mut header = pixel_header();
        header[1] = 64;
        let mut code = vec![
            0x4c98_0788_0007_0002, // R2 = c[2][0], unknown during compilation
            0x3660_0000_0077_0207 | (2 << 49) | (7 << 39), // P0 = R2 == 7
            mov(8, 1_f32.to_bits()),
            mov(9, 2_f32.to_bits()),
            0xef55_0000_0207_ff08, // STL.64 [RZ+32], R8:R9
            mov(8, 3_f32.to_bits()),
        ];
        if branch {
            code.push(bra(6, 8, 0, false));
            code.push(0xef54_0000_0247_ff08); // Replace the second word only.
        } else {
            code.push(0xef54_0000_0240_ff08); // @P0 STL.32 [RZ+36], R8
        }
        code.extend([0xef44_0000_0247_ff00, EXIT]); // LDL.32 R0, [RZ+36]
        let (function, _) = fixture(MaxwellShaderStage::Pixel, Some(header), &code, 16);
        function.verify().unwrap();
        assert!(
            function
                .blocks
                .iter()
                .flat_map(|b| &b.phis)
                .any(|p| function.values[p.result.index()].ty == Type::U32)
        );
        assert!(
            !function
                .blocks
                .iter()
                .flat_map(|b| &b.instructions)
                .any(|i| matches!(
                    i.op,
                    Op::Load {
                        space: MemorySpace::Local,
                        ..
                    } | Op::Store {
                        space: MemorySpace::Local,
                        ..
                    }
                ))
        );
        for initial in [0, 7] {
            let result = output(&function, &BTreeMap::from([((2, 0), initial)]));
            let expected = if (initial == 7) ^ branch {
                3_f32
            } else {
                2_f32
            };
            assert_eq!(
                result[&(ShaderIoLocation::Color(0), 0)],
                u64::from(expected.to_bits())
            );
        }
    }
}

#[test]
fn local_spills_do_not_initialize_undefined_words_or_accept_unproved_addresses() {
    let mut header = pixel_header();
    header[1] = 64;
    let (function, binary) = fixture(
        MaxwellShaderStage::Pixel,
        Some(header),
        &[0xef44_0000_0207_ff00, EXIT],
        16,
    );
    assert!(
        function
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .any(|i| matches!(i.op, Op::Undefined(Type::U32)))
    );
    for encoding in [
        0xef55_0000_0407_ff08, // Beyond SPH allocation.
        0xef55_0000_0207_0108, // Dynamic address.
        0xef55_0000_0207_ff09, // Misaligned vector register.
        0xef52_0000_0207_ff08, // Subword access needs exact merging.
    ] {
        assert!(
            super::super::local_memory::LocalMemory::default()
                .decode(&binary, 8, encoding, 16, &mut 16)
                .is_err()
        );
    }
}

#[test]
fn indexed_local_loads_alias_static_vector_stores_and_freeze_overwritten_base_registers() {
    let mut header = pixel_header();
    header[1] = 64;
    header[3] = 32;
    header[18] = 3;
    let (function, _) = fixture(
        MaxwellShaderStage::Pixel,
        Some(header),
        &[
            0x4c98_0788_0007_0000,
            mov(8, 1_f32.to_bits()),
            mov(9, 2_f32.to_bits()),
            0xef55_0000_0207_ff08,
            mov(10, 3_f32.to_bits()),
            mov(11, 4_f32.to_bits()),
            0xef55_0000_0287_ff0a,
            0xef45_0000_0007_0000,
            EXIT,
        ],
        16,
    );
    assert_eq!(function.local_memory_bytes, 96);
    function.verify().unwrap();
    for (address, pair) in [(32, [1_f32, 2_f32]), (40, [3_f32, 4_f32])] {
        let result = output(&function, &BTreeMap::from([((2, 0), address)]));
        for (component, value) in pair.into_iter().enumerate() {
            assert_eq!(
                result[&(ShaderIoLocation::Color(0), component as u32)],
                u64::from(value.to_bits())
            );
        }
    }
}

#[test]
fn range_reduction_crosses_a_dominated_branch_join_with_unchanged_data() {
    for initial in [0, 7] {
        let instructions = [
            mov(0, initial),
            0x3660_0000_0077_0007 | (2 << 49) | (7 << 39), // P0 = R0 == 7
            mov(28, 0.5f32.to_bits()),
            0x5c90_0000_01c7_0022, // RRO.SINCOS R34, R28
            bra(4, 6, 0, false),
            mov(28, 9f32.to_bits()), // original input can change after RRO
            0x5080_0000_0017_2200,   // MUFU.SIN R0, R34
            EXIT,
        ];
        let (function, _) = fixture(
            MaxwellShaderStage::Pixel,
            Some(pixel_header()),
            &instructions,
            40,
        );
        let values = output(&function, &BTreeMap::new());
        let value = f32::from_bits(values[&(ShaderIoLocation::Color(0), 0)] as u32);
        assert!((value - 0.5f32.sin()).abs() < 1e-6);
    }
}

#[test]
fn range_reduced_bits_may_be_preserved_and_discarded_but_never_observed() {
    let lop = |op: u64, dst: u8, a: u8, b: u8| {
        0x5c47_0000_0007_0000
            | (op << 41)
            | u64::from(dst)
            | (u64::from(a) << 8)
            | (u64::from(b) << 20)
    };
    let code = [
        mov(13, 0.5_f32.to_bits()),
        mov(1, 0xffff0000),
        mov(2, 0x1234),
        0x5c90_0080_00d7_000d,
        0x5080_0000_0027_0d05,
        lop(0, 13, 13, 1),
        lop(1, 13, 13, 2),
        mov(1, 0xffff),
        lop(0, 0, 13, 1),
        EXIT,
    ];
    let (function, mut binary) =
        fixture(MaxwellShaderStage::Pixel, Some(pixel_header()), &code, 20);
    assert_eq!(
        output(&function, &BTreeMap::new())[&(ShaderIoLocation::Color(0), 0)],
        0x1234
    );
    assert!(
        !function
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .any(|i| matches!(i.op, Op::Undefined(_)))
    );
    // A full-width read would expose the still-unrepresented upper bits.
    binary.bundles[8 / 3].instructions[8 % 3] = 0x5c98_0780_00d7_0000;
    assert!(matches!(
        lower_binary(&binary, 20),
        Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            detail: "RRO reduced representation has observable bits outside a compatible MUFU",
            ..
        })
    ));
}

#[test]
fn packed_half_writes_can_replace_both_halves_of_an_exp2_reduction_register() {
    let code = [
        mov(13, 0.5_f32.to_bits()),
        mov(17, 1_f32.to_bits()),
        0x5c90_0080_00d7_000d,
        0x5080_0000_0027_0d05,
        0x7a04_883c_0f07_110d,
        0x7a87_0804_00e7_ff0d,
        0x5c98_0780_00d7_0000,
        EXIT,
    ];
    let (function, _) = fixture(MaxwellShaderStage::Pixel, Some(pixel_header()), &code, 24);
    function.verify().unwrap();
    assert!(
        !function
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .any(|i| matches!(i.op, Op::Undefined(_)))
    );
}

#[test]
fn opposite_predicates_can_reuse_a_reduction_register_for_ordinary_math() {
    let code = [
        0x4c98_0788_0007_0002,
        0x3660_0000_0077_0207 | (2 << 49) | (7 << 39),
        mov(3, 0.5_f32.to_bits()),
        mov(14, 4_f32.to_bits()),
        0x5c90_0080_0038_000e,                 // @!P0 RRO.EX2 R14,R3
        0x5080_0000_0028_0e00,                 // @!P0 MUFU.EX2 R0,R14
        mov(14, 4_f32.to_bits()) & !(7 << 16), // @P0 overwrite the ordinary value
        0x5080_0000_0030_0e00,                 // @P0 MUFU.LG2 R0,R14
        EXIT,
    ];
    let (function, mut binary) =
        fixture(MaxwellShaderStage::Pixel, Some(pixel_header()), &code, 20);
    for initial in [0, 7] {
        let result = output(&function, &BTreeMap::from([((2, 0), initial)]));
        let actual = f32::from_bits(result[&(ShaderIoLocation::Color(0), 0)] as u32);
        let expected = if initial == 7 { 2.0 } else { 0.5_f32.exp2() };
        assert!((actual - expected).abs() < 1e-6);
    }
    // An unguarded LG2 would really consume the reduced EX2 representation.
    binary.bundles[7 / 3].instructions[7 % 3] |= 7 << 16;
    assert!(matches!(
        lower_binary(&binary, 20),
        Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            detail: "RRO reduced representation has observable bits outside a compatible MUFU",
            ..
        })
    ));
}

fn pixel_header() -> [u32; 20] {
    let mut h = [0; 20];
    h[0] = 0x0002_5462;
    h[4] = 0x000f_f000;
    h[6] = 0x0000_0077;
    h[13] = 0x0007_f000;
    h[18] = 1;
    h
}

fn fixture(
    stage: MaxwellShaderStage,
    header: Option<[u32; 20]>,
    instructions: &[u64],
    register_count: u8,
) -> (Function, MaxwellShaderBinary) {
    let (program, binary) = program_fixture(stage, header, instructions, register_count);
    (std::sync::Arc::unwrap_or_clone(program.function), binary)
}

fn program_fixture(
    stage: MaxwellShaderStage,
    header: Option<[u32; 20]>,
    instructions: &[u64],
    register_count: u8,
) -> (MaxwellSsaProgram, MaxwellShaderBinary) {
    let (allocation, address_space, address) = mapped_memory();
    let mut bytes: Vec<u8> = header
        .map(|h| h.into_iter().flat_map(u32::to_le_bytes).collect())
        .unwrap_or_default();
    bytes.extend(
        words(instructions)
            .iter()
            .flat_map(|word| word.to_le_bytes()),
    );
    allocation.write(0, &bytes).unwrap();
    let writes = CanonicalWriteBatch::new();
    let request = MaxwellSsaRequest {
        stage,
        address,
        register_count,
        workgroup_size: (stage == MaxwellShaderStage::Compute).then_some([32, 1, 1]),
    };
    let program = compile_maxwell_shader_ssa(&address_space, &writes, request).unwrap();
    let memory = MaxwellShaderMemoryView::new(&address_space, &writes);
    let binary = if stage == MaxwellShaderStage::Compute {
        read_shader_code(
            &memory,
            stage,
            address,
            MaxwellShaderMetadata::Compute {
                register_count,
                workgroup_size: [32, 1, 1],
            },
            None,
            0,
        )
        .unwrap()
    } else {
        read_shader_binary(&memory, stage, address).unwrap()
    };
    program.function.verify().unwrap();
    (program, binary)
}

#[test]
fn constant_buffer_bounds_cover_all_static_words_and_keep_dynamic_bindings_declared() {
    let mut header = pixel_header();
    header[18] = 15;
    // LDC.128 c[1][0x110] consumes all four words through byte 0x11f.
    let (program, _) = program_fixture(
        MaxwellShaderStage::Pixel,
        Some(header),
        &[0xef96_0010_1107_ff00, EXIT],
        4,
    );
    let binding = program
        .bindings
        .iter()
        .find(|b| b.source == MaxwellSsaBindingSource::ConstantBuffer { slot: 1 })
        .unwrap();
    assert_eq!(
        program.constant_buffer_extents()[&binding.resource],
        Some(0x120)
    );

    header[18] = 7;
    // R0 comes from c[2][16], then indexes c[1]. The later static c[1][16]
    // access cannot narrow that dynamic binding. c[2] still needs only 20 bytes.
    let (program, _) = program_fixture(
        MaxwellShaderStage::Pixel,
        Some(header),
        &[
            0x4c98_0788_0047_0000,
            0xef95_0010_0007_0000,
            0x4c98_0784_0047_0002,
            EXIT,
        ],
        3,
    );
    let extents = program.constant_buffer_extents();
    for binding in &program.bindings {
        match binding.source {
            MaxwellSsaBindingSource::ConstantBuffer { slot: 1 } => {
                assert_eq!(extents[&binding.resource], None);
            }
            MaxwellSsaBindingSource::ConstantBuffer { slot: 2 } => {
                assert_eq!(extents[&binding.resource], Some(20));
            }
            other => panic!("unexpected fixture binding: {other:?}"),
        }
    }
    assert_eq!(program.bindings.len(), 2);
}

// Independent bounded fixture execution of typed SSA; not compiled into the
// frontend or renderer. Effects and phi snapshots follow the CFG edges.
fn output(
    function: &Function,
    constants: &BTreeMap<(u32, u32), u32>,
) -> BTreeMap<(ShaderIoLocation, u32), u64> {
    let mut values = vec![None; function.values.len()];
    let mut local = vec![None; function.local_memory_bytes as usize / 4];
    let mut result = BTreeMap::new();
    let mut block = function.entry();
    let mut previous = None;
    for _ in 0..500 {
        let b = &function.blocks[block.index()];
        let phis: Vec<_> = b
            .phis
            .iter()
            .map(|p| {
                (
                    p.result,
                    values[p
                        .incoming
                        .iter()
                        .find(|(pred, _)| Some(*pred) == previous)
                        .unwrap()
                        .1
                        .index()],
                )
            })
            .collect();
        for (id, bits) in phis {
            values[id.index()] = bits;
        }
        for inst in &b.instructions {
            let a: Vec<_> = inst
                .arguments
                .iter()
                .map(|v| {
                    values[v.index()]
                        .unwrap_or_else(|| panic!("undefined SSA read at {:?}", inst.source))
                })
                .collect();
            let bits = match inst.op {
                Op::Constant(c) => Some(c.bits),
                Op::Undefined(_) => None,
                Op::EffectStart => Some(0),
                Op::Bitcast { .. } | Op::Copy(_) => Some(a[0]),
                Op::Unary {
                    op: Unary::Not,
                    ty: Type::Bool,
                } => Some(u64::from(a[0] == 0)),
                Op::Binary { op, ty, .. } => Some(
                    match op {
                        Binary::Add => a[0].wrapping_add(a[1]),
                        Binary::Multiply => a[0].wrapping_mul(a[1]),
                        Binary::And => a[0] & a[1],
                        Binary::Or => a[0] | a[1],
                        Binary::Xor => a[0] ^ a[1],
                        Binary::ShiftLeft => a[0] << a[1],
                        _ => panic!("fixture operation"),
                    } & if ty == Type::U32 {
                        u64::from(u32::MAX)
                    } else {
                        u64::MAX
                    },
                ),
                Op::IntegerCompare { ty, comparison } => {
                    let order = if ty == Type::I32 {
                        (a[0] as i32).cmp(&(a[1] as i32))
                    } else {
                        a[0].cmp(&a[1])
                    };
                    Some(u64::from(match comparison {
                        ShaderIntegerComparison::Less => order.is_lt(),
                        ShaderIntegerComparison::Equal => order.is_eq(),
                        ShaderIntegerComparison::NotEqual => !order.is_eq(),
                        _ => panic!("fixture comparison"),
                    }))
                }
                Op::Math {
                    function:
                        nixe_gpu::ssa::MathFunction::Special(nixe_gpu::ShaderSpecialFunction::Sine),
                    ..
                } => Some(u64::from(f32::from_bits(a[0] as u32).sin().to_bits())),
                Op::Math {
                    function:
                        nixe_gpu::ssa::MathFunction::Special(nixe_gpu::ShaderSpecialFunction::Exp2),
                    ..
                } => Some(u64::from(f32::from_bits(a[0] as u32).exp2().to_bits())),
                Op::Math {
                    function:
                        nixe_gpu::ssa::MathFunction::Special(nixe_gpu::ShaderSpecialFunction::Log2),
                    ..
                } => Some(u64::from(f32::from_bits(a[0] as u32).log2().to_bits())),
                Op::Output {
                    location,
                    component,
                    ..
                } => {
                    result.insert((location, component), a[0]);
                    None
                }
                Op::Load {
                    space: MemorySpace::Local,
                    ..
                } => Some(local[a[0] as usize / 4].expect("uninitialized local fixture word")),
                Op::Store {
                    space: MemorySpace::Local,
                    ..
                } => {
                    local[a[0] as usize / 4] = Some(a[1]);
                    None
                }
                Op::Load {
                    space: MemorySpace::Constant,
                    resource: Some(binding),
                    ..
                } => Some(u64::from(constants[&(binding.index() as u32, a[0] as u32)])),
                _ => panic!("unsupported fixture operation {:?}", inst.op),
            };
            if inst.op.ordered() {
                if inst.results.len() > 1 {
                    values[inst.results[0].index()] = bits;
                }
                values[inst.results.last().unwrap().index()] = Some(a.last().unwrap() + 1);
            } else {
                values[inst.results[0].index()] = bits;
            }
        }
        previous = Some(block);
        block = match b.terminator {
            Terminator::Jump(to) => to,
            Terminator::Branch {
                condition,
                when_true,
                when_false,
            } => {
                if values[condition.index()].unwrap() != 0 {
                    when_true
                } else {
                    when_false
                }
            }
            Terminator::Return | Terminator::Discard => return result,
            _ => panic!("fixture terminator"),
        };
    }
    panic!("fixture exceeded step limit")
}

fn compare_oracle(
    function: &Function,
    binary: &MaxwellShaderBinary,
    register_count: u8,
    constants: &BTreeMap<(u32, u32), u32>,
) {
    let ir = super::super::translate::translate_shader_binary(
        binary,
        register_count,
        &BTreeMap::new(),
        false,
    )
    .unwrap()
    .ir;
    let ir = VerifiedShaderIr::verify(ir).unwrap();
    let mut inputs = ShaderEvaluationInputs::default();
    for (&(binding, offset), &bits) in constants {
        inputs = inputs.with_constant_buffer_bits(binding as u8, offset, bits);
    }
    let oracle = evaluate_shader_ir(&ir, &inputs, 500).unwrap();
    for ((location, component), bits) in output(function, constants) {
        assert_eq!(
            oracle.output_bits(location, component as u8),
            Some(bits as u32)
        );
    }
}

#[test]
fn actual_snapshot_redefinitions_and_fragment_exports_match_oracle() {
    let (function, binary) = fixture(
        MaxwellShaderStage::Pixel,
        Some(pixel_header()),
        &[mov(0, 7), mov(0, 11), EXIT],
        1,
    );
    assert_eq!(
        output(&function, &BTreeMap::new())[&(ShaderIoLocation::Color(0), 0)],
        11
    );
    compare_oracle(&function, &binary, 1, &BTreeMap::new());
    assert!(!function.blocks.iter().any(|b| {
        b.instructions
            .iter()
            .any(|i| matches!(i.op, Op::Undefined(_)))
    }));
}

#[test]
fn predication_joins_and_guest_defined_rz_pt_values_match_oracle() {
    // PSETP P0, PT, PT, PT, !PT: P0=false; a predicated MOV must retain R0.
    let psetp = 0x5090_0000_0007_0007 | (7 << 12) | (7 << 29) | (15 << 39);
    let instructions = [mov(0, 17), psetp, mov(0, 29) & !(7 << 16), EXIT];
    let (function, binary) = fixture(
        MaxwellShaderStage::Pixel,
        Some(pixel_header()),
        &instructions,
        1,
    );
    assert_eq!(
        output(&function, &BTreeMap::new())[&(ShaderIoLocation::Color(0), 0)],
        17
    );
    compare_oracle(&function, &binary, 1, &BTreeMap::new());
    // MOV R0,RZ uses the architectural zero, independently of SSA live-ins.
    let rz = 0x5c98_0000_0007_0000 | (15 << 39) | (255 << 20);
    let (function, binary) = fixture(
        MaxwellShaderStage::Pixel,
        Some(pixel_header()),
        &[rz, EXIT],
        1,
    );
    compare_oracle(&function, &binary, 1, &BTreeMap::new());
    assert_eq!(
        output(&function, &BTreeMap::new())[&(ShaderIoLocation::Color(0), 0)],
        0
    );
}

#[test]
fn loop_carried_redefinitions_from_sass_match_oracle() {
    // R0=0; do { R0+=1; P0=R0<5; } while(P0); export R0.
    let add = 0x3810_0000_0017_0000;
    let isetp = 0x3660_0000_0057_0007 | (1 << 49) | (7 << 39);
    let instructions = [mov(0, 0), add, isetp, bra(3, 1, 0, false), EXIT];
    let (function, binary) = fixture(
        MaxwellShaderStage::Pixel,
        Some(pixel_header()),
        &instructions,
        1,
    );
    assert_eq!(
        output(&function, &BTreeMap::new())[&(ShaderIoLocation::Color(0), 0)],
        5
    );
    assert!(function.blocks.iter().any(|b| !b.phis.is_empty()));
    compare_oracle(&function, &binary, 1, &BTreeMap::new());
}

#[test]
fn executed_ssy_sync_edges_lower_directly_without_control_stack_ir() {
    let instructions = [
        0xe290_0000_0387_000f,
        bra(1, 3, 7, false),
        0xf0f8_0000_0007_000f,
        mov(0, 31),
        0xf0f8_0000_0007_000f,
        mov(0, 99),
        mov(0, 43),
        mov(0, 47),
        EXIT,
    ];
    let (function, binary) = fixture(
        MaxwellShaderStage::Pixel,
        Some(pixel_header()),
        &instructions,
        1,
    );
    compare_oracle(&function, &binary, 1, &BTreeMap::new());
    assert_eq!(
        output(&function, &BTreeMap::new())[&(ShaderIoLocation::Color(0), 0)],
        47
    );
    assert!(function.dump().contains("0x58"));
}

#[test]
fn predicated_exit_exports_only_taken_values() {
    let isetp = 0x3660_0000_0077_0007 | (2 << 49) | (7 << 39); // R0==7
    let instructions = [mov(0, 7), isetp, EXIT & !(7 << 16), mov(0, 19), EXIT];
    let (function, binary) = fixture(
        MaxwellShaderStage::Pixel,
        Some(pixel_header()),
        &instructions,
        1,
    );
    compare_oracle(&function, &binary, 1, &BTreeMap::new());
    assert_eq!(
        output(&function, &BTreeMap::new())[&(ShaderIoLocation::Color(0), 0)],
        7
    );
}

#[test]
fn indexed_multiword_load_snapshots_an_aliased_address_and_orders_loads() {
    let ldc = 0xef95_0000_0007_0000 | (1 << 36); // B64 c[1][R0], writes R0/R1
    let mut header = pixel_header();
    header[18] = 3;
    let (function, binary) = fixture(
        MaxwellShaderStage::Pixel,
        Some(header),
        &[mov(0, 17), ldc, EXIT],
        2,
    );
    let constants = BTreeMap::from([((1, 16), 0x1234_5678), ((1, 20), 0x89ab_cdef)]);
    compare_oracle(&function, &binary, 2, &constants);
    let result = output(&function, &constants);
    assert_eq!(result[&(ShaderIoLocation::Color(0), 0)], 0x1234_5678);
    assert_eq!(result[&(ShaderIoLocation::Color(0), 1)], 0x89ab_cdef);
    assert_eq!(
        function
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .filter(|i| matches!(i.op, Op::Load { .. }))
            .count(),
        2
    );
}

#[test]
fn compute_geometry_and_undefined_live_ins_are_preserved() {
    let (function, _) = fixture(MaxwellShaderStage::Compute, None, &[EXIT], 0);
    assert_eq!(function.stage, ShaderStage::Compute);
    assert_eq!(function.workgroup_size, Some([32, 1, 1]));
    let (function, _) = fixture(MaxwellShaderStage::Pixel, Some(pixel_header()), &[EXIT], 1);
    assert!(
        function
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .any(|i| i.op == Op::Undefined(Type::U32))
    );
}

#[test]
fn unsupported_scalar_reports_guest_pc() {
    let (allocation, space, address) = mapped_memory();
    let code = words(&[0xdead_beef_1237_5678, EXIT]);
    allocation
        .write(
            0,
            &code
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let error = compile_maxwell_shader_ssa(
        &space,
        &CanonicalWriteBatch::new(),
        MaxwellSsaRequest {
            stage: MaxwellShaderStage::Compute,
            address,
            register_count: 1,
            workgroup_size: Some([1, 1, 1]),
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        MaxwellShaderTranslationError::UnsupportedInstruction {
            instruction_offset: 8,
            encoding: 0xdead_beef_1237_5678,
            ..
        }
    ));
}

#[test]
fn fragment_discard_has_no_lexical_fallthrough_or_exports() {
    let mut header = pixel_header();
    header[0] |= 1 << 15;
    let (function, _) = fixture(
        MaxwellShaderStage::Pixel,
        Some(header),
        &[0xe330_0000_0007_000f, 0xdead_beef_1234_5678, EXIT],
        1,
    );
    assert!(
        function
            .blocks
            .iter()
            .any(|b| b.terminator == Terminator::Discard)
    );
    assert!(
        !function
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .any(|i| matches!(i.op, Op::Output { .. }))
    );
}

#[test]
fn path_dependent_reconvergence_requires_explicit_cfg_cloning() {
    let token = |index, target| {
        0xe290_0000_0000_0000
            | (((i64::from(pc(target)) - i64::from(pc(index)) - 8) as u64) & 0xff_ffff) << 20
    };
    let instructions = [
        bra(0, 3, 0, false),
        token(1, 6),
        bra(2, 5, 7, false),
        token(3, 7),
        bra(4, 5, 7, false),
        0xf0f8_0000_0007_000f,
        EXIT,
        EXIT,
    ];
    let (allocation, space, address) = mapped_memory();
    allocation
        .write(
            0,
            &words(&instructions)
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let error = compile_maxwell_shader_ssa(
        &space,
        &CanonicalWriteBatch::new(),
        MaxwellSsaRequest {
            stage: MaxwellShaderStage::Compute,
            address,
            register_count: 0,
            workgroup_size: Some([1, 1, 1]),
        },
    )
    .unwrap_err();
    assert!(
        matches!(error, MaxwellShaderTranslationError::MalformedInstruction { instruction_offset, reason: "path-dependent control-stack branch requires CFG cloning", .. } if instruction_offset == pc(5))
    );
}

#[test]
fn nested_break_sync_and_continue_use_only_executed_tokens() {
    let token = |opcode: u16, index, target| {
        (u64::from(opcode) << 48)
            | (((i64::from(pc(target)) - i64::from(pc(index)) - 8) as u64) & 0xff_ffff) << 20
    };
    for instructions in [
        vec![
            token(0xe2a0, 0, 8),
            token(0xe290, 1, 6),
            token(0xe2a0, 2, 5),
            0xe340_0000_0007_000f,
            bra(4, 3, 7, false),
            0xf0f8_0000_0007_000f,
            0xe340_0000_0007_000f,
            mov(0, 99),
            mov(0, 47),
            EXIT,
        ],
        vec![
            token(0xe2a0, 0, 6),
            token(0xe2b0, 1, 3),
            0xe350_0000_0007_000f,
            mov(0, 47),
            0xe340_0000_0007_000f,
            mov(0, 99),
            EXIT,
        ],
    ] {
        let (function, binary) = fixture(
            MaxwellShaderStage::Pixel,
            Some(pixel_header()),
            &instructions,
            1,
        );
        compare_oracle(&function, &binary, 1, &BTreeMap::new());
        assert_eq!(
            output(&function, &BTreeMap::new())[&(ShaderIoLocation::Color(0), 0)],
            47
        );
    }
}

#[test]
fn predicated_multi_result_opcode_snapshots_guard_and_all_predicate_inputs() {
    let initialize = 0x5090_0000_0007_0001 | (7 << 12) | (7 << 29) | (7 << 39); // P0=true, P1=false
    let replace = 0x5090_0000_0000_0001 | (1 << 15) | (7 << 29) | (7 << 39); // @P0 P0=!P0, P1=P0
    let instructions = [
        mov(0, 17),
        initialize,
        replace,
        (mov(0, 47) & !(7 << 16)) | (1 << 16),
        EXIT,
    ];
    let (function, binary) = fixture(
        MaxwellShaderStage::Pixel,
        Some(pixel_header()),
        &instructions,
        1,
    );
    compare_oracle(&function, &binary, 1, &BTreeMap::new());
    assert_eq!(
        output(&function, &BTreeMap::new())[&(ShaderIoLocation::Color(0), 0)],
        47
    );
}
#[test]
fn dual_vertex_attributes_are_values_and_returns_enter_b() {
    use nixe_gpu::ShaderInterfaceElement as E;
    let mut a = FunctionBuilder::new(ShaderStage::Vertex);
    let entry = a.entry();
    let bits = a.emit(
        entry,
        ShaderSourceLocation::new(8),
        Op::Constant(Constant::u32(11.0f32.to_bits())),
        vec![],
    )[0];
    a.emit(
        entry,
        ShaderSourceLocation::new(16),
        Op::Output {
            location: ShaderIoLocation::Generic(1),
            component: 0,
            ty: Type::U32,
        },
        vec![bits.into()],
    );
    a.terminate(entry, ShaderSourceLocation::new(24), Terminator::Return);
    let mut a = a.finish().unwrap();
    a.outputs.push(
        E::new(
            ShaderIoLocation::Generic(1),
            0,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap(),
    );
    let mut b = FunctionBuilder::new(ShaderStage::Vertex);
    let entry = b.entry();
    let value = b.emit(
        entry,
        ShaderSourceLocation::new(8),
        Op::Input {
            location: ShaderIoLocation::Generic(1),
            component: 0,
            ty: Type::U32,
        },
        vec![],
    )[0];
    b.emit(
        entry,
        ShaderSourceLocation::new(16),
        Op::Output {
            location: ShaderIoLocation::Position,
            component: 0,
            ty: Type::U32,
        },
        vec![value.into()],
    );
    b.terminate(entry, ShaderSourceLocation::new(24), Terminator::Return);
    let mut b = b.finish().unwrap();
    b.inputs.push(
        E::new(
            ShaderIoLocation::Generic(1),
            0,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap(),
    );
    b.outputs.push(
        E::new(
            ShaderIoLocation::Position,
            0,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap(),
    );
    let merged = super::vertex::merge(&a, &b).unwrap();
    assert!(merged.inputs.is_empty());
    assert_eq!(
        output(&merged, &BTreeMap::new())[&(ShaderIoLocation::Position, 0)],
        u64::from(11.0f32.to_bits())
    );
    assert_eq!(
        merged
            .blocks
            .iter()
            .filter(|b| b.terminator == Terminator::Return)
            .count(),
        1
    );
    let module = nixe_gpu::ssa::spirv::emit(&merged, spirv_options()).unwrap();
    let parsed = rspirv::dr::load_words(&module.words).unwrap();
    assert!(
        !parsed
            .all_inst_iter()
            .any(|i| i.class.opcode == rspirv::spirv::Op::Variable
                && i.operands.contains(&rspirv::dr::Operand::StorageClass(
                    rspirv::spirv::StorageClass::Function
                )))
    );
}
fn spirv_options() -> nixe_gpu::ssa::spirv::Options {
    nixe_gpu::ssa::spirv::Options {
        shader: nixe_gpu::SpirvShaderOptions {
            depth_clip_negative_one_to_one: false,
            vertex_position_transform: None,
            input_control_points: 0,
            tessellation_mode: None,
            native_half_conversions: false,
            native_half_fma: false,
            float32: nixe_gpu::SpirvFloat32Capabilities {
                denorm_preserve: true,
                rounding_mode_rte: true,
                signed_zero_inf_nan_preserve: true,
                fused_multiply_add: true,
                ..Default::default()
            },
            float64: Default::default(),
        },
        integer64: false,
        subgroup_quad: false,
        shared_memory_bytes: 0,
        uniform_buffer_bytes: 65536,
    }
}
#[test]
fn frontend_temporaries_are_not_limited_by_the_oracle_register_width() {
    let mut next = 65536;
    let register = super::super::decode::allocate_shader_temporary(
        MaxwellShaderStage::Compute,
        8,
        0,
        "temporary index overflow",
        &mut next,
    )
    .unwrap();
    assert_eq!(register.index(), 65536);
    assert_eq!(next, 65537);
}
#[test]
fn captured_graphics_pipeline_compiles_to_linked_ssa_without_oracle_modules() {
    use super::super::test_support::program_three_d;
    use crate::{MaxwellChannelId, MaxwellChannelOwner, MaxwellGpuChannel, SWITCH_1_GM20B_PROFILE};
    let (allocation, space, address) = mapped_memory();
    let mut header = [0u32; 20];
    header[0] = 0x0002_0461;
    let bytes: Vec<_> = header
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .chain(words(&[EXIT]).into_iter().flat_map(u64::to_le_bytes))
        .collect();
    allocation.write(0, &bytes).unwrap();
    let mut fragment = pixel_header();
    fragment[6] = 2;
    let code = [0xe0c3_ff88_0077_ff00, EXIT];
    let bytes: Vec<_> = fragment
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .chain(words(&code).into_iter().flat_map(u64::to_le_bytes))
        .collect();
    allocation.write(0x100, &bytes).unwrap();
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    program_three_d(
        &mut channel,
        0,
        SWITCH_1_GM20B_PROFILE.classes().three_d().0,
    );
    for (method, value) in [
        (0x2140, 0x50),
        (0x1608, (address >> 32) as u32),
        (0x160c, address as u32),
        (0x2000, 0x11),
        (0x2004, 0),
        (0x200c, 4),
        (0x2040, 0x51),
        (0x2044, 0x100),
        (0x204c, 4),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let pipeline =
        compile_maxwell_pipeline_ssa(channel.three_d(), &space, &CanonicalWriteBatch::new())
            .unwrap();
    assert_eq!(pipeline.programs.len(), 2);
    let fragment = &pipeline
        .programs
        .iter()
        .find(|p| p.function.stage == ShaderStage::Fragment)
        .unwrap()
        .function;
    assert!(fragment.inputs.is_empty());
    assert_eq!(
        output(fragment, &BTreeMap::new())[&(ShaderIoLocation::Color(0), 0)],
        0
    );
    let stages: Vec<_> = pipeline
        .programs
        .iter()
        .map(|p| (p.function.as_ref(), spirv_options()))
        .collect();
    let modules = nixe_gpu::ssa::spirv::emit_pipeline(&stages).unwrap();
    assert_eq!(modules.len(), 2);
    assert!(pipeline.source_is_current(&space));
}

#[test]
fn native_shader_cache_uses_content_identity_and_fresh_write_dependencies() {
    use super::super::test_support::{canonical_shader_writes, program_three_d};
    use crate::{MaxwellChannelId, MaxwellChannelOwner, MaxwellGpuChannel, SWITCH_1_GM20B_PROFILE};
    use nixe_gpu::ssa::compile::CompilePool;
    use std::{sync::Arc, time::Duration};
    let (allocation, space, address) = mapped_memory();
    let mut header = [0u32; 20];
    header[0] = 0x0002_0461;
    let vertex: Vec<_> = header
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .chain(words(&[EXIT]).into_iter().flat_map(u64::to_le_bytes))
        .collect();
    allocation.write(0, &vertex).unwrap();
    let fragment: Vec<_> = pixel_header()
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .chain(
            words(&[mov(0, 1f32.to_bits()), EXIT])
                .into_iter()
                .flat_map(u64::to_le_bytes),
        )
        .collect();
    allocation.write(0x100, &fragment).unwrap();
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    program_three_d(
        &mut channel,
        0,
        SWITCH_1_GM20B_PROFILE.classes().three_d().0,
    );
    for (method, value) in [
        (0x2140, 0x50),
        (0x1608, (address >> 32) as u32),
        (0x160c, address as u32),
        (0x2000, 0x11),
        (0x2004, 0),
        (0x200c, 4),
        (0x2040, 0x51),
        (0x2044, 0x100),
        (0x204c, 4),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let snapshot =
        MaxwellShaderSnapshot::capture(channel.three_d(), &space, &CanonicalWriteBatch::new())
            .unwrap();
    let identity = snapshot.identity();
    let pool = Arc::new(CompilePool::new(2, 8).unwrap());
    let compiler = MaxwellShaderCompiler::new(pool, 16, 1024 * 1024);
    let code = compiler
        .request(snapshot.clone())
        .unwrap()
        .wait_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    eprintln!(
        "NATIVE_MAXWELL_COMPILE shader=graphics_pair translation_ns={} cache={:?}",
        code.translation_ns,
        compiler.usage()
    );
    let again = compiler
        .request(snapshot.clone())
        .unwrap()
        .wait_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&code, &again));
    // A staged code write changes identity before publication; unrelated writes
    // do not. Current mapping/CPU proofs remain separate from compiled code.
    let encoding = mov(0, 2f32.to_bits());
    let staged = canonical_shader_writes(
        &space,
        &[
            (address + 0x158, encoding as u32),
            (address + 0x15c, (encoding >> 32) as u32),
        ],
    );
    assert!(!snapshot.staged_writes_are_irrelevant(&staged));
    assert!(
        snapshot.staged_writes_are_irrelevant(&canonical_shader_writes(
            &space,
            &[(address + 0x800, 123)]
        ))
    );
    let changed = MaxwellShaderSnapshot::capture(channel.three_d(), &space, &staged).unwrap();
    assert_ne!(changed.identity(), identity);
    let two = compiler
        .request(changed)
        .unwrap()
        .wait_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let f = &two
        .programs
        .iter()
        .find(|p| p.function.stage == ShaderStage::Fragment)
        .unwrap()
        .function;
    assert_eq!(
        output(f, &BTreeMap::new())[&(ShaderIoLocation::Color(0), 0)],
        2f32.to_bits() as u64
    );
    allocation.write(0x200, &vertex).unwrap();
    allocation.write(0x300, &fragment).unwrap();
    program_three_d(&mut channel, 0x2004, 0x200);
    program_three_d(&mut channel, 0x2044, 0x300);
    let relocated =
        MaxwellShaderSnapshot::capture(channel.three_d(), &space, &CanonicalWriteBatch::new())
            .unwrap();
    assert_eq!(relocated.identity(), identity);
    let reused = compiler
        .request(relocated.clone())
        .unwrap()
        .wait_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&code, &reused));
    assert!(relocated.source_is_current(&space));
    let program = reused
        .programs
        .iter()
        .find(|p| p.stage == MaxwellShaderStage::Pixel)
        .unwrap();
    assert_eq!(
        relocated
            .locate_source(program, ShaderSourceLocation::new(88))
            .unwrap()
            .1,
        address + 0x300
    );
    allocation.write(0x358, &encoding.to_le_bytes()).unwrap();
    assert!(!relocated.source_is_current(&space));
    assert_ne!(
        MaxwellShaderSnapshot::capture(channel.three_d(), &space, &CanonicalWriteBatch::new())
            .unwrap()
            .identity(),
        identity
    );
}
