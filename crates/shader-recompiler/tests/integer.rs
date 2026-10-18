use nixe_shader_recompiler::{
    arena::{BlockId, InstId},
    diagnostics::FailureCategory,
    ir::{Ir, Opcode as I, Program, Type, Value},
    maxwell::{
        bits::insert,
        decode::{DecodedInstruction, ENCODINGS, Opcode},
        instruction::Instruction,
        location::Location,
        translate::translate_instruction,
    },
    types::{ProgramOffset, Stage},
};
use std::collections::HashMap;

struct Machine {
    registers: [u32; 256],
    predicates: [bool; 7],
    flags: [bool; 4],
    values: HashMap<InstId, u64>,
}
impl Default for Machine {
    fn default() -> Self {
        Self {
            registers: [0; 256],
            predicates: [false; 7],
            flags: [false; 4],
            values: HashMap::new(),
        }
    }
}
impl Machine {
    fn value(&self, value: Value) -> u64 {
        match value {
            Value::U1(x) => x as u64,
            Value::U32(x) => x as u64,
            Value::U64(x) => x,
            Value::Reg(x) => x.0 as u64,
            Value::Pred(x) => x.index() as u64,
            Value::Inst(x) => self.values[&x],
            _ => panic!("Unexpected operand {value:?}"),
        }
    }
    fn run(&mut self, ir: &Ir, block: BlockId) {
        for &id in ir.block(block).unwrap().instructions() {
            let inst = ir.inst(id).unwrap();
            let mut a = [0u64; 5];
            for (index, value) in a.iter_mut().enumerate().take(inst.num_args()) {
                *value = self.value(inst.arg(index).unwrap());
            }
            let [x, y, z, w, _] = a;
            let sx = x as u32 as i32;
            let sy = y as u32 as i32;
            let mask = |n: u64| {
                if n >= 32 {
                    u32::MAX as u64
                } else {
                    (1u64 << n) - 1
                }
            };
            let result = match inst.opcode() {
                I::GetRegister => {
                    if x == 255 {
                        0
                    } else {
                        self.registers[x as usize] as u64
                    }
                }
                I::SetRegister => {
                    if x != 255 {
                        self.registers[x as usize] = y as u32;
                    }
                    0
                }
                I::GetPred => {
                    if x == 7 {
                        1
                    } else {
                        self.predicates[x as usize] as u64
                    }
                }
                I::SetPred => {
                    if x != 7 {
                        self.predicates[x as usize] = y != 0;
                    }
                    0
                }
                I::GetZFlag => self.flags[0] as u64,
                I::GetSFlag => self.flags[1] as u64,
                I::GetCFlag => self.flags[2] as u64,
                I::GetOFlag => self.flags[3] as u64,
                I::SetZFlag => {
                    self.flags[0] = x != 0;
                    0
                }
                I::SetSFlag => {
                    self.flags[1] = x != 0;
                    0
                }
                I::SetCFlag => {
                    self.flags[2] = x != 0;
                    0
                }
                I::SetOFlag => {
                    self.flags[3] = x != 0;
                    0
                }
                I::IAdd32 => x.wrapping_add(y),
                I::ISub32 => x.wrapping_sub(y),
                I::IMul32 => x.wrapping_mul(y),
                I::INeg32 => (x as u32).wrapping_neg() as u64,
                I::INeg64 => x.wrapping_neg(),
                I::IAbs32 => sx.wrapping_abs() as u32 as u64,
                I::BitwiseAnd32 | I::LogicalAnd => x & y,
                I::BitwiseOr32 | I::LogicalOr => x | y,
                I::BitwiseXor32 | I::LogicalXor => x ^ y,
                I::BitwiseNot32 => !x,
                I::LogicalNot => (x == 0) as u64,
                I::IEqual => (x == y) as u64,
                I::INotEqual => (x != y) as u64,
                I::SLessThan => (sx < sy) as u64,
                I::ULessThan => (x < y) as u64,
                I::SLessThanEqual => (sx <= sy) as u64,
                I::ULessThanEqual => (x <= y) as u64,
                I::SGreaterThan => (sx > sy) as u64,
                I::UGreaterThan => (x > y) as u64,
                I::SGreaterThanEqual => (sx >= sy) as u64,
                I::UGreaterThanEqual => (x >= y) as u64,
                I::SMin32 => sx.min(sy) as u32 as u64,
                I::UMin32 => x.min(y),
                I::SMax32 => sx.max(sy) as u32 as u64,
                I::UMax32 => x.max(y),
                I::SClamp32 => sx.clamp(sy, z as u32 as i32) as u32 as u64,
                I::UClamp32 => x.clamp(y, z),
                I::SelectU32 | I::SelectU1 => {
                    if x != 0 {
                        y
                    } else {
                        z
                    }
                }
                I::ShiftLeftLogical32 => (x as u32).wrapping_shl(y as u32) as u64,
                I::ShiftLeftLogical64 => x.wrapping_shl(y as u32),
                I::ShiftRightLogical32 => (x as u32).wrapping_shr(y as u32) as u64,
                I::ShiftRightLogical64 => x.wrapping_shr(y as u32),
                I::ShiftRightArithmetic32 => sx.wrapping_shr(y as u32) as u32 as u64,
                I::ShiftRightArithmetic64 => (x as i64).wrapping_shr(y as u32) as u64,
                I::BitFieldUExtract | I::BitFieldSExtract => {
                    let count = z.min(32u64.saturating_sub(y));
                    if count == 0 {
                        0
                    } else {
                        let value = (x >> y) & mask(count);
                        if inst.opcode() == I::BitFieldSExtract && value & (1 << (count - 1)) != 0 {
                            value | !mask(count)
                        } else {
                            value
                        }
                    }
                }
                I::BitFieldInsert => {
                    let count = w.min(32u64.saturating_sub(z));
                    if count == 0 {
                        x
                    } else {
                        (x & !(mask(count) << z)) | ((y & mask(count)) << z)
                    }
                }
                I::BitReverse32 => (x as u32).reverse_bits() as u64,
                I::BitCount32 => (x as u32).count_ones() as u64,
                I::FindUMsb32 | I::FindSMsb32 => {
                    let bits = if inst.opcode() == I::FindSMsb32 && sx < 0 {
                        !(x as u32)
                    } else {
                        x as u32
                    };
                    if bits == 0 {
                        u32::MAX as u64
                    } else {
                        31 - bits.leading_zeros() as u64
                    }
                }
                I::GetZeroFromOp => (x == 0) as u64,
                I::GetSignFromOp => (sx < 0) as u64,
                I::GetCarryFromOp | I::GetOverflowFromOp => {
                    let Value::Inst(producer) = inst.arg(0).unwrap() else {
                        panic!()
                    };
                    let producer = ir.inst(producer).unwrap();
                    assert_eq!(producer.opcode(), I::IAdd32);
                    let left = self.value(producer.arg(0).unwrap()) as u32;
                    let right = self.value(producer.arg(1).unwrap()) as u32;
                    if inst.opcode() == I::GetCarryFromOp {
                        left.overflowing_add(right).1 as u64
                    } else {
                        (left as i32).overflowing_add(right as i32).1 as u64
                    }
                }
                I::CompositeConstructU32x2 => x | (y << 32),
                I::CompositeExtractU32x2 => (x >> (32 * y)) & 0xffffffff,
                I::PackUint2x32 | I::UnpackUint2x32 => x,
                _ => panic!("Unsupported test operation {:?}", inst.opcode()),
            };
            self.values.insert(
                id,
                if inst.ty() == Type::U32 {
                    result & 0xffffffff
                } else {
                    result
                },
            );
        }
    }
}

fn word(op: Opcode) -> u64 {
    let c = match op {
        Opcode::IADD3_reg
        | Opcode::XMAD_reg
        | Opcode::LOP3_reg
        | Opcode::LEA_hi_reg
        | Opcode::SHF_l_reg
        | Opcode::SHF_r_reg
        | Opcode::BFI_reg
        | Opcode::ICMP_reg
        | Opcode::VMAD
        | Opcode::VMNMX => 3 << 39,
        _ => 0,
    };
    op.encoding().value | 7 << 16 | 1 << 8 | 2 << 20 | c
}
fn compile(opcode: Opcode, word: u64) -> nixe_shader_recompiler::Result<(Ir, BlockId)> {
    let mut ir = Ir::new()?;
    let block = ir.new_block()?;
    translate_instruction(
        &nixe_shader_recompiler::environment::EnvironmentMetadata::default(),
        &mut ir,
        block,
        DecodedInstruction {
            location: Location::new(ProgramOffset(8))?,
            instruction: Instruction(word),
            opcode,
        },
    )?;
    let mut program = Program::new(Stage::Compute);
    program.blocks.push(block);
    ir.verify(&program)?;
    Ok((ir, block))
}
fn execute(op: Opcode, word: u64, a: u32, b: u32, c: u32) -> Machine {
    let (ir, block) = compile(op, word).unwrap();
    let mut machine = Machine::default();
    machine.registers[1] = a;
    machine.registers[2] = b;
    machine.registers[3] = c;
    machine.run(&ir, block);
    machine
}

#[test]
fn add_wraparound_and_pseudo_flags_are_attached_to_the_producer() {
    for (a, b) in [
        (u32::MAX, 1),
        (0x7fffffff, 1),
        (0x80000000, 0x80000000),
        (0, 0),
        (1234, 5678),
    ] {
        let machine = execute(Opcode::IADD_reg, word(Opcode::IADD_reg) | 1 << 47, a, b, 0);
        let result = a.wrapping_add(b);
        assert_eq!(machine.registers[0], result);
        assert_eq!(
            machine.flags,
            [
                result == 0,
                (result as i32) < 0,
                a.overflowing_add(b).1,
                (a as i32).overflowing_add(b as i32).1
            ]
        );
    }
    let machine = execute(
        Opcode::IADD3_reg,
        insert::<37, 2>(word(Opcode::IADD3_reg), 1),
        u32::MAX,
        1,
        4,
    );
    assert_eq!(machine.registers[0], 0x10004);
    let machine = execute(
        Opcode::ISCADD_reg,
        insert::<39, 5>(word(Opcode::ISCADD_reg), 3) | 3 << 48 | 1 << 47,
        2,
        u32::MAX,
        0,
    );
    assert_eq!(machine.registers[0], 16);
    assert!(machine.flags[2]);
}

#[test]
fn lut_bit_order_is_exhaustive_and_does_not_add_per_bit_runtime_loops() {
    for lut in 0..=255 {
        let encoded = insert::<48, 3>(insert::<28, 8>(word(Opcode::LOP3_reg), lut), 7);
        let result = execute(
            Opcode::LOP3_reg,
            encoded,
            0xf0f0f0f0,
            0xcccccccc,
            0xaaaaaaaa,
        );
        assert_eq!(
            result.registers[0],
            (lut as u32) * 0x01010101,
            "table {lut}"
        );
        let (ir, block) = compile(Opcode::LOP3_reg, encoded).unwrap();
        assert!(ir.block(block).unwrap().instructions().len() <= 13);
    }
}

#[test]
fn shifts_and_empty_bitfields_preserve_boundary_rules() {
    for shift in [0, 1, 31, 32, 33, 63, 255] {
        for wrapped in [false, true] {
            let encoded = insert::<39, 1>(word(Opcode::SHL_reg), wrapped as u64);
            let result = execute(Opcode::SHL_reg, encoded, 0x89abcdef, shift, 0);
            let expected = if wrapped || shift < 32 {
                0x89abcdefu32.wrapping_shl(shift)
            } else {
                0
            };
            assert_eq!(result.registers[0], expected);
            let encoded = insert::<39, 1>(word(Opcode::SHR_reg), wrapped as u64) | 1 << 48;
            let result = execute(Opcode::SHR_reg, encoded, 0x89abcdef, shift, 0);
            let expected = if wrapped || shift < 32 {
                (0x89abcdefu32 as i32).wrapping_shr(shift) as u32
            } else {
                u32::MAX
            };
            assert_eq!(result.registers[0], expected);
        }
    }
    for (offset, count) in [(0, 0), (31, 0), (33, 0), (16, 8), (31, 1)] {
        let result = execute(
            Opcode::BFE_reg,
            word(Opcode::BFE_reg),
            0xdeadbeef,
            offset | count << 8,
            0,
        );
        let expected = if count == 0 {
            0
        } else {
            (0xdeadbeefu32 >> offset) & ((1 << count) - 1)
        };
        assert_eq!(result.registers[0], expected);
    }
}

#[test]
fn xmad_half_signedness_shift_merge_and_c_selection() {
    for signed_a in [false, true] {
        for signed_b in [false, true] {
            for upper_a in [false, true] {
                for upper_b in [false, true] {
                    for mode in [0, 1, 2, 4] {
                        for psl in [false, true] {
                            for merge in [false, true] {
                                let mut encoded = word(Opcode::XMAD_reg);
                                encoded = insert::<48, 1>(encoded, signed_a as u64);
                                encoded = insert::<49, 1>(encoded, signed_b as u64);
                                encoded = insert::<53, 1>(encoded, upper_a as u64);
                                encoded = insert::<35, 1>(encoded, upper_b as u64);
                                encoded = insert::<36, 1>(encoded, psl as u64);
                                encoded = insert::<37, 1>(encoded, merge as u64);
                                encoded = insert::<50, 3>(encoded, mode);
                                let (a, b, c) = (0x8765fedcu32, 0xabcd9234u32, 0x13579bdfu32);
                                let half = |v: u32, upper: bool, signed: bool| {
                                    let v = (v >> if upper { 16 } else { 0 }) as u16;
                                    if signed { v as i16 as u32 } else { v as u32 }
                                };
                                let mut product = half(a, upper_a, signed_a)
                                    .wrapping_mul(half(b, upper_b, signed_b));
                                if psl {
                                    product <<= 16;
                                }
                                let selected = match mode {
                                    0 => c,
                                    1 => c & 65535,
                                    2 => c >> 16,
                                    4 => (b << 16).wrapping_add(c),
                                    _ => unreachable!(),
                                };
                                let mut expected = product.wrapping_add(selected);
                                if merge {
                                    expected = (expected & 65535) | (b << 16);
                                }
                                assert_eq!(
                                    execute(Opcode::XMAD_reg, encoded, a, b, c).registers[0],
                                    expected
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn predicate_transfers_mask_selected_bytes_and_preserve_other_flags() {
    for cc in [false, true] {
        for selector in 0..4 {
            let encoded = insert::<20, 19>(
                insert::<41, 2>(insert::<40, 1>(word(Opcode::P2R_imm), cc as u64), selector),
                0x55,
            );
            let (ir, block) = compile(Opcode::P2R_imm, encoded).unwrap();
            let mut state = Machine::default();
            state.registers[1] = u32::MAX;
            state.predicates = [true, false, true, false, false, true, true];
            state.flags = [true, false, false, true];
            state.run(&ir, block);
            let packed = if cc { 1 } else { 0x45 };
            assert_eq!(
                state.registers[0],
                !(0x55 << (8 * selector)) | packed << (8 * selector)
            );
            let encoded = insert::<20, 19>(
                insert::<41, 2>(insert::<40, 1>(word(Opcode::R2P_imm), cc as u64), selector),
                5,
            );
            let (ir, block) = compile(Opcode::R2P_imm, encoded).unwrap();
            state.values.clear();
            state.registers[1] = 4 << (8 * selector);
            state.run(&ir, block);
            if cc {
                assert_eq!(state.flags, [false, false, true, true]);
            } else {
                assert_eq!(
                    state.predicates,
                    [false, false, true, false, false, true, true]
                );
            }
        }
    }
}

#[test]
fn integer_comparisons_cover_signed_unsigned_and_predicate_destinations() {
    for signed in [false, true] {
        for op in 0..8 {
            let encoded = insert::<0, 6>(
                insert::<39, 3>(
                    insert::<49, 3>(insert::<48, 1>(word(Opcode::ISETP_reg), signed as u64), op),
                    7,
                ),
                1,
            );
            let machine = execute(Opcode::ISETP_reg, encoded, u32::MAX, 1, 0);
            let expected = match op {
                0 => false,
                1 | 3 => signed,
                2 => false,
                4 | 6 => !signed,
                5 | 7 => true,
                _ => unreachable!(),
            };
            assert_eq!(&machine.predicates[..2], &[expected, !expected]);
        }
    }
}

#[test]
fn integer_conversion_saturates_and_lea_keeps_high_word_arithmetic() {
    let encoded = insert::<8, 2>(insert::<10, 2>(word(Opcode::I2I_reg), 2), 0) | 1 << 13 | 1 << 50;
    for (src, expected) in [(u32::MAX, 0), (512, 255), (128, 128)] {
        assert_eq!(
            execute(Opcode::I2I_reg, encoded, 0, src, 0).registers[0],
            expected
        );
    }
    let encoded = insert::<28, 5>(insert::<48, 3>(word(Opcode::LEA_hi_reg), 7), 4);
    assert_eq!(
        execute(Opcode::LEA_hi_reg, encoded, 0x80000000, 10, 3).registers[0],
        66
    );
    let encoded = insert::<39, 5>(insert::<48, 3>(word(Opcode::LEA_lo_reg), 7), 4) | 1 << 45;
    assert_eq!(
        execute(Opcode::LEA_lo_reg, encoded, 3, 10, 0).registers[0],
        10u32.wrapping_sub(48)
    );
}

#[test]
fn recognized_unsupported_forms_and_modifiers_fail_explicitly() {
    for encoding in ENCODINGS.iter().filter(|encoding| encoding.unsupported) {
        let error = compile(encoding.opcode, encoding.value).err().unwrap();
        assert_eq!(error.category, FailureCategory::Unsupported);
    }
    for (opcode, bits) in [
        (Opcode::IADD_reg, 1 << 50),
        (Opcode::IADD_reg, 1 << 43 | 1 << 47),
        (Opcode::XMAD_reg, 1 << 47),
        (Opcode::SHL_reg, 1 << 43),
        (Opcode::LOP3_reg, 1 << 47),
        (Opcode::LOP_reg, 1 << 43),
        (Opcode::IMNMX_reg, 1 << 43),
        (Opcode::SHF_r_reg, 1 << 48),
    ] {
        assert_eq!(
            compile(opcode, word(opcode) | bits).err().unwrap().category,
            FailureCategory::Unsupported
        );
    }
}

#[test]
fn extended_comparisons_consume_carry_and_zero_before_predicate_writes() {
    for signed in [false, true] {
        for carry in [false, true] {
            for zero in [false, true] {
                for (a, b) in [
                    (0u32, 0u32),
                    (u32::MAX, 1),
                    (1, u32::MAX),
                    (0x80000000, 0x7fffffff),
                ] {
                    let difference = a.wrapping_sub(b).wrapping_sub(u32::from(!carry));
                    let negative = (difference as i32) < 0;
                    let flip = !signed && ((a ^ b) & 0x80000000 != 0);
                    let less = if flip { !negative } else { negative };
                    let greater = if flip {
                        (difference as i32) <= 0
                    } else {
                        (difference as i32) > 0
                    };
                    let ge = if flip { negative } else { !negative };
                    let expected = [
                        false,
                        less,
                        difference == 0 && zero,
                        less || (difference == 0 && zero),
                        greater || (difference == 0 && !zero),
                        difference != 0 || !zero,
                        ge || (difference == 0 && zero),
                        true,
                    ];
                    for (op, expected) in expected.into_iter().enumerate() {
                        let encoded = insert::<0, 6>(
                            insert::<39, 3>(
                                insert::<49, 3>(
                                    insert::<48, 1>(word(Opcode::ISETP_reg), signed as u64),
                                    op as u64,
                                ),
                                7,
                            ),
                            1,
                        ) | 1 << 43;
                        let (ir, block) = compile(Opcode::ISETP_reg, encoded).unwrap();
                        let mut machine = Machine::default();
                        machine.registers[1] = a;
                        machine.registers[2] = b;
                        machine.flags = [zero, false, carry, false];
                        machine.run(&ir, block);
                        assert_eq!(machine.predicates[0], expected);
                        assert_eq!(machine.predicates[1], !expected);
                    }
                }
            }
        }
    }
}

#[test]
fn narrow_video_and_special_register_routing_preserve_supported_modes() {
    let encoded = insert::<37, 2>(insert::<29, 2>(word(Opcode::VMAD) | 1 << 50, 0), 0);
    assert_eq!(
        execute(Opcode::VMAD, encoded, 0x1234, 0x4567, 10).registers[0],
        0x34 * 0x67 + 10
    );
    let encoded = insert::<51, 3>(
        insert::<37, 2>(insert::<29, 2>(word(Opcode::VMNMX) | 1 << 50, 3), 3),
        6,
    );
    assert_eq!(execute(Opcode::VMNMX, encoded, 17, 23, 19).registers[0], 19);
    for (selector, expected) in [
        (0, I::LaneId),
        (17, I::InvocationId),
        (29, I::InvocationInfo),
        (30, I::SR_WScaleFactorXY),
        (31, I::SR_WScaleFactorZ),
        (56, I::SubgroupEqMask),
        (60, I::SubgroupGeMask),
    ] {
        let (ir, block) =
            compile(Opcode::S2R, insert::<20, 8>(word(Opcode::S2R), selector)).unwrap();
        assert!(
            ir.block(block)
                .unwrap()
                .instructions()
                .iter()
                .any(|&id| ir.inst(id).unwrap().opcode() == expected)
        );
    }
    for selector in [1, 28, 99, 255] {
        assert_eq!(
            execute(
                Opcode::S2R,
                insert::<20, 8>(word(Opcode::S2R), selector),
                0,
                0,
                0
            )
            .registers[0],
            0
        );
    }
}

#[test]
fn movement_forms_keep_masks_rz_and_raw_immediates() {
    let encoded = insert::<12, 4>(insert::<20, 32>(word(Opcode::MOV32I), 0xdeadbeef), 15);
    assert_eq!(
        execute(Opcode::MOV32I, encoded, 0, 0, 0).registers[0],
        0xdeadbeef
    );
    let encoded = insert::<39, 4>(word(Opcode::MOV_reg), 2);
    let (ir, block) = compile(Opcode::MOV_reg, encoded).unwrap();
    assert!(
        !ir.block(block)
            .unwrap()
            .instructions()
            .iter()
            .any(|&id| ir.inst(id).unwrap().opcode() == I::SetRegister)
    );
    let encoded = insert::<20, 8>(insert::<39, 4>(word(Opcode::MOV_reg), 15), 255);
    assert_eq!(execute(Opcode::MOV_reg, encoded, 0, 0, 0).registers[0], 0);
}
