use nixe_shader_recompiler::{
    diagnostics::FailureCategory,
    environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
    ir::{Emitter, Ir, Opcode as IrOpcode, Reg, Value},
    maxwell::{
        bits::*,
        decode::{self, Opcode},
        instruction::Instruction,
        location::{FlowLocation, Location},
        operands::Operands,
        special_register::SpecialRegister,
    },
    types::{ByteOffset, CbufIndex, ProgramOffset, Stage},
};

fn location(offset: u32) -> Location {
    Location::new(ProgramOffset(offset)).unwrap()
}
fn id(value: Value) -> nixe_shader_recompiler::arena::InstId {
    let Value::Inst(id) = value else {
        panic!("Expected instruction")
    };
    id
}

#[test]
fn decoder_covers_every_prefix_and_preserves_first_match() {
    let mut entries = Vec::new();
    for row in include_str!("data/maxwell-encodings.txt").lines() {
        let (name, pattern) = row.split_once(' ').unwrap();
        assert_eq!(pattern.len(), 16);
        let (mut mask, mut value) = (0u64, 0u64);
        for (index, ch) in pattern.bytes().enumerate() {
            let bit = 1u64 << (63 - index);
            if ch != b'-' {
                mask |= bit;
            }
            if ch == b'1' {
                value |= bit;
            }
        }
        entries.push((name, mask, value));
    }
    assert_eq!(entries.len(), 280);
    for (encoding, &(name, mask, value)) in decode::ENCODINGS.iter().zip(&entries) {
        assert_eq!(
            (encoding.name, encoding.mask, encoding.value),
            (name, mask, value)
        );
        for word in [
            value,
            value | !mask,
            value | (!mask & 0xaaaa_aaaa_aaaa_aaaa),
        ] {
            let expected = entries
                .iter()
                .find(|(_, mask, value)| word & mask == *value)
                .unwrap()
                .0;
            assert_eq!(decode::decode(word).unwrap().encoding().name, expected);
        }
    }
    for prefix in 0..=u16::MAX {
        let word = u64::from(prefix) << 48 | 0x0000_1234_5678_9abc;
        let expected = entries
            .iter()
            .find(|(_, mask, value)| word & mask == *value)
            .map(|entry| entry.0);
        assert_eq!(
            decode::decode(word).ok().map(|op| op.encoding().name),
            expected,
            "prefix {prefix:#x}"
        );
    }
    assert_eq!(
        decode::decode(Opcode::B2R.encoding().value).unwrap(),
        Opcode::B2R
    );
    assert!(Opcode::B2R.encoding().unsupported);
    assert!(!Opcode::FADD_reg.encoding().unsupported);
    assert_eq!(
        decode::decode(0).unwrap_err().category,
        FailureCategory::UnknownInstruction
    );
}

#[test]
fn bundle_slots_and_synthetic_locations_are_distinct() {
    assert_eq!(location(0), location(8));
    assert_eq!(location(32), location(40));
    let slots: Vec<_> = (0..6).map(|n| location(0).advance(n).address().0).collect();
    assert_eq!(slots, [8, 16, 24, 40, 48, 56]);
    for offset in [8, 16, 24, 40, 48, 56, 0xffff_fff8] {
        let pc = location(offset);
        assert_eq!(pc.next().previous(), pc);
        assert_eq!(pc.previous().next(), pc);
    }
    assert_eq!(location(40).advance(-3), location(8));
    assert_eq!(location(24).branch_relative(-24).unwrap(), location(8));
    assert_eq!(location(24).branch_relative(0).unwrap(), location(40));
    assert!(Location::new(ProgramOffset(4)).is_err());
    let synthetic = FlowLocation::PredicateBefore(location(40));
    assert_eq!(synthetic.ordering_offset(), 36);
    assert!(synthetic.instruction().is_err());
    assert!(FlowLocation::Instruction(location(24)) < synthetic);
    assert!(synthetic < FlowLocation::Instruction(location(40)));
}

#[test]
fn field_extraction_and_immediates_preserve_sign_and_bits() {
    assert_eq!(extract::<0, 64>(u64::MAX), u64::MAX);
    assert_eq!(extract_signed::<0, 64>(u64::MAX), -1);
    assert_eq!(extract_signed::<20, 24>(0xff_fffe << 20), -2);
    assert_eq!(insert::<0, 64>(123, u64::MAX), u64::MAX);
    assert_eq!(insert::<63, 1>(u64::MAX, 0), i64::MAX as u64);
    let original = 0x1234_5678_9abc_def0;
    let changed = insert::<20, 14>(original, u64::MAX);
    assert_eq!(extract::<20, 14>(changed), 0x3fff);
    assert_eq!(changed & !(0x3fff << 20), original & !(0x3fff << 20));
    for bits in [0, 1, 0x3ffff, 0x7ffff] {
        for sign in [0u64, 1] {
            let word = Instruction((bits as u64) << 20 | sign << 56);
            assert_eq!(
                word.imm20(),
                if sign == 0 { bits } else { bits | 0xfff8_0000 }
            );
            assert_eq!(word.float_imm20_bits(), bits << 12 | (sign as u32) << 31);
            assert_eq!(word.double_imm20_bits(), (bits as u64) << 44 | sign << 63);
            assert_eq!(word.packed_imm20(), u64::from(word.imm20()) << 32);
        }
    }
    let word = Instruction((0xdead_beef_u64 << 20) | (0b1010 << 16));
    assert_eq!(word.imm32(), 0xdead_beef);
    assert_eq!(word.branch_absolute(), 0xdead_beef);
    assert_eq!(word.predicate().0.index(), 2);
    assert!(word.predicate().1);
    assert_eq!(Instruction::from_le_bytes(word.0.to_le_bytes()), word);
    assert_eq!(
        Instruction(17 << 20).special_register().unwrap(),
        SpecialRegister::SR_INVOCATION_ID
    );
    assert!(Instruction(255 << 20).special_register().is_err());
}

#[test]
fn cbuf_offsets_are_bytes_and_double_alignment_has_distinct_behavior() {
    for words in [2u64, 3] {
        let instruction = Instruction(3 << 34 | words << 20);
        assert_eq!(
            instruction.cbuf_address().unwrap(),
            (CbufIndex(3), ByteOffset(words as u32 * 4))
        );
        let mut ir = Ir::new().unwrap();
        let block = ir.new_block().unwrap();
        let mut emit = Emitter::new(&mut ir, block).unwrap();
        let result = Operands {
            emit: &mut emit,
            instruction,
        }
        .double_cbuf()
        .unwrap();
        assert_eq!(
            ir.inst(id(result)).unwrap().opcode(),
            IrOpcode::PackDouble2x32
        );
        let pair = ir.inst(id(result)).unwrap().arg(0).unwrap();
        let vector = ir.inst(id(pair)).unwrap();
        if words == 3 {
            assert_eq!(vector.arg(0).unwrap(), Value::U32(0));
        } else {
            let low = ir.inst(id(vector.arg(0).unwrap())).unwrap();
            assert_eq!(low.arg(1).unwrap(), Value::U32(8));
        }
        let high = ir.inst(id(vector.arg(1).unwrap())).unwrap();
        assert_eq!(high.arg(0).unwrap(), Value::U32(3));
        assert_eq!(high.arg(1).unwrap(), Value::U32(12));
        let mut emit = Emitter::new(&mut ir, block).unwrap();
        let packed = Operands {
            emit: &mut emit,
            instruction,
        }
        .packed_cbuf();
        assert_eq!(packed.is_ok(), words == 2);
    }
    assert!(Instruction(18 << 34).cbuf_address().is_err());
    assert_eq!(
        Instruction((17 << 34) | (0x3fff << 20))
            .cbuf_address()
            .unwrap(),
        (CbufIndex(17), ByteOffset(0xfffc))
    );
}

#[test]
fn paired_registers_preserve_order_and_alignment_restrictions() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let mut emit = Emitter::new(&mut ir, block).unwrap();
    let mut operands = Operands {
        emit: &mut emit,
        instruction: Instruction(0),
    };
    let pair = operands.read_u64(Reg(2)).unwrap();
    operands.write_pair(Reg(4), pair).unwrap();
    assert!(operands.read_f64(Reg(3)).is_err());
    assert!(operands.write_pair(Reg(3), pair).is_err());
    assert!(operands.read_u64(Reg(254)).is_err());
    assert!(operands.read_u64(Reg::RZ).is_ok());
    let writes: Vec<_> = ir
        .block(block)
        .unwrap()
        .instructions()
        .iter()
        .map(|&id| ir.inst(id).unwrap())
        .filter(|inst| inst.opcode() == IrOpcode::SetRegister)
        .collect();
    assert_eq!(writes.len(), 2);
    assert_eq!(writes[0].arg(0).unwrap(), Value::Reg(Reg(4)));
    assert_eq!(writes[1].arg(0).unwrap(), Value::Reg(Reg(5)));
}

#[test]
fn fetch_errors_carry_stage_pc_and_encoding() {
    let mut environment = SnapshotEnvironment::new(
        EnvironmentMetadata {
            stage: Stage::Compute,
            ..Default::default()
        },
        Observations {
            instructions: [(ProgramOffset(8), 0)].into(),
            ..Default::default()
        },
    );
    let error = decode::fetch(&mut environment, location(8)).unwrap_err();
    assert_eq!(error.stage, Some(Stage::Compute));
    assert_eq!(error.pc, Some(ProgramOffset(8)));
    assert_eq!(error.instruction, Some(0));
    let error = decode::fetch(&mut environment, location(16)).unwrap_err();
    assert_eq!(error.category, FailureCategory::MissingObservation);
    assert_eq!(error.pc, Some(ProgramOffset(16)));
    assert_eq!(error.instruction, None);
}
