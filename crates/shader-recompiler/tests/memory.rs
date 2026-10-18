use nixe_shader_recompiler::{
    arena::BlockId,
    diagnostics::FailureCategory,
    environment::EnvironmentMetadata,
    ir::{
        Attribute, FmzMode, FpControl, FpRounding, Instruction as IrInstruction, Ir, Opcode as I,
        Program, Reg, Value,
    },
    maxwell::{
        bits::insert,
        decode::{DecodedInstruction, Opcode},
        header::ProgramHeader,
        instruction::Instruction,
        location::Location,
        translate::translate_instruction,
    },
    types::{ProgramOffset, Stage},
};
fn word(op: Opcode) -> u64 {
    op.encoding().value | 7 << 16 | 2 << 8 | 4 << 20
}
fn compile(
    opcode: Opcode,
    word: u64,
    metadata: &EnvironmentMetadata,
) -> nixe_shader_recompiler::Result<(Ir, BlockId)> {
    let mut ir = Ir::new()?;
    let block = ir.new_block()?;
    translate_instruction(
        metadata,
        &mut ir,
        block,
        DecodedInstruction {
            location: Location::new(ProgramOffset(8))?,
            instruction: Instruction(word),
            opcode,
        },
    )?;
    let mut program = Program::new(metadata.stage);
    program.blocks.push(block);
    ir.verify(&program)?;
    Ok((ir, block))
}
fn ops(ir: &Ir, block: BlockId, opcode: I) -> Vec<&IrInstruction> {
    ir.block(block)
        .unwrap()
        .instructions()
        .iter()
        .filter_map(|&id| {
            let inst = ir.inst(id).unwrap();
            (inst.opcode() == opcode).then_some(inst)
        })
        .collect()
}
fn reads(ir: &Ir, block: BlockId) -> Vec<Value> {
    ops(ir, block, I::GetRegister)
        .iter()
        .map(|inst| inst.arg(0).unwrap())
        .collect()
}
fn meta() -> EnvironmentMetadata {
    EnvironmentMetadata {
        local_memory_size: 32,
        ..Default::default()
    }
}
#[test]
fn narrow_spaces_keep_signedness_and_byte_or_word_addressing() {
    for (size, shared, global, cbuf) in [
        (0, I::LoadSharedU8, I::LoadGlobalU8, I::GetCbufU8),
        (1, I::LoadSharedS8, I::LoadGlobalS8, I::GetCbufS8),
        (2, I::LoadSharedU16, I::LoadGlobalU16, I::GetCbufU16),
        (3, I::LoadSharedS16, I::LoadGlobalS16, I::GetCbufS16),
    ] {
        for (op, expected) in [
            (Opcode::LDS, shared),
            (Opcode::LDG, global),
            (Opcode::LDC, cbuf),
        ] {
            let w = insert::<48, 3>(word(op), size);
            let (ir, b) = compile(op, w, &meta()).unwrap();
            assert_eq!(ops(&ir, b, expected).len(), 1);
            assert!(ops(&ir, b, I::LoadLocal).is_empty());
        }
        let w = insert::<20, 24>(
            insert::<8, 8>(insert::<48, 3>(word(Opcode::LDL), size), 255),
            7,
        );
        let (ir, b) = compile(Opcode::LDL, w, &meta()).unwrap();
        assert_eq!(ops(&ir, b, I::LoadLocal)[0].arg(0).unwrap(), Value::U32(1));
        let extract = if size % 2 == 1 {
            I::BitFieldSExtract
        } else {
            I::BitFieldUExtract
        };
        assert_eq!(
            ops(&ir, b, extract)[0].arg(2).unwrap(),
            Value::U32(if size < 2 { 8 } else { 16 })
        );
        assert_eq!(ops(&ir, b, I::SelectU32).len(), 1);
    }
}
#[test]
fn partial_local_stores_merge_old_word_and_immediate_out_of_bounds_drops() {
    let w = insert::<20, 24>(insert::<8, 8>(word(Opcode::STL), 255), 3);
    let (ir, b) = compile(Opcode::STL, w, &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::LoadLocal).len(), 1);
    let merge = ops(&ir, b, I::BitFieldInsert)[0];
    assert_eq!(merge.arg(3).unwrap(), Value::U32(8));
    assert_eq!(ops(&ir, b, I::WriteLocal).len(), 1);
    let (ir, b) = compile(Opcode::STL, insert::<20, 24>(w, 32), &meta()).unwrap();
    assert!(ir.block(b).unwrap().instructions().is_empty());
    let (ir, b) = compile(Opcode::STL, insert::<8, 8>(w, 2), &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::ShiftRightArithmetic32).len(), 1);
    assert_eq!(ops(&ir, b, I::WriteLocal).len(), 1);
}
#[test]
fn vector_layout_and_shared_store_alignment_are_distinct() {
    for op in [Opcode::LDL, Opcode::LDS, Opcode::LDG] {
        let w = insert::<48, 3>(word(op), 6);
        let (ir, b) = compile(op, w, &meta()).unwrap();
        assert_eq!(ops(&ir, b, I::SetRegister).len(), 4);
        assert!(compile(op, insert::<0, 8>(w, 2), &meta()).is_err());
    }
    let w = insert::<0, 8>(insert::<48, 3>(word(Opcode::STS), 6), 2);
    let (ir, b) = compile(Opcode::STS, w, &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::WriteSharedU128).len(), 1);
    assert_eq!(
        reads(&ir, b),
        vec![
            Value::Reg(Reg(2)),
            Value::Reg(Reg(2)),
            Value::Reg(Reg(3)),
            Value::Reg(Reg(4)),
            Value::Reg(Reg(5))
        ]
    );
    assert!(compile(Opcode::STG, w, &meta()).is_err());
    let (ir, b) = compile(Opcode::LDG, insert::<48, 3>(word(Opcode::LDG), 7), &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::LoadGlobal128).len(), 1);
    let (ir, b) = compile(Opcode::STG, insert::<48, 3>(word(Opcode::STG), 7), &meta()).unwrap();
    assert!(
        !ir.block(b)
            .unwrap()
            .instructions()
            .iter()
            .any(|&id| ir.inst(id).unwrap().may_have_side_effects())
    );
}
#[test]
fn global_addresses_sign_extend_relative_offsets_and_pair_extended_sources() {
    let w = insert::<20, 24>(insert::<48, 3>(word(Opcode::LDG), 4), 0xffffff);
    let (ir, b) = compile(Opcode::LDG, w, &meta()).unwrap();
    assert_eq!(
        ops(&ir, b, I::IAdd64)[0].arg(1).unwrap(),
        Value::U64(u64::MAX)
    );
    let (ir, b) = compile(Opcode::LDG, insert::<8, 8>(w, 255), &meta()).unwrap();
    assert_eq!(
        ops(&ir, b, I::IAdd64)[0].arg(1).unwrap(),
        Value::U64(0xffffff)
    );
    let (ir, b) = compile(Opcode::LDG, w | 1 << 45, &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::PackUint2x32).len(), 1);
    assert_eq!(reads(&ir, b), vec![Value::Reg(Reg(2)), Value::Reg(Reg(3))]);
    assert!(compile(Opcode::LDG, insert::<8, 8>(w | 1 << 45, 3), &meta()).is_err());
}
#[test]
fn segmented_constant_reads_keep_both_address_fields_and_pair_results() {
    let w = insert::<44, 2>(insert::<48, 3>(word(Opcode::LDC), 5), 2);
    let (ir, b) = compile(Opcode::LDC, w, &meta()).unwrap();
    let extracts = ops(&ir, b, I::BitFieldUExtract);
    assert_eq!(extracts.len(), 2);
    assert_eq!(extracts[0].arg(1).unwrap(), Value::U32(16));
    assert_eq!(extracts[1].arg(1).unwrap(), Value::U32(0));
    assert_eq!(ops(&ir, b, I::GetCbufU32x2).len(), 1);
    assert_eq!(ops(&ir, b, I::SetRegister).len(), 2);
    assert!(compile(Opcode::LDC, insert::<44, 2>(w, 1), &meta()).is_err());
}
#[test]
fn atomics_preserve_effects_results_and_load_only_cases() {
    for (size, op, expected) in [
        (0, 0, I::GlobalAtomicIAdd32),
        (1, 1, I::GlobalAtomicSMin32),
        (2, 2, I::GlobalAtomicUMax64),
        (5, 1, I::GlobalAtomicSMin64),
        (3, 0, I::GlobalAtomicAddF32),
        (4, 2, I::GlobalAtomicMaxF16x2),
    ] {
        let w = insert::<52, 4>(insert::<49, 3>(word(Opcode::ATOM), size), op);
        let (ir, b) = compile(Opcode::ATOM, w, &meta()).unwrap();
        assert_eq!(ops(&ir, b, expected).len(), 1);
        if size == 3 {
            let control = FpControl::from_bits(ops(&ir, b, expected)[0].flags).unwrap();
            assert_eq!(control.rounding, FpRounding::RN);
            assert_eq!(control.fmz_mode, FmzMode::FTZ);
        }
    }
    for (size, op, load) in [
        (0, 3, I::LoadGlobal32),
        (2, 4, I::LoadGlobal64),
        (5, 0, I::LoadGlobal64),
        (4, 8, I::LoadGlobal32),
    ] {
        let w = insert::<52, 4>(insert::<49, 3>(word(Opcode::ATOM), size), op);
        let (ir, b) = compile(Opcode::ATOM, w, &meta()).unwrap();
        assert_eq!(ops(&ir, b, load).len(), 1);
    }
    let w = insert::<52, 4>(insert::<49, 3>(word(Opcode::ATOM), 3), 1);
    assert_eq!(
        compile(Opcode::ATOM, w, &meta()).err().unwrap().category,
        FailureCategory::InvalidArgument
    );
    let w = insert::<23, 3>(insert::<20, 3>(word(Opcode::RED), 0), 0);
    let (ir, b) = compile(Opcode::RED, w, &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::GlobalAtomicIAdd32).len(), 1);
    assert_eq!(
        ops(&ir, b, I::SetRegister)[0].arg(0).unwrap(),
        Value::Reg(Reg::RZ)
    );
    assert!(compile(Opcode::ATOM_cas, word(Opcode::ATOM_cas), &meta()).is_err());
}
#[test]
fn shared_atomic_exchange_supports_pairs_and_keeps_byte_offset() {
    let w = insert::<52, 4>(
        insert::<28, 2>(insert::<30, 22>(word(Opcode::ATOMS), 0x3fffff), 2),
        8,
    );
    let (ir, b) = compile(Opcode::ATOMS, w, &meta()).unwrap();
    assert_eq!(
        ops(&ir, b, I::IAdd32)[0].arg(1).unwrap(),
        Value::U32(u32::MAX - 3)
    );
    assert_eq!(ops(&ir, b, I::SharedAtomicExchange64).len(), 1);
    assert_eq!(ops(&ir, b, I::SetRegister).len(), 2);
    assert!(compile(Opcode::ATOMS, insert::<52, 4>(w, 0), &meta()).is_err());
    let w = insert::<28, 2>(insert::<52, 4>(w, 3), 0);
    let (ir, b) = compile(Opcode::ATOMS, w, &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::SharedAtomicInc32).len(), 1);
}
#[test]
fn stage_attributes_preserve_vertex_patch_and_indexed_paths() {
    let w = insert::<47, 2>(
        insert::<20, 10>(insert::<8, 8>(word(Opcode::ALD), 255), 128),
        2,
    );
    let (ir, b) = compile(Opcode::ALD, w, &meta()).unwrap();
    let attrs = ops(&ir, b, I::GetAttribute);
    assert_eq!(attrs.len(), 3);
    assert_eq!(
        attrs[0].arg(0).unwrap(),
        Value::Attribute(Attribute::Generic0X)
    );
    assert_eq!(
        attrs[2].arg(0).unwrap(),
        Value::Attribute(Attribute::Generic0Z)
    );
    let (ir, b) = compile(Opcode::ALD, w | 1 << 31, &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::GetPatch).len(), 3);
    let (ir, b) = compile(Opcode::ALD, insert::<8, 8>(w, 2), &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::GetAttributeIndexed).len(), 3);
    let adds = ops(&ir, b, I::IAdd32);
    assert_eq!(adds[0].arg(1).unwrap(), Value::U32(4));
    assert_eq!(adds[1].arg(1).unwrap(), Value::U32(8));
    assert!(compile(Opcode::ALD, insert::<8, 8>(w | 1 << 31, 2), &meta()).is_err());
    assert!(compile(Opcode::AST, insert::<8, 8>(w, 2), &meta()).is_err());
    let (ir, b) = compile(Opcode::AST, w | 1 << 31, &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::SetPatch).len(), 3);
}
#[test]
fn sparse_fragment_outputs_advance_over_masked_components_and_skip_empty_targets() {
    let mut bytes = [0; 80];
    bytes[72..76].copy_from_slice(&(0b0101u32 << 4 | 0b1010 << 20).to_le_bytes());
    bytes[76] = 3;
    let metadata = EnvironmentMetadata {
        stage: Stage::Fragment,
        sph: ProgramHeader::from_bytes(&bytes),
        ..meta()
    };
    let (ir, b) = compile(Opcode::EXIT, word(Opcode::EXIT), &metadata).unwrap();
    assert_eq!(
        reads(&ir, b),
        [0, 2, 5, 7, 8, 9].map(|r| Value::Reg(Reg(r)))
    );
    let colors = ops(&ir, b, I::SetFragColor);
    assert_eq!(
        colors
            .iter()
            .map(|i| (i.arg(0).unwrap(), i.arg(1).unwrap()))
            .collect::<Vec<_>>(),
        [(1, 0), (1, 2), (5, 1), (5, 3)].map(|(r, c)| (Value::U32(r), Value::U32(c)))
    );
    assert_eq!(ops(&ir, b, I::SetSampleMask).len(), 1);
    assert_eq!(ops(&ir, b, I::SetFragDepth).len(), 1);
    let (ir, b) = compile(Opcode::EXIT, word(Opcode::EXIT), &meta()).unwrap();
    assert!(ir.block(b).unwrap().instructions().is_empty());
}
#[test]
fn barriers_and_quad_shuffle_have_exact_scope_and_write_order() {
    for scope in 0..4 {
        let (ir, b) = compile(
            Opcode::MEMBAR,
            insert::<8, 2>(word(Opcode::MEMBAR), scope),
            &meta(),
        )
        .unwrap();
        assert_eq!(
            ops(
                &ir,
                b,
                if scope == 0 {
                    I::WorkgroupMemoryBarrier
                } else {
                    I::DeviceMemoryBarrier
                }
            )
            .len(),
            1
        );
    }
    let w = Opcode::BAR.encoding().value | 7 << 16 | 1 << 39 | 1 << 43 | 1 << 44;
    let (ir, b) = compile(Opcode::BAR, w, &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::Barrier).len(), 1);
    assert!(compile(Opcode::BAR, w | 1 << 42, &meta()).is_err());
    let w = insert::<34, 13>(
        insert::<20, 5>(word(Opcode::SHFL), 2) | 1 << 28 | 1 << 29,
        (28 << 8) | 3,
    );
    let metadata = EnvironmentMetadata {
        stage: Stage::Fragment,
        ..meta()
    };
    let (ir, b) = compile(Opcode::SHFL, w, &metadata).unwrap();
    assert_eq!(ops(&ir, b, I::QuadBroadcast).len(), 1);
    let insts = ir.block(b).unwrap().instructions();
    assert_eq!(
        ir.inst(insts[insts.len() - 2]).unwrap().opcode(),
        I::SetRegister
    );
    assert_eq!(
        ir.inst(*insts.last().unwrap()).unwrap().opcode(),
        I::SetPred
    );
    let (ir, b) = compile(Opcode::SHFL, w, &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::ShuffleIndex).len(), 1);
    assert_eq!(ops(&ir, b, I::GetInBoundsFromOp).len(), 1);
}

#[test]
fn interpolation_consumes_first_active_component_and_keeps_multiply_and_saturation() {
    let w = insert::<54, 2>(insert::<30, 8>(word(Opcode::IPA), 32), 1) | 1 << 51;
    let (ir, b) = compile(Opcode::IPA, w, &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::FPMul32).len(), 2);
    assert_eq!(ops(&ir, b, I::FPSaturate32).len(), 1);
    let mut bytes = [0; 80];
    bytes[24] = 1 << 4; // The third component is constant; the first two are unused.
    let metadata = EnvironmentMetadata {
        sph: ProgramHeader::from_bytes(&bytes),
        ..meta()
    };
    let (ir, b) = compile(Opcode::IPA, w, &metadata).unwrap();
    assert_eq!(ops(&ir, b, I::FPMul32).len(), 1);
    assert!(compile(Opcode::IPA, insert::<30, 8>(w, 255), &metadata).is_err());
}

#[test]
fn internal_buffers_geometry_sample_and_vote_keep_builtin_routes() {
    let w = insert::<36, 4>(insert::<47, 2>(word(Opcode::ISBERD), 0), 1) | 1 << 32;
    let (ir, b) = compile(Opcode::ISBERD, w, &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::LoadGlobalU16).len(), 1);
    let w = insert::<33, 2>(w & !(1 << 32), 1);
    assert_eq!(
        compile(Opcode::ISBERD, w, &meta()).err().unwrap().category,
        FailureCategory::UnsupportedReferenceCase
    );
    let (ir, b) = compile(Opcode::ISBERD, insert::<33, 2>(w, 3), &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::GetAttributeIndexed).len(), 1);
    let (ir, b) = compile(
        Opcode::OUT_reg,
        word(Opcode::OUT_reg) | 1 << 39 | 1 << 40,
        &meta(),
    )
    .unwrap();
    assert_eq!(ops(&ir, b, I::EmitVertex).len(), 1);
    assert_eq!(ops(&ir, b, I::EndPrimitive).len(), 1);
    assert_eq!(
        ops(&ir, b, I::SetRegister)[0].arg(1).unwrap(),
        Value::U32(0)
    );
    let w = insert::<31, 3>(
        insert::<20, 8>(insert::<8, 8>(word(Opcode::PIXLD), 255), 0),
        5,
    ) | 7 << 45;
    let (ir, b) = compile(Opcode::PIXLD, w, &meta()).unwrap();
    assert_eq!(ops(&ir, b, I::SampleId).len(), 1);
    let (ir, b) = compile(
        Opcode::VOTE,
        insert::<48, 2>(word(Opcode::VOTE), 1),
        &meta(),
    )
    .unwrap();
    assert_eq!(ops(&ir, b, I::VoteAny).len(), 1);
    assert_eq!(ops(&ir, b, I::SubgroupBallot).len(), 1);
}
