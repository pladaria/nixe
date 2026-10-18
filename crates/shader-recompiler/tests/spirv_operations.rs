use nixe_shader_recompiler::{
    Profile, RuntimeInfo,
    arena::{BlockId, InstId},
    artifact::{Bindings, ConstantBufferDescriptor, StorageBufferDescriptor, TextureDescriptor},
    diagnostics::FailureCategory,
    ir::{Attribute, FpControl, Ir, Opcode as O, Program, TextureInstInfo, Type, Value as V},
    spirv::{EmissionRoute, EmitContext, Id, emission_route},
    types::{ByteOffset, CbufIndex, Stage, TextureType},
};
use spirv::{Decoration, Op};
fn instructions(words: &[u32]) -> Vec<(Op, &[u32])> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        out.push((
            Op::from_u32(words[i] & 65535).unwrap(),
            &words[i + 1..i + n],
        ));
        i += n;
    }
    out
}
fn append(ir: &mut Ir, block: BlockId, op: O, args: &[V]) -> InstId {
    ir.append(block, op, args, 0).unwrap()
}
fn emit(ctx: &mut EmitContext<'_>, ir: &mut Ir, block: BlockId) {
    let ty = ctx.builder.type_function(ctx.types.void, &[]).unwrap();
    ctx.builder.begin_function(ctx.types.void, ty).unwrap();
    ctx.builder.new_label().unwrap();
    for id in ir.block(block).unwrap().instructions().to_vec() {
        ctx.emit_instruction(ir, id).unwrap();
    }
    ctx.builder.instruction(Op::Return, &[]).unwrap();
    ctx.builder.end_function().unwrap();
}
fn definition(ir: &Ir, id: InstId) -> u32 {
    ir.inst(id).unwrap().definition
}
fn constant(inst: &[(Op, &[u32])], id: u32) -> u32 {
    inst.iter()
        .find(|(op, a)| *op == Op::Constant && a[1] == id)
        .unwrap()
        .1[2]
}

#[test]
fn split_draw_instance_ids_subtract_the_logical_base() {
    for base in [0, 7] {
        let mut ir = Ir::new().unwrap();
        let block = ir.new_block().unwrap();
        let instance = append(
            &mut ir,
            block,
            O::GetAttributeU32,
            &[V::Attribute(Attribute::InstanceId), V::U32(0)],
        );
        let logical_base = append(
            &mut ir,
            block,
            O::GetAttributeU32,
            &[V::Attribute(Attribute::BaseInstance), V::U32(0)],
        );
        let mut p = Program::new(Stage::VertexB);
        p.info.loads.set(Attribute::InstanceId, true);
        p.info.loads.set(Attribute::BaseInstance, true);
        let profile = profile();
        let runtime = RuntimeInfo {
            base_instance: Some(base),
            ..Default::default()
        };
        let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
        emit(&mut ctx, &mut ir, block);
        let words = ctx.assemble().unwrap();
        let ops = instructions(&words);
        let subtract = ops
            .iter()
            .find(|(op, args)| *op == Op::ISub && args[1] == definition(&ir, instance))
            .unwrap()
            .1;
        assert_eq!(constant(&ops, subtract[3]), base);
        assert_eq!(constant(&ops, definition(&ir, logical_base)), base);
        // Vulkan's InstanceIndex contains firstInstance even for a single-instance
        // host draw. The logical guest sequence must keep advancing across draws.
        for relative in 0..4u32 {
            assert_eq!(
                (base + relative).wrapping_sub(constant(&ops, subtract[3])),
                relative
            );
        }
    }
}
fn profile() -> Profile {
    Profile {
        supported_spirv: 0x00010500,
        unified_descriptor_binding: true,
        ..Default::default()
    }
}
#[test]
fn dispatch_separates_illegal_unlowered_and_unsupported_operations() {
    assert_eq!(O::ALL.len(), 534);
    assert_eq!(emission_route(O::GetZeroFromOp), EmissionRoute::Invariant);
    assert_eq!(
        emission_route(O::BindlessImageFetch),
        EmissionRoute::Invariant
    );
    assert_eq!(
        emission_route(O::GlobalAtomicIAdd32),
        EmissionRoute::Unsupported
    );
    let p = Program::new(Stage::Compute);
    let profile = profile();
    let runtime = RuntimeInfo::default();
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let undef = append(&mut ir, block, O::UndefU64, &[]);
    let reg = append(
        &mut ir,
        block,
        O::GetRegister,
        &[V::Reg(nixe_shader_recompiler::ir::Reg(0))],
    );
    assert_eq!(
        ctx.emit_instruction(&mut ir, undef).unwrap_err().category,
        FailureCategory::UnsupportedReferenceCase
    );
    assert_eq!(
        ctx.emit_instruction(&mut ir, reg).unwrap_err().category,
        FailureCategory::Internal
    );
}
#[test]
fn add_defines_and_consumes_all_four_auxiliary_flags() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let sum = append(&mut ir, block, O::IAdd32, &[V::U32(u32::MAX), V::U32(1)]);
    let flags = [
        O::GetCarryFromOp,
        O::GetZeroFromOp,
        O::GetSignFromOp,
        O::GetOverflowFromOp,
    ]
    .map(|op| append(&mut ir, block, op, &[V::Inst(sum)]));
    let p = Program::new(Stage::Compute);
    let profile = profile();
    let runtime = RuntimeInfo::default();
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    emit(&mut ctx, &mut ir, block);
    for id in flags {
        assert_eq!(ir.inst(id).unwrap().opcode(), O::Void);
        assert_ne!(definition(&ir, id), 0);
    }
    let words = ctx.assemble().unwrap();
    let inst = instructions(&words);
    assert_eq!(
        inst.iter().filter(|(op, _)| *op == Op::IAddCarry).count(),
        1
    );
    assert!(
        inst.iter()
            .any(|(op, a)| *op == Op::INotEqual && a[1] == definition(&ir, flags[0]))
    );
    assert!(
        inst.iter()
            .any(|(op, a)| *op == Op::IEqual && a[1] == definition(&ir, flags[1]))
    );
    assert!(
        inst.iter()
            .any(|(op, a)| *op == Op::SLessThan && a[1] == definition(&ir, flags[2]))
    );
    assert!(
        inst.iter()
            .any(|(op, a)| *op == Op::Select && a[1] == definition(&ir, flags[3]))
    );
}
#[test]
fn floating_controls_and_literal_nan_comparison_branch_are_retained() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let sum = ir
        .append(
            block,
            O::FPAdd32,
            &[V::f32(1.0), V::f32(2.0)],
            FpControl {
                no_contraction: true,
                ..Default::default()
            }
            .bits(),
        )
        .unwrap();
    let compare = append(
        &mut ir,
        block,
        O::FPOrdNotEqual32,
        &[V::Inst(sum), V::f32(f32::NAN)],
    );
    let p = Program::new(Stage::Compute);
    let profile = Profile {
        ignore_nan_fp_comparisons: true,
        ..profile()
    };
    let runtime = RuntimeInfo::default();
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    emit(&mut ctx, &mut ir, block);
    let words = ctx.assemble().unwrap();
    let inst = instructions(&words);
    assert!(inst.iter().any(|(op, a)| *op == Op::Decorate
        && *a == [definition(&ir, sum), Decoration::NoContraction as u32]));
    assert!(inst.iter().any(|(op, _)| *op == Op::FOrdEqual));
    assert!(!inst.iter().any(|(op, _)| *op == Op::FOrdNotEqual));
    assert!(
        inst.iter()
            .any(|(op, a)| *op == Op::LogicalAnd && a[1] == definition(&ir, compare))
    );
}
#[test]
fn narrow_conversions_work_without_native_integer_widths() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let narrow = append(&mut ir, block, O::ConvertU8U32, &[V::U32(0x1234)]);
    let wide = append(&mut ir, block, O::ConvertU32U8, &[V::Inst(narrow)]);
    let signed = append(&mut ir, block, O::ConvertF32S8, &[V::U32(0x80)]);
    let p = Program::new(Stage::Compute);
    let profile = profile();
    let runtime = RuntimeInfo::default();
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    emit(&mut ctx, &mut ir, block);
    let words = ctx.assemble().unwrap();
    let inst = instructions(&words);
    for id in [narrow, wide] {
        assert!(inst.iter().any(|(op, a)| *op == Op::BitFieldUExtract
            && a[1] == definition(&ir, id)
            && a[0] == ctx.types.u32[0].0));
    }
    assert!(
        inst.iter()
            .any(|(op, a)| *op == Op::ConvertSToF && a[1] == definition(&ir, signed))
    );
    let extract = inst
        .iter()
        .find(|(op, _)| *op == Op::BitFieldSExtract)
        .unwrap()
        .1;
    assert_eq!(constant(&inst, extract[4]), 8);
}
#[test]
fn cbuf_alignment_and_nonaliased_vector_component_selection() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let load = append(&mut ir, block, O::GetCbufU32, &[V::U32(0), V::U32(22)]);
    let mut p = Program::new(Stage::Compute);
    p.info.used_constant_buffer_types = Type::U32;
    p.info
        .constant_buffer_descriptors
        .push(ConstantBufferDescriptor {
            index: CbufIndex(0),
            count: 1,
        })
        .unwrap();
    let profile = profile();
    let runtime = RuntimeInfo::default();
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    emit(&mut ctx, &mut ir, block);
    let words = ctx.assemble().unwrap();
    let inst = instructions(&words);
    let pointer = inst
        .iter()
        .find(|(op, _)| *op == Op::AccessChain)
        .unwrap()
        .1;
    assert_eq!(constant(&inst, pointer[4]), 1);
    let extract = inst
        .iter()
        .find(|(op, a)| *op == Op::CompositeExtract && a[1] == definition(&ir, load))
        .unwrap()
        .1;
    assert_eq!(extract[3], 1);
}
#[test]
fn storage_vector_fallback_and_narrow_write_use_scalar_views_and_cas() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    append(&mut ir, block, O::LoadStorage64, &[V::U32(0), V::U32(12)]);
    append(
        &mut ir,
        block,
        O::WriteStorageU8,
        &[V::U32(0), V::U32(7), V::U32(0xff)],
    );
    let mut p = Program::new(Stage::Compute);
    p.info.uses_int8 = true;
    p.info.used_storage_buffer_types = Type::U8 | Type::U32x2;
    p.info
        .storage_buffers_descriptors
        .push(StorageBufferDescriptor {
            cbuf_index: CbufIndex(0),
            cbuf_offset: ByteOffset(0),
            count: 1,
            is_written: true,
        })
        .unwrap();
    let profile = profile();
    let runtime = RuntimeInfo::default();
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    emit(&mut ctx, &mut ir, block);
    let words = ctx.assemble().unwrap();
    let inst = instructions(&words);
    assert!(inst.iter().any(|(op, _)| *op == Op::AtomicCompareExchange));
    let call = inst
        .iter()
        .find(|(op, a)| {
            *op == Op::FunctionCall && a[2] == ctx.objects.write_storage_cas_loop_func.0
        })
        .unwrap()
        .1;
    assert_eq!(constant(&inst, call[5]), 24);
    assert_eq!(constant(&inst, call[6]), 8);
    let indices: Vec<_> = inst
        .iter()
        .filter(|(op, a)| {
            *op == Op::AccessChain
                && a[0]
                    == ctx.storage_types[nixe_shader_recompiler::spirv::BufferKind::U32 as usize]
                        .element
                        .0
        })
        .map(|(_, a)| constant(&inst, a[4]))
        .collect();
    assert_eq!(indices, [3, 4, 1]);
}
#[test]
fn shared_atomic_and_barriers_keep_device_and_workgroup_scopes() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    append(
        &mut ir,
        block,
        O::SharedAtomicIAdd32,
        &[V::U32(4), V::U32(1)],
    );
    append(&mut ir, block, O::Barrier, &[]);
    append(&mut ir, block, O::DeviceMemoryBarrier, &[]);
    let mut p = Program::new(Stage::Compute);
    p.execution.shared_memory_size = 64;
    let profile = profile();
    let runtime = RuntimeInfo::default();
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    emit(&mut ctx, &mut ir, block);
    let words = ctx.assemble().unwrap();
    let inst = instructions(&words);
    let atomic = inst.iter().find(|(op, _)| *op == Op::AtomicIAdd).unwrap().1;
    assert_eq!(constant(&inst, atomic[3]), spirv::Scope::Device as u32);
    assert_eq!(constant(&inst, atomic[4]), 0);
    let barrier = inst
        .iter()
        .find(|(op, _)| *op == Op::ControlBarrier)
        .unwrap()
        .1;
    assert_eq!(constant(&inst, barrier[0]), spirv::Scope::Workgroup as u32);
    assert_eq!(
        constant(&inst, barrier[2]),
        (spirv::MemorySemantics::ACQUIRE_RELEASE | spirv::MemorySemantics::WORKGROUP_MEMORY).bits()
    );
}
#[test]
fn shuffle_rewrite_and_bounds_pseudo_survive_assembly() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let shuffle = append(
        &mut ir,
        block,
        O::ShuffleUp,
        &[V::U32(7), V::U32(1), V::U32(0), V::U32(0)],
    );
    let bounds = append(&mut ir, block, O::GetInBoundsFromOp, &[V::Inst(shuffle)]);
    let mut p = Program::new(Stage::Compute);
    p.info.uses_subgroup_invocation_id = true;
    p.info.uses_subgroup_shuffles = true;
    let profile = Profile {
        support_shuffle_relative: true,
        warp_size_potentially_larger_than_guest: true,
        max_subgroup_size: 64,
        ..profile()
    };
    let runtime = RuntimeInfo::default();
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    emit(&mut ctx, &mut ir, block);
    let words = ctx.assemble().unwrap();
    let inst = instructions(&words);
    assert!(
        inst.iter()
            .any(|(op, _)| *op == Op::GroupNonUniformShuffleUp)
    );
    assert!(
        !inst
            .iter()
            .any(|(op, _)| *op == Op::GroupNonUniformShuffleXor)
    );
    assert_eq!(ir.inst(bounds).unwrap().opcode(), O::Void);
    assert!(
        inst.iter()
            .any(|(op, a)| *op == Op::SGreaterThanEqual && a[1] == definition(&ir, bounds))
    );
}
#[test]
fn disabled_subgroups_use_defined_single_lane_vote_behavior() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let equal = append(&mut ir, block, O::VoteEqual, &[V::U1(false)]);
    let any = append(&mut ir, block, O::VoteAny, &[V::U1(false)]);
    let ballot = append(&mut ir, block, O::SubgroupBallot, &[V::U1(true)]);
    let p = Program::new(Stage::Compute);
    let profile = Profile {
        supported_subgroup_stages: 0,
        ..profile()
    };
    let runtime = RuntimeInfo::default();
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    emit(&mut ctx, &mut ir, block);
    assert_eq!(definition(&ir, equal), ctx.constants.true_value.0);
    assert_eq!(definition(&ir, any), ctx.constants.false_value.0);
    let words = ctx.assemble().unwrap();
    let inst = instructions(&words);
    assert!(
        inst.iter()
            .any(|(op, a)| *op == Op::Select && a[1] == definition(&ir, ballot))
    );
}
fn image_flags() -> u32 {
    let mut info = TextureInstInfo::default();
    info.set_texture_type(TextureType::Color2D);
    info.0
}
#[test]
fn sparse_sampling_has_residency_and_compute_uses_explicit_zero_lod() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let coords = append(
        &mut ir,
        block,
        O::CompositeConstructF32x2,
        &[V::f32(0.25), V::f32(0.5)],
    );
    let sample = ir
        .append(
            block,
            O::ImageSampleImplicitLod,
            &[V::U32(0), V::Inst(coords), V::None, V::None],
            image_flags(),
        )
        .unwrap();
    let sparse = append(&mut ir, block, O::GetSparseFromOp, &[V::Inst(sample)]);
    let mut p = Program::new(Stage::Compute);
    p.info.texture_descriptors.push(TextureDescriptor {
        texture_type: TextureType::Color2D,
        count: 1,
        ..Default::default()
    });
    let profile = profile();
    let runtime = RuntimeInfo::default();
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    emit(&mut ctx, &mut ir, block);
    let words = ctx.assemble().unwrap();
    let inst = instructions(&words);
    let sample = inst
        .iter()
        .find(|(op, _)| *op == Op::ImageSparseSampleExplicitLod)
        .unwrap()
        .1;
    assert_eq!(sample[4], spirv::ImageOperands::LOD.bits());
    assert_eq!(constant(&inst, sample[5]), 0);
    assert_eq!(ir.inst(sparse).unwrap().opcode(), O::Void);
    assert!(
        inst.iter()
            .any(|(op, a)| *op == Op::ImageSparseTexelsResident && a[1] == definition(&ir, sparse))
    );
}
#[test]
fn image_offsets_keep_constants_and_nonuniform_array_access_marks_all_objects() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let coords = append(
        &mut ir,
        block,
        O::CompositeConstructF32x2,
        &[V::f32(0.0), V::f32(0.0)],
    );
    let offset = append(
        &mut ir,
        block,
        O::CompositeConstructU32x2,
        &[V::U32(u32::MAX), V::U32(2)],
    );
    let index = append(&mut ir, block, O::UndefU32, &[]);
    ir.append(
        block,
        O::ImageSampleExplicitLod,
        &[
            V::Inst(index),
            V::Inst(coords),
            V::f32(2.0),
            V::Inst(offset),
        ],
        image_flags(),
    )
    .unwrap();
    let mut p = Program::new(Stage::Fragment);
    p.info.texture_descriptors.push(TextureDescriptor {
        texture_type: TextureType::Color2D,
        count: 4,
        ..Default::default()
    });
    let profile = Profile {
        support_sampled_image_array_nonuniform_indexing: true,
        ..profile()
    };
    let runtime = RuntimeInfo::default();
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    emit(&mut ctx, &mut ir, block);
    let words = ctx.assemble().unwrap();
    let inst = instructions(&words);
    let sample = inst
        .iter()
        .find(|(op, _)| *op == Op::ImageSampleExplicitLod)
        .unwrap()
        .1;
    assert_eq!(
        sample[4],
        (spirv::ImageOperands::LOD | spirv::ImageOperands::CONST_OFFSET).bits()
    );
    let vector = inst
        .iter()
        .find(|(op, a)| *op == Op::ConstantComposite && a[1] == sample[6])
        .unwrap()
        .1;
    assert_eq!(vector[0], ctx.types.s32[1].0);
    assert_eq!(constant(&inst, vector[2]), u32::MAX);
    assert_eq!(ctx.non_uniform_ids.len(), 3);
    assert!(ctx.uses_nonuniform[0]);
}
#[test]
fn vertex_defaults_and_depth_conversion_write_position() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    append(&mut ir, block, O::Prologue, &[]);
    append(
        &mut ir,
        block,
        O::SetAttribute,
        &[V::Attribute(Attribute::PositionZ), V::f32(-1.0), V::U32(0)],
    );
    append(&mut ir, block, O::Epilogue, &[]);
    let p = Program::new(Stage::VertexB);
    let profile = profile();
    let runtime = RuntimeInfo {
        convert_depth_mode: true,
        ..Default::default()
    };
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    emit(&mut ctx, &mut ir, block);
    let words = ctx.assemble().unwrap();
    let inst = instructions(&words);
    assert!(inst.iter().any(|(op, _)| *op == Op::FAdd));
    assert!(inst.iter().any(|(op, _)| *op == Op::FMul));
    assert_eq!(
        inst.iter()
            .filter(|(op, a)| *op == Op::Store && a[0] == ctx.objects.output_position.0)
            .count(),
        2
    );
}
#[test]
fn phi_accepts_forward_values_but_rejects_unsupported_types() {
    let mut ir = Ir::new().unwrap();
    let a = ir.new_block().unwrap();
    let b = ir.new_block().unwrap();
    let join = ir.new_block().unwrap();
    let phi = ir.new_phi(join, Type::U32).unwrap();
    let later = append(&mut ir, b, O::IAdd32, &[V::U32(1), V::U32(2)]);
    ir.add_phi_operand(phi, a, V::U32(9)).unwrap();
    ir.add_phi_operand(phi, b, V::Inst(later)).unwrap();
    let p = Program::new(Stage::Compute);
    let profile = profile();
    let runtime = RuntimeInfo::default();
    let mut ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    for block in [a, b, join] {
        ir.block_mut(block).unwrap().definition = ctx.builder.reserve_id().unwrap().0;
    }
    ctx.emit_instruction(&mut ir, phi).unwrap();
    assert!(ctx.assemble().is_err());
    ctx.emit_instruction(&mut ir, later).unwrap();
    ctx.patch_phis(&ir).unwrap();
    let words = ctx.assemble().unwrap();
    let inst = instructions(&words);
    let args = inst.iter().find(|(op, _)| *op == Op::Phi).unwrap().1;
    assert_eq!(args[4], definition(&ir, later));
    assert_eq!(args[5], ir.block(b).unwrap().definition);
    assert_ne!(Id(definition(&ir, phi)), Id::NONE);
}
