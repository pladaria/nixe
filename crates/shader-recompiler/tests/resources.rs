use nixe_shader_recompiler::{
    Environment, HostTranslateInfo,
    arena::{BlockId, InstId},
    environment::{
        CbufWordKey, EnvironmentCaches, EnvironmentMetadata, Observations, RecordingEnvironment,
        SnapshotEnvironment,
    },
    ir::{Emitter, Ir, Opcode as O, Program, TextureInstInfo, Value},
    passes::{
        collect_shader_info, dead_code_elimination, global_memory_to_storage, remove_identities,
        texture_pass, verify_program,
    },
    types::{
        ByteOffset, CbufIndex, Stage, TextureHandle, TexturePixelFormat as P, TextureType as T,
    },
};
fn program() -> (Ir, Program, BlockId) {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let mut program = Program::new(Stage::Compute);
    program.blocks.push(block);
    program.post_order.push(block);
    (ir, program, block)
}
fn id(value: Value) -> InstId {
    let Value::Inst(id) = value else { panic!() };
    id
}
fn flags(ty: T) -> u32 {
    let mut f = TextureInstInfo::default();
    f.set_texture_type(ty);
    f.0
}
fn host() -> HostTranslateInfo {
    HostTranslateInfo {
        min_ssbo_alignment: 256,
        max_per_stage_descriptor_sampled_images: 32,
        max_descriptor_set_sampled_images: 64,
        max_per_stage_resources: 64,
        support_snorm_render_buffer: true,
        ..Default::default()
    }
}
struct SourceEnvironment(SnapshotEnvironment);
impl Environment for SourceEnvironment {
    fn metadata(&self) -> &EnvironmentMetadata {
        self.0.metadata()
    }
    fn read_instruction(
        &mut self,
        address: nixe_shader_recompiler::types::ProgramOffset,
    ) -> nixe_shader_recompiler::Result<u64> {
        self.0.read_instruction(address)
    }
    fn read_cbuf_value(
        &mut self,
        index: CbufIndex,
        offset: ByteOffset,
    ) -> nixe_shader_recompiler::Result<u32> {
        self.0.read_cbuf_value(index, offset)
    }
    fn read_texture_type(&mut self, handle: TextureHandle) -> nixe_shader_recompiler::Result<T> {
        self.0.read_texture_type(handle)
    }
    fn read_texture_pixel_format(
        &mut self,
        handle: TextureHandle,
    ) -> nixe_shader_recompiler::Result<P> {
        self.0.read_texture_pixel_format(handle)
    }
    fn is_texture_pixel_format_integer(
        &mut self,
        handle: TextureHandle,
    ) -> nixe_shader_recompiler::Result<bool> {
        self.0.is_texture_pixel_format_integer(handle)
    }
    fn read_viewport_transform_state(&mut self) -> nixe_shader_recompiler::Result<u32> {
        self.0.read_viewport_transform_state()
    }
    fn get_replace_const_buffer(
        &mut self,
        bank: CbufIndex,
        offset: ByteOffset,
    ) -> nixe_shader_recompiler::Result<Option<nixe_shader_recompiler::types::ReplaceConstant>>
    {
        self.0.get_replace_const_buffer(bank, offset)
    }
}

fn env(
    words: &[(u32, u32, u32)],
    types: &[(u32, T)],
    integers: &[(u32, bool)],
) -> SourceEnvironment {
    let mut obs = Observations::default();
    for &(index, offset, word) in words {
        obs.cbuf_words.insert(
            CbufWordKey {
                index: CbufIndex(index),
                offset: ByteOffset(offset),
            },
            word,
        );
    }
    for &(handle, ty) in types {
        obs.texture_types.insert(TextureHandle(handle), ty);
    }
    for &(handle, integer) in integers {
        obs.texture_integer_formats
            .insert(TextureHandle(handle), integer);
    }
    SourceEnvironment(SnapshotEnvironment::new(
        EnvironmentMetadata {
            stage: Stage::Compute,
            texture_bound_buffer: CbufIndex(2),
            ..Default::default()
        },
        obs,
    ))
}
#[test]
fn sorted_shared_and_independent_texture_descriptors_record_only_consumed_facts() {
    let (mut ir, mut p, block) = program();
    let mut e = Emitter::new(&mut ir, block).unwrap();
    let coords = e
        .emit(O::CompositeConstructF32x2, &[Value::f32(0.0); 2], 0)
        .unwrap();
    let mut values = Vec::new();
    for offset in [32, 8, 32] {
        values.push(
            e.emit(
                O::BoundImageSampleExplicitLod,
                &[Value::U32(offset), coords, Value::f32(0.0), Value::None],
                flags(T::Color2D),
            )
            .unwrap(),
        );
    }
    let mut source = env(
        &[(2, 8, 11), (2, 32, 22), (2, 64, 99)],
        &[],
        &[(11, false), (22, true)],
    );
    let mut recorder = RecordingEnvironment::new(&mut source);
    let mut caches = EnvironmentCaches::default();
    texture_pass(&mut ir, &mut p, &mut recorder, &host(), &mut caches).unwrap();
    assert_eq!(p.info.texture_descriptors.len(), 2);
    assert_eq!(
        p.info
            .texture_descriptors
            .iter()
            .map(|d| d.cbuf_offset.0)
            .collect::<Vec<_>>(),
        [8, 32]
    );
    for (value, expected) in values.iter().zip([1, 0, 1]) {
        let inst = ir.inst(id(*value)).unwrap();
        assert_eq!(inst.opcode(), O::ImageSampleExplicitLod);
        assert_eq!(inst.arg(0).unwrap(), Value::None);
        assert_eq!(TextureInstInfo(inst.flags).descriptor_index(), expected);
        assert_eq!(TextureInstInfo(inst.flags).is_integer(), expected == 1);
    }
    let snapshot = recorder.into_snapshot();
    assert!(snapshot.observations().cbuf_words.is_empty());
    assert_eq!(snapshot.observations().texture_handles.len(), 2);
    assert_eq!(snapshot.observations().texture_integer_formats.len(), 2);
    assert!(snapshot.observations().texture_types.is_empty());
    assert!(caches.cbuf_word_cache.is_empty());
    assert_eq!(caches.handle_cache.len(), 2);
    verify_program(&ir, &p).unwrap();
}
#[test]
fn separate_sampler_tracks_shifts_and_rejects_storage_images() {
    for storage in [false, true] {
        let (mut ir, mut p, block) = program();
        let mut e = Emitter::new(&mut ir, block).unwrap();
        let tex = e
            .get_cbuf(Value::U32(2), Value::U32(16), 32, false)
            .unwrap();
        let sampler = e.get_cbuf(Value::U32(3), Value::U32(8), 32, false).unwrap();
        let sampler = e
            .emit(O::ShiftLeftLogical32, &[sampler, Value::U32(20)], 0)
            .unwrap();
        let handle = e.emit(O::BitwiseOr32, &[sampler, tex], 0).unwrap();
        if storage {
            e.emit(
                O::BindlessImageRead,
                &[handle, Value::U32(0)],
                flags(T::Color1D),
            )
            .unwrap();
        } else {
            e.emit(
                O::BindlessImageQueryDimensions,
                &[handle, Value::U32(0), Value::U1(true)],
                0,
            )
            .unwrap();
        }
        let raw = 7 | (5 << 20);
        let mut env = env(
            &[(2, 16, 7), (3, 8, 5)],
            &[(raw, T::Color1D)],
            &[(raw, false)],
        );
        let mut caches = EnvironmentCaches::default();
        let result = texture_pass(&mut ir, &mut p, &mut env, &host(), &mut caches);
        if storage {
            assert!(result.unwrap_err().message.contains("separate sampler"));
        } else {
            result.unwrap();
            let d = &p.info.texture_descriptors[0];
            assert!(d.has_secondary);
            assert_eq!(d.cbuf_index, CbufIndex(2));
            assert_eq!(d.secondary_cbuf_index, CbufIndex(3));
            assert_eq!(d.secondary_shift_left, 20);
            assert_eq!(
                caches.handle_cache.values().copied().collect::<Vec<_>>(),
                [raw]
            );
        }
    }
}
#[test]
fn dynamic_arrays_use_stride_and_instruction_budget_before_deduplication() {
    let (mut ir, mut p, block) = program();
    let mut e = Emitter::new(&mut ir, block).unwrap();
    let dynamic = e.emit(O::UndefU32, &[], 0).unwrap();
    let dynamic = e
        .emit(O::ShiftLeftLogical32, &[dynamic, Value::U32(4)], 0)
        .unwrap();
    let offset = e.i_add(dynamic, Value::U32(16)).unwrap();
    let handle = e.get_cbuf(Value::U32(2), offset, 32, false).unwrap();
    let mut queries = Vec::new();
    for _ in 0..2 {
        queries.push(
            e.emit(
                O::BindlessImageQueryDimensions,
                &[handle, Value::U32(0), Value::U1(true)],
                flags(T::Color2D),
            )
            .unwrap(),
        );
    }
    let mut env = env(&[(2, 16, 123)], &[(123, T::Color2D)], &[(123, false)]);
    let mut h = host();
    h.max_per_stage_resources = 20;
    texture_pass(
        &mut ir,
        &mut p,
        &mut env,
        &h,
        &mut EnvironmentCaches::default(),
    )
    .unwrap();
    let d = &p.info.texture_descriptors[0];
    assert_eq!(p.info.texture_descriptors.len(), 1);
    assert_eq!(d.count, 10);
    assert_eq!(d.size_shift, 4);
    for query in queries {
        let index = id(ir.inst(id(query)).unwrap().arg(0).unwrap());
        assert_eq!(ir.inst(index).unwrap().opcode(), O::UMin32);
        assert_eq!(ir.inst(index).unwrap().arg(1).unwrap(), Value::U32(9));
        let shift = id(ir.inst(index).unwrap().arg(0).unwrap());
        assert_eq!(ir.inst(shift).unwrap().arg(1).unwrap(), Value::U32(4));
    }
    verify_program(&ir, &p).unwrap();
}
#[test]
fn rectangle_and_snorm_patches_have_typed_operands_and_preserve_fetches() {
    let (mut ir, mut p, block) = program();
    let mut e = Emitter::new(&mut ir, block).unwrap();
    let coords = e
        .emit(
            O::CompositeConstructF32x2,
            &[Value::f32(4.0), Value::f32(8.0)],
            0,
        )
        .unwrap();
    let sample = e
        .emit(
            O::BoundImageSampleImplicitLod,
            &[Value::U32(8), coords, Value::None, Value::None],
            flags(T::Color2D),
        )
        .unwrap();
    let fetch = e
        .emit(
            O::BoundImageFetch,
            &[
                Value::U32(16),
                Value::U32(4),
                Value::None,
                Value::U32(0),
                Value::None,
            ],
            flags(T::Color1D),
        )
        .unwrap();
    let mut observations = Observations::default();
    observations.cbuf_words.insert(
        CbufWordKey {
            index: CbufIndex(2),
            offset: ByteOffset(8),
        },
        1,
    );
    observations.cbuf_words.insert(
        CbufWordKey {
            index: CbufIndex(2),
            offset: ByteOffset(16),
        },
        2,
    );
    observations
        .texture_types
        .insert(TextureHandle(1), T::Color2DRect);
    observations
        .texture_types
        .insert(TextureHandle(2), T::Buffer);
    observations
        .texture_integer_formats
        .insert(TextureHandle(1), false);
    observations
        .texture_pixel_formats
        .insert(TextureHandle(2), P::R8_SNORM);
    let mut env = SourceEnvironment(SnapshotEnvironment::new(
        EnvironmentMetadata {
            texture_bound_buffer: CbufIndex(2),
            ..Default::default()
        },
        observations,
    ));
    let mut h = host();
    h.support_snorm_render_buffer = false;
    texture_pass(
        &mut ir,
        &mut p,
        &mut env,
        &h,
        &mut EnvironmentCaches::default(),
    )
    .unwrap();
    assert_ne!(ir.inst(id(sample)).unwrap().arg(1).unwrap(), coords);
    let converted = id(ir.resolve(fetch).unwrap());
    assert_eq!(
        ir.inst(converted).unwrap().opcode(),
        O::CompositeConstructF32x4
    );
    let mul = ir
        .inst(id(ir.inst(converted).unwrap().arg(0).unwrap()))
        .unwrap();
    assert_eq!(mul.opcode(), O::FPMul32);
    assert_eq!(mul.arg(1).unwrap(), Value::f32(1.0 / 127.0));
    assert_eq!(p.info.texture_buffer_descriptors.len(), 1);
    assert_eq!(
        ir.block(block)
            .unwrap()
            .instructions()
            .iter()
            .filter(|&&i| ir.inst(i).unwrap().opcode() == O::ImageFetch)
            .count(),
        1
    );
    remove_identities(&mut ir, &p).unwrap();
    verify_program(&ir, &p).unwrap();
}
#[test]
fn storage_tracking_prefers_nvn_and_keeps_untracked_global_access() {
    let (mut ir, mut p, block) = program();
    let mut e = Emitter::new(&mut ir, block).unwrap();
    let nvn = e
        .get_cbuf(Value::U32(0), Value::U32(0x110), 32, false)
        .unwrap();
    let other = e.get_cbuf(Value::U32(3), Value::U32(8), 32, false).unwrap();
    let low = e.i_add(nvn, other).unwrap();
    let pair = e
        .emit(O::CompositeConstructU32x2, &[low, Value::U32(0)], 0)
        .unwrap();
    let address = e.emit(O::PackUint2x32, &[pair], 0).unwrap();
    let address = e.emit(O::IAdd64, &[address, Value::U64(12)], 0).unwrap();
    let load = e.emit(O::LoadGlobal32, &[address], 0).unwrap();
    e.emit(O::WriteGlobal32, &[address, Value::U32(42)], 0)
        .unwrap();
    let untracked = e.emit(O::LoadGlobal32, &[Value::U64(123)], 0).unwrap();
    global_memory_to_storage(&mut ir, &mut p, &host()).unwrap();
    let d = &p.info.storage_buffers_descriptors[0];
    assert_eq!(d.cbuf_index, CbufIndex(0));
    assert_eq!(d.cbuf_offset, ByteOffset(0x110));
    assert!(d.is_written);
    let load = ir.inst(id(ir.resolve(load).unwrap())).unwrap();
    assert_eq!(load.opcode(), O::LoadStorage32);
    assert_eq!(load.arg(0).unwrap(), Value::U32(0));
    let sub = ir.inst(id(load.arg(1).unwrap())).unwrap();
    assert_eq!(sub.opcode(), O::ISub32);
    let aligned = ir.inst(id(sub.arg(1).unwrap())).unwrap();
    assert_eq!(aligned.opcode(), O::BitwiseAnd32);
    assert_eq!(aligned.arg(1).unwrap(), Value::U32(!255));
    assert_eq!(ir.inst(id(untracked)).unwrap().opcode(), O::LoadGlobal32);
    remove_identities(&mut ir, &p).unwrap();
    verify_program(&ir, &p).unwrap();
}
#[test]
fn dead_resource_reads_disappear_but_stores_and_their_sizes_survive() {
    let (mut ir, mut p, block) = program();
    let mut e = Emitter::new(&mut ir, block).unwrap();
    e.get_cbuf(Value::U32(7), Value::U32(200), 32, false)
        .unwrap();
    let read = e
        .get_cbuf(Value::U32(2), Value::U32(17), 32, false)
        .unwrap();
    e.emit(O::WriteStorage32, &[Value::U32(0), Value::U32(0), read], 0)
        .unwrap();
    e.emit(O::Barrier, &[], 0).unwrap();
    dead_code_elimination(&mut ir, &p).unwrap();
    collect_shader_info(&env(&[], &[], &[]), &ir, &mut p).unwrap();
    assert_eq!(p.info.constant_buffer_mask, 1 << 2);
    assert_eq!(p.info.constant_buffer_used_sizes[2], 32);
    assert_eq!(p.info.constant_buffer_used_sizes[7], 0);
    assert!(
        ir.block(block)
            .unwrap()
            .instructions()
            .iter()
            .any(|&i| ir.inst(i).unwrap().opcode() == O::Barrier)
    );
    verify_program(&ir, &p).unwrap();
}

#[test]
fn dynamic_cbufs_mark_all_banks_and_nvn_slots_but_static_reads_remain_bounded() {
    let (mut ir, mut p, block) = program();
    let mut e = Emitter::new(&mut ir, block).unwrap();
    let index = e.emit(O::UndefU32, &[], 0).unwrap();
    e.get_cbuf(index, Value::U32(12), 16, false).unwrap();
    collect_shader_info(&env(&[], &[], &[]), &ir, &mut p).unwrap();
    assert!(p.info.uses_cbuf_indirect);
    assert_eq!(p.info.constant_buffer_mask, (1 << 14) - 1);
    assert_eq!(p.info.nvn_buffer_used, u16::MAX);
    assert_eq!(&p.info.constant_buffer_used_sizes[..14], &[0x10000; 14]);
    assert_eq!(
        p.info.used_indirect_cbuf_types,
        nixe_shader_recompiler::ir::Type::U16
    );
    let (mut ir, mut p, block) = program();
    Emitter::new(&mut ir, block)
        .unwrap()
        .get_cbuf(Value::U32(0), Value::U32(0x338), 32, false)
        .unwrap();
    collect_shader_info(&env(&[], &[], &[]), &ir, &mut p).unwrap();
    assert_eq!(p.info.nvn_buffer_base, 0x310);
    assert_eq!(p.info.nvn_buffer_used, 4);
    assert_eq!(p.info.constant_buffer_used_sizes[0], 0x340);
}
#[test]
fn image_descriptor_access_flags_merge_without_aliasing_different_formats() {
    use nixe_shader_recompiler::{
        artifact::{ImageDescriptor, ShaderInfo},
        passes::join_texture_info,
        types::ImageFormat,
    };
    let mut base = ShaderInfo::default();
    let mut source = ShaderInfo::default();
    let a = ImageDescriptor {
        texture_type: T::Color2D,
        cbuf_index: CbufIndex(2),
        cbuf_offset: ByteOffset(8),
        count: 1,
        size_shift: 3,
        is_read: true,
        ..Default::default()
    };
    base.image_descriptors.push(a.clone());
    let mut b = a.clone();
    b.is_read = false;
    b.is_written = true;
    b.is_integer = true;
    source.image_descriptors.push(b);
    let mut c = a;
    c.format = ImageFormat::R32_UINT;
    source.image_descriptors.push(c);
    join_texture_info(&mut base, &source);
    assert_eq!(base.image_descriptors.len(), 2);
    assert!(
        base.image_descriptors[0].is_read
            && base.image_descriptors[0].is_written
            && base.image_descriptors[0].is_integer
    );
}
#[test]
fn position_transform_observes_viewport_only_for_vertex_b_and_rewrites_xy() {
    use nixe_shader_recompiler::{ir::Attribute, passes::position};
    for stage in [Stage::VertexA, Stage::VertexB] {
        let (mut ir, mut p, block) = program();
        p.stage = stage;
        let mut e = Emitter::new(&mut ir, block).unwrap();
        let x = e
            .emit(
                O::SetAttribute,
                &[
                    Value::Attribute(Attribute::PositionX),
                    Value::f32(10.0),
                    Value::U32(0),
                ],
                0,
            )
            .unwrap();
        let z = e
            .emit(
                O::SetAttribute,
                &[
                    Value::Attribute(Attribute::PositionZ),
                    Value::f32(20.0),
                    Value::U32(0),
                ],
                0,
            )
            .unwrap();
        let obs = Observations {
            viewport_transform_state: Some(0),
            ..Default::default()
        };
        let mut source = SnapshotEnvironment::new(
            EnvironmentMetadata {
                stage,
                ..Default::default()
            },
            obs,
        );
        let mut recorder = RecordingEnvironment::new(&mut source);
        position(&mut recorder, &mut ir, &mut p).unwrap();
        assert_eq!(
            recorder.observations().viewport_transform_state,
            if stage == Stage::VertexB {
                Some(0)
            } else {
                None
            }
        );
        assert_eq!(p.info.uses_render_area, stage == Stage::VertexB);
        assert_eq!(ir.inst(id(z)).unwrap().arg(1).unwrap(), Value::f32(20.0));
        if stage == Stage::VertexB {
            let value = ir
                .inst(id(ir.inst(id(x)).unwrap().arg(1).unwrap()))
                .unwrap();
            assert_eq!(value.opcode(), O::FPFma32);
            assert_eq!(value.arg(2).unwrap(), Value::f32(-1.0));
        }
        verify_program(&ir, &p).unwrap();
    }
}

#[test]
fn fallback_nvn_slots_join_by_offset_and_inherit_global_writes() {
    use nixe_shader_recompiler::{
        artifact::StorageBufferDescriptor, passes::add_nvn_storage_buffers,
    };
    let (_, mut p, _) = program();
    p.info.uses_global_memory = true;
    p.info.stores_global_memory = true;
    p.info.nvn_buffer_used = 5;
    p.info
        .storage_buffers_descriptors
        .push(StorageBufferDescriptor {
            cbuf_index: CbufIndex(3),
            cbuf_offset: ByteOffset(0x310),
            count: 1,
            is_written: false,
        })
        .unwrap();
    add_nvn_storage_buffers(&mut p).unwrap();
    assert_eq!(p.info.storage_buffers_descriptors.len(), 2);
    assert_eq!(
        p.info.storage_buffers_descriptors[0].cbuf_index,
        CbufIndex(3)
    );
    assert_eq!(
        p.info.storage_buffers_descriptors[1].cbuf_offset,
        ByteOffset(0x330)
    );
    assert!(
        p.info
            .storage_buffers_descriptors
            .iter()
            .all(|d| d.is_written)
    );
}
