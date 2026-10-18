use nixe_shader_recompiler::{
    Profile, RuntimeInfo,
    artifact::{
        Bindings, ConstantBufferDescriptor, HostUniformLayout, ImageDescriptor,
        StorageBufferDescriptor, TextureDescriptor,
    },
    ir::{Attribute, Program, Type},
    profile::TransformFeedbackVarying,
    spirv::{BufferKind, EmitContext, Id},
    types::{AttributeType, ByteOffset, CbufIndex, ImageFormat, Stage, TextureType},
};
use spirv::{Decoration as D, Op, StorageClass as S};
fn instructions(words: &[u32]) -> Vec<(Op, &[u32])> {
    let mut pos = 5;
    let mut out = Vec::new();
    while pos < words.len() {
        let count = (words[pos] >> 16) as usize;
        out.push((
            Op::from_u32(words[pos] & 65535).unwrap(),
            &words[pos + 1..pos + count],
        ));
        pos += count;
    }
    out
}
fn profile() -> Profile {
    Profile {
        supported_spirv: 0x00010500,
        unified_descriptor_binding: true,
        support_descriptor_aliasing: true,
        ..Default::default()
    }
}
fn decorated(inst: &[(Op, &[u32])], id: Id, decoration: D, value: u32) -> bool {
    inst.iter()
        .any(|(op, args)| *op == Op::Decorate && *args == [id.0, decoration as u32, value])
}
#[test]
fn declarations_use_pipeline_binding_counters_and_keep_array_binding_count() {
    let mut p = Program::new(Stage::Compute);
    p.info.used_constant_buffer_types = Type::U32;
    p.info.used_storage_buffer_types = Type::U32;
    p.info
        .constant_buffer_descriptors
        .push(ConstantBufferDescriptor {
            index: CbufIndex(2),
            count: 1,
        })
        .unwrap();
    p.info
        .storage_buffers_descriptors
        .push(StorageBufferDescriptor {
            cbuf_index: CbufIndex(0),
            cbuf_offset: ByteOffset(0x310),
            count: 1,
            is_written: true,
        })
        .unwrap();
    p.info.texture_descriptors.push(TextureDescriptor {
        texture_type: TextureType::Color2D,
        count: 12,
        ..Default::default()
    });
    p.info.image_descriptors.push(ImageDescriptor {
        texture_type: TextureType::Color2D,
        format: ImageFormat::R32_UINT,
        count: 3,
        is_written: true,
        is_integer: true,
        ..Default::default()
    });
    let profile = profile();
    let runtime = RuntimeInfo::default();
    let mut bindings = Bindings {
        unified: 7,
        texture_scaling_index: 3,
        image_scaling_index: 2,
        ..Default::default()
    };
    let ctx = EmitContext::new(&profile, &runtime, &p, &mut bindings).unwrap();
    assert_eq!(bindings.unified, 11);
    assert_eq!(bindings.texture_scaling_index, 4);
    assert_eq!(bindings.image_scaling_index, 3);
    assert_eq!(ctx.texture_scaling_index, 3);
    assert_eq!(ctx.image_scaling_index, 2);
    assert_eq!(
        ctx.resource_bindings
            .iter()
            .map(|b| b.binding.0)
            .collect::<Vec<_>>(),
        [7, 8, 9, 10]
    );
    assert_eq!(
        ctx.resource_bindings
            .iter()
            .map(|b| b.count)
            .collect::<Vec<_>>(),
        [1, 1, 12, 1]
    );
    let words = ctx.builder.assemble().unwrap();
    let inst = instructions(&words);
    assert!(decorated(&inst, ctx.textures[0].id, D::Binding, 9));
    let variable = inst
        .iter()
        .find(|(op, a)| *op == Op::Variable && a[1] == ctx.textures[0].id.0)
        .unwrap()
        .1;
    let pointer = inst
        .iter()
        .find(|(op, a)| *op == Op::TypePointer && a[0] == variable[0])
        .unwrap()
        .1;
    assert!(
        inst.iter()
            .any(|(op, a)| *op == Op::TypeArray && a[0] == pointer[2])
    );
    let next = EmitContext::new(&profile, &runtime, &p, &mut bindings).unwrap();
    assert_eq!(next.resource_bindings[0].binding.0, 11);
}
#[test]
fn narrow_buffer_views_share_bindings_and_have_exact_strides() {
    let mut p = Program::new(Stage::Compute);
    p.info.uses_int8 = true;
    p.info.uses_int16 = true;
    p.info.used_constant_buffer_types = Type::U8 | Type::U16 | Type::U32x2;
    p.info
        .constant_buffer_descriptors
        .push(ConstantBufferDescriptor {
            index: CbufIndex(0),
            count: 1,
        })
        .unwrap();
    let profile = Profile {
        support_int8: true,
        support_int16: true,
        support_uniform_and_storage_buffer_8bit: true,
        support_uniform_and_storage_buffer_16bit: true,
        ..profile()
    };
    let runtime = RuntimeInfo::default();
    let mut bindings = Bindings::default();
    let ctx = EmitContext::new(&profile, &runtime, &p, &mut bindings).unwrap();
    assert_eq!(bindings.unified, 1);
    let words = ctx.builder.assemble().unwrap();
    let inst = instructions(&words);
    for kind in [
        BufferKind::U8,
        BufferKind::S8,
        BufferKind::U16,
        BufferKind::S16,
        BufferKind::U32x2,
    ] {
        assert!(decorated(&inst, ctx.cbufs[0][kind as usize], D::Binding, 0));
    }
    for stride in [1, 2, 8] {
        assert!(
            inst.iter()
                .any(|(op, a)| *op == Op::Decorate && a[1..] == [D::ArrayStride as u32, stride])
        );
    }
    assert_eq!(ctx.cbufs[0][BufferKind::U32 as usize], Id::NONE);
    assert_eq!(ctx.types.u64, Id::NONE);
    assert_eq!(ctx.types.f64, [Id::NONE; 4]);
}
#[test]
fn host_uniform_abi_keeps_independent_overlapping_blocks() {
    for stage in [Stage::VertexB, Stage::Compute] {
        let mut p = Program::new(stage);
        p.info.uses_rescaling_uniform = true;
        p.info.uses_render_area = true;
        let profile = profile();
        let runtime = RuntimeInfo::default();
        let ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
        assert_eq!(
            ctx.host_uniforms,
            [
                HostUniformLayout::RescalingPushConstants {
                    texture_mask_offset: 0,
                    texture_mask_words: 4,
                    image_mask_offset: 16,
                    image_mask_words: 2,
                    down_factor_offset: if stage == Stage::Compute {
                        None
                    } else {
                        Some(24)
                    }
                },
                HostUniformLayout::RenderAreaPushConstants {
                    offset: 0,
                    size: 16
                }
            ]
        );
        let words = ctx.builder.assemble().unwrap();
        let inst = instructions(&words);
        assert_eq!(
            inst.iter()
                .filter(|(op, a)| *op == Op::Variable && a[2] == S::PushConstant as u32)
                .count(),
            2
        );
        for offset in [0, 16] {
            assert!(
                inst.iter().any(
                    |(op, a)| *op == Op::MemberDecorate && a[2..] == [D::Offset as u32, offset]
                )
            );
        }
    }
}
#[test]
fn geometry_inputs_are_arrays_and_xfb_outputs_split_components() {
    let mut p = Program::new(Stage::Geometry);
    p.info.loads.set(Attribute::generic(0, 0).unwrap(), true);
    p.info.stores.set(Attribute::generic(0, 0).unwrap(), true);
    let mut runtime = RuntimeInfo::default();
    runtime.previous_stage_stores.mask = [u64::MAX; 8];
    runtime.generic_input_types[0] = AttributeType::SignedScaled;
    runtime.xfb_count = 36;
    runtime.xfb_varyings[32] = TransformFeedbackVarying {
        components: 2,
        buffer: 1,
        stream: 2,
        stride: 16,
        offset: 4,
    };
    let profile = Profile {
        support_scaled_attributes: false,
        ..profile()
    };
    let ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    assert_eq!(
        ctx.input_generics[0].load,
        nixe_shader_recompiler::spirv::InputLoad::SignedToFloat
    );
    assert_eq!(ctx.output_generics[0][0].components, 2);
    assert_eq!(ctx.output_generics[0][2].first, 2);
    assert_ne!(ctx.output_generics[0][0].id, ctx.output_generics[0][2].id);
    let words = ctx.builder.assemble().unwrap();
    let inst = instructions(&words);
    assert!(decorated(
        &inst,
        ctx.output_generics[0][0].id,
        D::XfbBuffer,
        1
    ));
    assert!(decorated(&inst, ctx.output_generics[0][0].id, D::Stream, 2));
    assert!(decorated(
        &inst,
        ctx.output_generics[0][2].id,
        D::Component,
        2
    ));
    let input = inst
        .iter()
        .find(|(op, a)| *op == Op::Variable && a[1] == ctx.input_generics[0].id.0)
        .unwrap()
        .1;
    let pointer = inst
        .iter()
        .find(|(op, a)| *op == Op::TypePointer && a[0] == input[0])
        .unwrap()
        .1;
    assert!(
        inst.iter()
            .any(|(op, a)| *op == Op::TypeArray && a[0] == pointer[2])
    );
}
#[test]
fn indirect_cbuf_helpers_switch_over_fourteen_banks_with_zero_default_bank() {
    let mut p = Program::new(Stage::Compute);
    p.info.uses_cbuf_indirect = true;
    p.info.used_indirect_cbuf_types = Type::U32;
    for index in 0..14 {
        p.info
            .constant_buffer_descriptors
            .push(ConstantBufferDescriptor {
                index: CbufIndex(index),
                count: 1,
            })
            .unwrap();
    }
    let profile = profile();
    let runtime = RuntimeInfo::default();
    let ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    assert_ne!(ctx.objects.load_const_func_u32, Id::NONE);
    let words = ctx.builder.assemble().unwrap();
    let inst = instructions(&words);
    let switch = inst.iter().find(|(op, _)| *op == Op::Switch).unwrap().1;
    assert_eq!(switch.len(), 30);
    assert_eq!(switch[1], switch[3]);
    assert_eq!(switch[2], 0);
    assert_eq!(switch[28], 13);
    assert_eq!(
        inst.iter().filter(|(op, _)| *op == Op::ReturnValue).count(),
        14
    );
}
#[test]
fn shared_layout_and_narrow_cas_are_selected_from_capabilities() {
    for explicit in [false, true] {
        let mut p = Program::new(Stage::Compute);
        p.execution.shared_memory_size = 65;
        p.info.uses_int8 = true;
        let profile = Profile {
            support_int8: true,
            support_int16: true,
            support_explicit_workgroup_layout: explicit,
            support_workgroup_layout_8bit_access: explicit,
            ..profile()
        };
        let runtime = RuntimeInfo::default();
        let ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
        assert_eq!(ctx.uses_explicit_workgroup_layout, explicit);
        assert_eq!(ctx.objects.shared_store_u8_func == Id::NONE, explicit);
        let words = ctx.builder.assemble().unwrap();
        let inst = instructions(&words);
        assert_eq!(
            inst.iter().any(|(op, _)| *op == Op::AtomicCompareExchange),
            !explicit
        );
        assert_eq!(ctx.objects.shared_memory_u8 != Id::NONE, explicit);
    }
}

#[test]
fn global_range_helpers_use_aligned_base_and_all_three_element_widths() {
    for alias in [false, true] {
        let mut p = Program::new(Stage::Compute);
        p.info.uses_global_memory = true;
        p.info.uses_int64 = true;
        p.info.nvn_buffer_used = 1;
        p.info.used_constant_buffer_types = Type::U32 | Type::U32x2;
        p.info.used_storage_buffer_types = Type::U32 | Type::U32x2 | Type::U32x4;
        p.info
            .constant_buffer_descriptors
            .push(ConstantBufferDescriptor {
                index: CbufIndex(0),
                count: 1,
            })
            .unwrap();
        p.info
            .storage_buffers_descriptors
            .push(StorageBufferDescriptor {
                cbuf_index: CbufIndex(0),
                cbuf_offset: ByteOffset(0x310),
                count: 1,
                is_written: true,
            })
            .unwrap();
        let profile = Profile {
            min_ssbo_alignment: 256,
            support_int64: true,
            support_descriptor_aliasing: alias,
            ..profile()
        };
        let runtime = RuntimeInfo::default();
        let result = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default());
        if !alias {
            assert!(
                result
                    .err()
                    .unwrap()
                    .message
                    .contains("undeclared typed buffer views")
            );
            continue;
        }
        let ctx = result.unwrap();
        assert_ne!(ctx.objects.load_global_func_u32x4, Id::NONE);
        assert_ne!(ctx.objects.write_global_func_u32x2, Id::NONE);
        let words = ctx.builder.assemble().unwrap();
        let inst = instructions(&words);
        assert_eq!(inst.iter().filter(|(op, _)| *op == Op::Function).count(), 6);
        assert_eq!(
            inst.iter()
                .filter(|(op, _)| *op == Op::UGreaterThanEqual)
                .count(),
            6
        );
        assert_eq!(
            inst.iter().filter(|(op, _)| *op == Op::ULessThan).count(),
            6
        );
        assert!(inst.iter().any(|(op, args)| *op == Op::Constant
            && args[0] == ctx.types.u64.0
            && args[2..] == [0xffffff00, 0xffffffff]));
    }
}
#[test]
fn unrelated_decorated_structures_do_not_share_type_identity() {
    let mut p = Program::new(Stage::Geometry);
    p.info.loads.set(Attribute::PositionX, true);
    p.info.uses_render_area = true;
    let profile = Profile {
        has_broken_spirv_position_input: true,
        ..profile()
    };
    let runtime = RuntimeInfo::default();
    let ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    let words = ctx.builder.assemble().unwrap();
    let inst = instructions(&words);
    let builtin_struct = inst
        .iter()
        .find(|(op, a)| *op == Op::MemberDecorate && a[2] == D::BuiltIn as u32)
        .unwrap()
        .1[0];
    let push_struct = inst
        .iter()
        .find(|(op, a)| *op == Op::MemberDecorate && a[2] == D::Offset as u32)
        .unwrap()
        .1[0];
    assert_ne!(builtin_struct, push_struct);
}

#[test]
fn atomic_helpers_use_loop_merges_and_storage_pointer_parameters() {
    let mut p = Program::new(Stage::Compute);
    p.execution.shared_memory_size = 32;
    p.info.uses_shared_increment = true;
    p.info.uses_global_decrement = true;
    p.info
        .storage_buffers_descriptors
        .push(StorageBufferDescriptor {
            cbuf_index: CbufIndex(0),
            cbuf_offset: ByteOffset(0x310),
            count: 1,
            is_written: true,
        })
        .unwrap();
    let profile = profile();
    let runtime = RuntimeInfo::default();
    let ctx = EmitContext::new(&profile, &runtime, &p, &mut Bindings::default()).unwrap();
    assert_ne!(ctx.objects.increment_cas_shared, Id::NONE);
    assert_ne!(ctx.objects.decrement_cas_ssbo, Id::NONE);
    let words = ctx.builder.assemble().unwrap();
    let inst = instructions(&words);
    assert_eq!(
        inst.iter().filter(|(op, _)| *op == Op::LoopMerge).count(),
        2
    );
    assert_eq!(
        inst.iter()
            .filter(|(op, _)| *op == Op::AtomicCompareExchange)
            .count(),
        2
    );
    assert!(inst.iter().any(|(op, a)| *op == Op::FunctionParameter
        && a[0] == ctx.storage_types[BufferKind::U32 as usize].array.0));
    for (op, a) in &inst {
        if *op == Op::AtomicCompareExchange {
            assert_eq!(a[4], ctx.constants.u32_zero.0);
            assert_eq!(a[5], ctx.constants.u32_zero.0);
        }
    }
}
