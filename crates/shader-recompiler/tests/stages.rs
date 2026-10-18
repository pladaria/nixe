use nixe_shader_recompiler::{
    HostTranslateInfo, RuntimeInfo,
    arena::{BlockId, InstId},
    artifact::{StageProvenance, TextureDescriptor},
    environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
    ir::{Attribute, Emitter, Ir, Opcode as O, Program, SyntaxNode, Type, Value},
    maxwell::header::ProgramHeader,
    passes::*,
    profile::ResolutionScaling,
    types::{
        AttributeType, ByteOffset, CbufIndex, CompareFunction, Interpolation, OutputTopology,
        Stage, TessPrimitive, TessSpacing, TextureType,
    },
};
fn program(ir: &mut Ir, stage: Stage) -> (Program, BlockId) {
    let block = ir.new_block().unwrap();
    let mut p = Program::new(stage);
    p.blocks.push(block);
    p.post_order.push(block);
    p.syntax = vec![SyntaxNode::Block(block), SyntaxNode::Return];
    (p, block)
}
fn id(value: Value) -> InstId {
    let Value::Inst(id) = value else { panic!() };
    id
}
fn environment(stage: Stage, bytes: [u8; 80]) -> SnapshotEnvironment {
    SnapshotEnvironment::new(
        EnvironmentMetadata {
            stage,
            sph: ProgramHeader::from_bytes(&bytes),
            ..Default::default()
        },
        Observations::default(),
    )
}
fn count(ir: &Ir, p: &Program, op: O) -> usize {
    p.blocks
        .iter()
        .flat_map(|b| ir.block(*b).unwrap().instructions())
        .filter(|&&i| ir.inst(i).unwrap().opcode() == op)
        .count()
}
#[test]
fn interpolation_is_fragment_local_and_rejects_component_disagreement() {
    let mut ir = Ir::new().unwrap();
    let (mut p, _) = program(&mut ir, Stage::Fragment);
    let mut bytes = [0; 80];
    bytes[24] = 0x55;
    bytes[25] = 0xaa;
    bytes[26] = 0xff;
    collect_interpolation(&environment(Stage::Fragment, bytes), &mut p).unwrap();
    assert_eq!(
        &p.info.interpolation[..3],
        &[
            Interpolation::Flat,
            Interpolation::Smooth,
            Interpolation::NoPerspective
        ]
    );
    bytes[24] = 0x09;
    assert!(collect_interpolation(&environment(Stage::Fragment, bytes), &mut p).is_err());
    p.stage = Stage::VertexB;
    collect_interpolation(&environment(Stage::VertexB, bytes), &mut p).unwrap();
}
#[test]
fn legacy_outputs_allocate_whole_vectors_and_inputs_inherit_mapping() {
    let mut ir = Ir::new().unwrap();
    let (mut previous, b) = program(&mut ir, Stage::VertexB);
    let mut e = Emitter::new(&mut ir, b).unwrap();
    let output = e
        .set_attribute(
            Attribute::ColorFrontDiffuseR,
            Value::f32(1.0),
            Value::U32(0),
        )
        .unwrap();
    previous
        .info
        .stores
        .set(Attribute::ColorFrontDiffuseR, true);
    previous
        .info
        .stores
        .set(Attribute::generic(0, 2).unwrap(), true);
    convert_legacy_to_generic(&mut ir, &mut previous, &RuntimeInfo::default()).unwrap();
    let generic = Attribute::generic(1, 0).unwrap();
    assert_eq!(
        ir.inst(id(output)).unwrap().arg(0).unwrap(),
        Value::Attribute(generic)
    );
    assert_eq!(previous.info.legacy_stores_mapping.len(), 4);
    assert!(previous.info.stores.get(Attribute::ColorFrontDiffuseR));
    let (mut next, b) = program(&mut ir, Stage::Fragment);
    let input = Emitter::new(&mut ir, b)
        .unwrap()
        .get_attribute(Attribute::ColorFrontDiffuseR)
        .unwrap();
    next.info.loads.set(Attribute::ColorFrontDiffuseR, true);
    let runtime = make_runtime_info(
        &next,
        Some(&previous),
        false,
        &GraphicsRuntimeState::default(),
    );
    convert_legacy_to_generic(&mut ir, &mut next, &runtime).unwrap();
    assert_eq!(
        ir.inst(id(input)).unwrap().arg(0).unwrap(),
        Value::Attribute(generic)
    );
    assert!(next.info.loads.get(generic));
}
#[test]
fn layer_remap_and_generated_geometry_preserve_interfaces_and_vertex_count() {
    let mut ir = Ir::new().unwrap();
    let (mut source, b) = program(&mut ir, Stage::VertexB);
    let layer_store = Emitter::new(&mut ir, b)
        .unwrap()
        .set_attribute(Attribute::Layer, Value::f32(3.0), Value::U32(0))
        .unwrap();
    source.info.stores.set(Attribute::Layer, true);
    source
        .info
        .stores
        .set(Attribute::generic(0, 0).unwrap(), true);
    layer(&mut ir, &mut source, &HostTranslateInfo::default()).unwrap();
    assert_eq!(
        source.info.emulated_layer,
        Attribute::generic(1, 0).unwrap()
    );
    assert!(source.info.requires_layer_emulation);
    assert!(!source.info.stores.get(Attribute::Layer));
    assert_eq!(
        ir.inst(id(layer_store)).unwrap().arg(0).unwrap(),
        Value::Attribute(source.info.emulated_layer)
    );
    let generated =
        generate_geometry_passthrough(&mut ir, &source, OutputTopology::TriangleStrip).unwrap();
    assert_eq!(
        generated.execution.provenance,
        StageProvenance::LayerPassthrough
    );
    assert_eq!(generated.execution.output_vertices, 3);
    assert_eq!(count(&ir, &generated, O::EmitVertex), 3);
    assert_eq!(count(&ir, &generated, O::EndPrimitive), 1);
    assert!(generated.info.loads.get(source.info.emulated_layer));
    assert!(!generated.info.stores.get(source.info.emulated_layer));
    assert!(generated.info.stores.get(Attribute::Layer));
    verify_program(&ir, &generated).unwrap();
}
#[test]
fn native_and_expanded_geometry_passthrough_keep_mask_and_culling() {
    for native in [false, true] {
        let mut ir = Ir::new().unwrap();
        let (mut p, b) = program(&mut ir, Stage::Geometry);
        let mut e = Emitter::new(&mut ir, b).unwrap();
        e.set_attribute(Attribute::ViewportMask, Value::f32(0.0), Value::U32(0))
            .unwrap();
        e.emit(O::Epilogue, &[], 0).unwrap();
        let mut bytes = [0; 80];
        bytes[3] = 1;
        bytes[11] = 2;
        bytes[15] = 7;
        bytes[16] = 9;
        let mut metadata = EnvironmentMetadata {
            stage: Stage::Geometry,
            sph: ProgramHeader::from_bytes(&bytes),
            gp_passthrough_mask: [u32::MAX; 8],
            ..Default::default()
        };
        for a in [Attribute::PositionX, Attribute::generic(3, 0).unwrap()] {
            metadata.gp_passthrough_mask[a.index() / 32] &= !(1 << (a.index() % 32));
        }
        let env = SnapshotEnvironment::new(metadata, Observations::default());
        configure_stage(
            &mut ir,
            &mut p,
            &env,
            &HostTranslateInfo {
                support_geometry_shader_passthrough: native,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(p.execution.is_geometry_passthrough);
        assert_eq!(p.execution.invocations, 2);
        assert!(p.info.passthrough.get(Attribute::PositionX));
        assert_eq!(count(&ir, &p, O::EmitVertex), if native { 0 } else { 3 });
        if !native {
            assert_eq!(count(&ir, &p, O::FindUMsb32), 1);
            assert_eq!(count(&ir, &p, O::SelectF32), 12);
        }
        verify_program(&ir, &p).unwrap();
    }
}
#[test]
fn dual_vertex_merge_removes_first_markers_and_joins_resources() {
    let mut ir = Ir::new().unwrap();
    let (mut a, ba) = program(&mut ir, Stage::VertexA);
    let (mut b, bb) = program(&mut ir, Stage::VertexB);
    for block in [ba, bb] {
        let mut e = Emitter::new(&mut ir, block).unwrap();
        e.emit(O::Prologue, &[], 0).unwrap();
        e.set_attribute(Attribute::PositionX, Value::f32(1.0), Value::U32(0))
            .unwrap();
        e.emit(O::Epilogue, &[], 0).unwrap();
    }
    a.execution.local_memory_size = 128;
    b.execution.local_memory_size = 64;
    let descriptor = TextureDescriptor {
        texture_type: TextureType::Color2D,
        cbuf_index: CbufIndex(2),
        cbuf_offset: ByteOffset(8),
        count: 1,
        size_shift: 3,
        ..Default::default()
    };
    a.info.texture_descriptors.push(descriptor.clone());
    b.info.texture_descriptors.push(descriptor);
    b.info.texture_descriptors[0].is_integer = true;
    let p = merge_dual_vertex(&mut ir, a, b, &environment(Stage::VertexB, [0; 80]), true).unwrap();
    assert_eq!(p.blocks, [ba, bb]);
    assert_eq!(p.post_order, [bb, ba]);
    assert_eq!(
        p.syntax
            .iter()
            .filter(|s| matches!(s, SyntaxNode::Return))
            .count(),
        1
    );
    assert_eq!(count(&ir, &p, O::Prologue), 1);
    assert_eq!(count(&ir, &p, O::Epilogue), 1);
    assert_eq!(p.execution.local_memory_size, 128);
    assert_eq!(p.execution.provenance, StageProvenance::MergedVertex);
    assert_eq!(p.info.texture_descriptors.len(), 1);
    assert!(p.info.texture_descriptors[0].is_integer);
}
#[test]
fn runtime_settings_are_selected_per_stage_without_leaking() {
    let mut ir = Ir::new().unwrap();
    let (mut previous, _) = program(&mut ir, Stage::Geometry);
    previous.execution.is_geometry_passthrough = true;
    previous
        .info
        .passthrough
        .set(Attribute::generic(5, 0).unwrap(), true);
    let mut state = GraphicsRuntimeState {
        point_size: 7.0,
        ndc_minus_one_to_one: true,
        y_negate: true,
        tess_clockwise: true,
        tess_primitive: TessPrimitive::Quads,
        tess_spacing: TessSpacing::FractionalOdd,
        alpha_test_func: CompareFunction::Greater,
        alpha_test_reference: 0.25,
        xfb_enabled: true,
        support_transform_feedback: true,
        xfb_count: 2,
        use_integer_fragment_outputs: true,
        ..Default::default()
    };
    state.vertex_input_types[0] = AttributeType::UnsignedInt;
    state.color_output_types[0] = AttributeType::SignedInt;
    for stage in [
        Stage::VertexB,
        Stage::TessellationEval,
        Stage::Geometry,
        Stage::Fragment,
    ] {
        let (mut p, _) = program(&mut ir, stage);
        p.execution.output_topology = Some(OutputTopology::PointList);
        let r = make_runtime_info(&p, Some(&previous), false, &state);
        assert!(r.previous_stage_stores.generic(5));
        assert!(r.y_negate);
        assert_eq!(
            r.fixed_state_point_size,
            if matches!(stage, Stage::VertexB | Stage::Geometry) {
                Some(7.0)
            } else {
                None
            }
        );
        assert_eq!(
            r.xfb_count,
            if matches!(stage, Stage::VertexB | Stage::Geometry) {
                2
            } else {
                0
            }
        );
        assert_eq!(
            r.convert_depth_mode,
            matches!(stage, Stage::VertexB | Stage::Geometry)
        );
        assert_eq!(
            r.alpha_test_func,
            if stage == Stage::Fragment {
                Some(CompareFunction::Greater)
            } else {
                None
            }
        );
        assert_eq!(
            r.tess_primitive,
            if stage == Stage::TessellationEval {
                TessPrimitive::Quads
            } else {
                TessPrimitive::Isolines
            }
        );
        assert_eq!(
            r.generic_input_types[0],
            if stage == Stage::VertexB {
                AttributeType::UnsignedInt
            } else {
                AttributeType::Float
            }
        );
    }
    let (vertex, _) = program(&mut ir, Stage::VertexB);
    let r = make_runtime_info(&vertex, None, true, &state);
    assert!(!r.convert_depth_mode);
    assert_eq!(r.xfb_count, 0);
    assert_eq!(r.previous_stage_stores.mask, [u64::MAX; 8]);
}
#[test]
fn geometry_compaction_is_capability_gated_and_neutralizes_ballot_predicate() {
    for enabled in [false, true] {
        let mut ir = Ir::new().unwrap();
        let (mut p, b) = program(&mut ir, Stage::Geometry);
        p.execution.invocations = 3;
        let mut e = Emitter::new(&mut ir, b).unwrap();
        let pred = e.emit(O::UndefU1, &[], 0).unwrap();
        let ballot = e.emit(O::SubgroupBallot, &[pred], 0).unwrap();
        let amount = e.emit(O::BitCount32, &[ballot], 0).unwrap();
        let atomic = e
            .emit(
                O::StorageAtomicIAdd32,
                &[Value::U32(0), Value::U32(0), amount],
                0,
            )
            .unwrap();
        geometry_compaction(
            &mut ir,
            &p,
            &HostTranslateInfo {
                single_lane_geometry_subgroups: enabled,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(count(&ir, &p, O::StorageAtomicUMax32), usize::from(enabled));
        if enabled {
            assert_eq!(ir.resolve(pred).unwrap(), Value::U1(true));
            assert_ne!(ir.resolve(atomic).unwrap(), atomic);
            assert_eq!(count(&ir, &p, O::InvocationId), 1);
        }
        remove_identities(&mut ir, &p).unwrap();
        verify_program(&ir, &p).unwrap();
    }
}
#[test]
fn rescaling_preserves_array_layer_and_uses_resource_specific_predicate() {
    for stage in [Stage::Compute, Stage::Fragment] {
        let mut ir = Ir::new().unwrap();
        let (p, b) = program(&mut ir, stage);
        let mut e = Emitter::new(&mut ir, b).unwrap();
        let coords = e
            .composite_construct(&[Value::U32(2), Value::U32(3), Value::U32(9)])
            .unwrap();
        let mut flags = nixe_shader_recompiler::ir::TextureInstInfo::default();
        flags.set_texture_type(TextureType::ColorArray2D);
        flags.set_descriptor_index(4);
        let read = e
            .emit(O::ImageRead, &[Value::None, coords], flags.0)
            .unwrap();
        let settings = ResolutionScaling {
            active: true,
            up_scale: 3,
            down_shift: 1,
            up_factor: 1.5,
            down_factor: 2.0 / 3.0,
        };
        rescaling(&mut ir, &p, &settings).unwrap();
        let coords = ir
            .inst(id(ir.inst(id(read)).unwrap().arg(1).unwrap()))
            .unwrap();
        assert_eq!(coords.opcode(), O::CompositeConstructU32x3);
        let layer = ir.inst(id(coords.arg(2).unwrap())).unwrap();
        assert_eq!(layer.arg(1).unwrap(), Value::U32(2));
        assert_eq!(count(&ir, &p, O::IsImageScaled), 1);
        assert_eq!(count(&ir, &p, O::IsTextureScaled), 0);
        assert_eq!(count(&ir, &p, O::SelectU32), 2);
        assert_eq!(
            count(&ir, &p, O::FPFloor32),
            if stage == Stage::Fragment { 2 } else { 0 }
        );
        verify_program(&ir, &p).unwrap();
    }
}

#[test]
fn compaction_rewrites_shuffle_phi_inputs_and_stops_after_eight_edges() {
    for depth in [8, 9] {
        let mut ir = Ir::new().unwrap();
        let (mut p, atomic_block) = program(&mut ir, Stage::Geometry);
        let other = ir.new_block().unwrap();
        let merge = ir.new_block().unwrap();
        p.blocks = vec![atomic_block, other, merge];
        p.post_order = vec![merge, other, atomic_block];
        ir.add_branch(atomic_block, merge).unwrap();
        ir.add_branch(other, merge).unwrap();
        let mut e = Emitter::new(&mut ir, atomic_block).unwrap();
        let predicate = e.emit(O::UndefU1, &[], 0).unwrap();
        let mut amount = e.emit(O::SubgroupBallot, &[predicate], 0).unwrap();
        for _ in 0..depth {
            amount = e.emit(O::BitCount32, &[amount], 0).unwrap();
        }
        let atomic = e
            .emit(
                O::StorageAtomicIAdd32,
                &[Value::U32(0), Value::U32(0), amount],
                0,
            )
            .unwrap();
        let mut e = Emitter::new(&mut ir, other).unwrap();
        let shuffled = e
            .emit(
                O::ShuffleIndex,
                &[Value::U32(3), Value::U32(0), Value::U32(3), Value::U32(28)],
                0,
            )
            .unwrap();
        let phi = ir.new_phi(merge, Type::U32).unwrap();
        ir.add_phi_operand(phi, atomic_block, atomic).unwrap();
        ir.add_phi_operand(phi, other, shuffled).unwrap();
        geometry_compaction(
            &mut ir,
            &p,
            &HostTranslateInfo {
                single_lane_geometry_subgroups: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            count(&ir, &p, O::StorageAtomicUMax32),
            usize::from(depth == 8)
        );
        assert_eq!(
            ir.inst(phi).unwrap().arg(1).unwrap() == shuffled,
            depth == 9
        );
        remove_identities(&mut ir, &p).unwrap();
        verify_program(&ir, &p).unwrap();
    }
}
#[test]
fn vertex_tightening_skips_loops_and_vendor_pattern_keeps_its_literal_guard() {
    let mut ir = Ir::new().unwrap();
    let (mut p, b) = program(&mut ir, Stage::Geometry);
    p.execution.output_vertices = 32;
    let mut e = Emitter::new(&mut ir, b).unwrap();
    for _ in 0..3 {
        e.emit(O::EmitVertex, &[Value::U32(0)], 0).unwrap();
    }
    tighten_output_vertices(&ir, &mut p).unwrap();
    assert_eq!(p.execution.output_vertices, 3);
    p.execution.output_vertices = 32;
    p.syntax.insert(
        0,
        SyntaxNode::Loop {
            body: b,
            continue_block: b,
            merge: b,
        },
    );
    tighten_output_vertices(&ir, &mut p).unwrap();
    assert_eq!(p.execution.output_vertices, 32);
    let (p, b) = program(&mut ir, Stage::Fragment);
    let mut e = Emitter::new(&mut ir, b).unwrap();
    let a = e.emit(O::UndefU32, &[], 0).unwrap();
    let b = e.emit(O::UndefU32, &[], 0).unwrap();
    let low = e
        .emit(O::BitFieldUExtract, &[a, Value::U32(0), Value::U32(16)], 0)
        .unwrap();
    let high = e
        .emit(O::BitFieldUExtract, &[b, Value::U32(16), Value::U32(16)], 0)
        .unwrap();
    let shifted = e
        .emit(O::ShiftLeftLogical32, &[low, Value::U32(16)], 0)
        .unwrap();
    let result = e.i_add(shifted, high).unwrap();
    vendor_workaround(&mut ir, &p).unwrap();
    assert_eq!(
        ir.inst(id(ir.resolve(result).unwrap())).unwrap().opcode(),
        O::BitwiseOr32
    );
    remove_identities(&mut ir, &p).unwrap();
    verify_program(&ir, &p).unwrap();
}
#[test]
fn query_rescaling_divides_dimensions_and_preserves_layers_and_mips() {
    let mut ir = Ir::new().unwrap();
    let (p, b) = program(&mut ir, Stage::Compute);
    let mut e = Emitter::new(&mut ir, b).unwrap();
    let mut f = nixe_shader_recompiler::ir::TextureInstInfo::default();
    f.set_texture_type(TextureType::ColorArray2D);
    f.set_descriptor_index(6);
    let query = e
        .emit(
            O::ImageQueryDimensions,
            &[Value::None, Value::U32(0), Value::U1(false)],
            f.0,
        )
        .unwrap();
    rescaling(
        &mut ir,
        &p,
        &ResolutionScaling {
            active: true,
            up_scale: 3,
            down_shift: 1,
            up_factor: 1.5,
            down_factor: 2.0 / 3.0,
        },
    )
    .unwrap();
    let result = ir.inst(id(ir.resolve(query).unwrap())).unwrap();
    assert_eq!(result.opcode(), O::CompositeConstructU32x4);
    for i in 2..4 {
        let extract = ir.inst(id(result.arg(i).unwrap())).unwrap();
        assert_eq!(extract.opcode(), O::CompositeExtractU32x4);
        assert_eq!(extract.arg(1).unwrap(), Value::U32(i as u32));
    }
    assert_eq!(count(&ir, &p, O::UDiv32), 2);
    assert_eq!(count(&ir, &p, O::ShiftLeftLogical32), 2);
    assert_eq!(count(&ir, &p, O::ImageQueryDimensions), 1);
    remove_identities(&mut ir, &p).unwrap();
    verify_program(&ir, &p).unwrap();
}

#[test]
fn marked_fragment_shuffle_keeps_literal_float_replacement() {
    let mut ir = Ir::new().unwrap();
    let (p, b) = program(&mut ir, Stage::Fragment);
    let mut e = Emitter::new(&mut ir, b).unwrap();
    let coord = e.get_attribute(Attribute::PositionX).unwrap();
    let bits = e.emit(O::BitCastU32F32, &[coord], 0).unwrap();
    let shuffle = e
        .emit(
            O::ShuffleIndex,
            &[bits, Value::U32(0), Value::U32(3), Value::U32(28)],
            0,
        )
        .unwrap();
    e.emit(O::IAdd32, &[shuffle, Value::U32(1)], 0).unwrap();
    assert!(has_broken_shuffle_pattern(&ir, &p).unwrap());
    rescaling(
        &mut ir,
        &p,
        &ResolutionScaling {
            active: true,
            up_scale: 2,
            up_factor: 2.0,
            down_factor: 0.5,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(ir.inst(id(coord)).unwrap().flags, 0xdeadbeef);
    assert_eq!(
        ir.value_type(ir.resolve(shuffle).unwrap()).unwrap(),
        Type::F32
    );
    assert_eq!(count(&ir, &p, O::ShuffleIndex), 1);
    assert!(verify_program(&ir, &p).is_err());
}
