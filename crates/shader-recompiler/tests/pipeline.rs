use nixe_shader_recompiler::{
    Compiler, CompilerSettings, HostTranslateInfo, Profile, RuntimeInfo, StageArtifact,
    artifact::{Bindings, StageProvenance},
    environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
    ir::{Ir, Opcode as O, Program, SyntaxNode as N, Type, Value as V},
    maxwell::{bits::insert, decode::Opcode as G},
    passes::GraphicsRuntimeState,
    spirv::emit_spirv,
    types::{OutputTopology, ProgramOffset, Stage},
};
use spirv::Op;
use std::{
    process::Command,
    sync::atomic::{AtomicU32, Ordering},
};
fn profile() -> Profile {
    Profile {
        supported_spirv: 0x00010500,
        unified_descriptor_binding: true,
        support_geometry_streams: true,
        max_user_clip_distances: 8,
        ..Default::default()
    }
}
fn validate(artifact: &StageArtifact, label: &str) {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let path = std::env::temp_dir().join(format!(
        "nixe-{label}-{}-{}.spv",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let bytes: Vec<u8> = artifact
        .words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    std::fs::write(&path, bytes).unwrap();
    let result = Command::new(std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into()))
        .args(["--target-env", "vulkan1.2"])
        .arg(&path)
        .output()
        .expect("Install SPIRV-Tools or set SPIRV_VAL to validate compiler output");
    assert!(
        result.status.success(),
        "{}\nmodule: {}",
        String::from_utf8_lossy(&result.stderr),
        path.display()
    );
    std::fs::remove_file(path).unwrap();
}
fn operations(words: &[u32]) -> Vec<(Op, &[u32])> {
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
fn word(op: G) -> u64 {
    insert::<16, 4>(op.encoding().value, 7)
}
fn environment(stage: Stage) -> SnapshotEnvironment {
    let start = if stage == Stage::Compute { 8 } else { 0x50 };
    SnapshotEnvironment::new(
        EnvironmentMetadata {
            stage,
            workgroup_size: [1, 1, 1],
            ..Default::default()
        },
        Observations {
            instructions: [(ProgramOffset(start), word(G::EXIT) | 15)].into(),
            viewport_transform_state: Some(1),
            ..Default::default()
        },
    )
}

#[test]
fn consecutive_stages_observe_their_own_texture_handles() {
    use nixe_shader_recompiler::{
        environment::{HandleKey, RecordingEnvironment},
        types::{ByteOffset, CbufIndex, TextureHandle, TextureType},
    };
    let tex = insert::<51, 3>(insert::<31, 4>(insert::<28, 3>(word(G::TEX), 2), 15), 7);
    let offset = ByteOffset(((tex >> 36) as u32 & 8191) * 4);
    let make = |handle, integer| {
        SnapshotEnvironment::new(
            EnvironmentMetadata {
                stage: Stage::Compute,
                texture_bound_buffer: CbufIndex(2),
                workgroup_size: [1; 3],
                ..Default::default()
            },
            Observations {
                instructions: [
                    (ProgramOffset(8), tex),
                    (ProgramOffset(16), word(G::EXIT) | 15),
                ]
                .into(),
                texture_handles: [(
                    HandleKey {
                        index: CbufIndex(2),
                        offset,
                        count: 1,
                        ..Default::default()
                    },
                    TextureHandle(handle),
                )]
                .into(),
                texture_types: [(TextureHandle(handle), TextureType::Color2D)].into(),
                texture_integer_formats: [(TextureHandle(handle), integer)].into(),
                viewport_transform_state: Some(1),
                ..Default::default()
            },
        )
    };
    let mut compiler = Compiler::new().unwrap();
    let mut first = make(3, false);
    compiler
        .translate(
            &mut first,
            &HostTranslateInfo::default(),
            &CompilerSettings::default(),
        )
        .unwrap();
    let mut second = make(7, true);
    let mut recorded = RecordingEnvironment::new(&mut second);
    compiler
        .translate(
            &mut recorded,
            &HostTranslateInfo::default(),
            &CompilerSettings::default(),
        )
        .unwrap();
    assert_eq!(
        recorded
            .observations()
            .texture_handles
            .values()
            .map(|handle| handle.0)
            .collect::<Vec<_>>(),
        vec![7]
    );
    assert_eq!(
        recorded
            .observations()
            .texture_integer_formats
            .get(&TextureHandle(7)),
        Some(&true)
    );
}
#[test]
fn whole_compute_translation_owns_output_across_resets_and_workers() {
    let mut compiler = Compiler::new().unwrap();
    let settings = CompilerSettings {
        renderer_debug: true,
        ..Default::default()
    };
    let mut env = environment(Stage::Compute);
    let first = compiler
        .translate(&mut env, &HostTranslateInfo::default(), &settings)
        .unwrap();
    assert!(compiler.dump_ir(first).unwrap().contains("Prologue"));
    let artifact = compiler
        .emit(
            first,
            &profile(),
            &RuntimeInfo::default(),
            &settings,
            &mut Bindings::default(),
        )
        .unwrap();
    assert!(compiler.program(first).is_err());
    validate(&artifact, "compute");
    let words = artifact.words.clone();
    compiler.reset().unwrap();
    assert!(compiler.program(first).is_err());
    let second = compiler
        .translate(&mut env, &HostTranslateInfo::default(), &settings)
        .unwrap();
    assert_ne!(first, second);
    let mut other = Compiler::new().unwrap();
    let alien = other
        .translate(&mut env, &HostTranslateInfo::default(), &settings)
        .unwrap();
    assert!(compiler.program(alien).is_err());
    assert_eq!(artifact.words, words);
}
#[test]
fn all_emitted_stage_models_validate() {
    for stage in [
        Stage::VertexB,
        Stage::TessellationControl,
        Stage::TessellationEval,
        Stage::Geometry,
        Stage::Fragment,
        Stage::Compute,
    ] {
        let mut ir = Ir::new().unwrap();
        let b = ir.new_block().unwrap();
        let mut p = Program::new(stage);
        p.blocks = vec![b];
        p.syntax = vec![N::Block(b), N::Return];
        p.execution.workgroup_size = [2, 3, 4];
        p.execution.invocations = 1;
        p.execution.output_vertices = 3;
        p.execution.output_topology = Some(OutputTopology::TriangleStrip);
        let a = emit_spirv(
            &mut ir,
            p,
            &profile(),
            &RuntimeInfo::default(),
            &CompilerSettings::default(),
            &mut Bindings::default(),
        )
        .unwrap();
        validate(&a, &format!("{stage:?}"));
    }
}
#[test]
fn nested_selection_and_loop_phi_validate_with_both_safety_modes() {
    for disabled in [false, true] {
        let mut ir = Ir::new().unwrap();
        let entry = ir.new_block().unwrap();
        let header = ir.new_block().unwrap();
        let body = ir.new_block().unwrap();
        let selected = ir.new_block().unwrap();
        let selection_merge = ir.new_block().unwrap();
        let cont = ir.new_block().unwrap();
        let merge = ir.new_block().unwrap();
        let phi = ir.new_phi(header, Type::U32).unwrap();
        let next = ir
            .append(cont, O::IAdd32, &[V::Inst(phi), V::U32(1)], 0)
            .unwrap();
        let cond = ir
            .append(cont, O::ULessThan, &[V::Inst(next), V::U32(4)], 0)
            .unwrap();
        ir.add_phi_operand(phi, entry, V::U32(0)).unwrap();
        ir.add_phi_operand(phi, cont, V::Inst(next)).unwrap();
        let mut p = Program::new(Stage::Compute);
        p.execution.workgroup_size = [1, 1, 1];
        p.blocks = vec![entry, header, body, selected, selection_merge, cont, merge];
        p.syntax = vec![
            N::Block(entry),
            N::Block(header),
            N::Loop {
                body,
                continue_block: cont,
                merge,
            },
            N::Block(body),
            N::If {
                condition: V::U1(true),
                body: selected,
                merge: selection_merge,
            },
            N::Block(selected),
            N::EndIf {
                merge: selection_merge,
            },
            N::Block(selection_merge),
            N::Block(cont),
            N::Repeat {
                condition: V::Inst(cond),
                loop_header: header,
                merge,
            },
            N::Block(merge),
            N::Return,
        ];
        let a = emit_spirv(
            &mut ir,
            p,
            &profile(),
            &RuntimeInfo::default(),
            &CompilerSettings {
                disable_shader_loop_safety_checks: disabled,
                ..Default::default()
            },
            &mut Bindings::default(),
        )
        .unwrap();
        validate(&a, "loop");
        assert_eq!(
            operations(&a.words)
                .iter()
                .any(|(op, _)| *op == Op::SGreaterThanEqual),
            !disabled
        );
    }
}
#[test]
fn fragment_demote_fallback_has_structured_kill() {
    let mut ir = Ir::new().unwrap();
    let b = ir.new_block().unwrap();
    ir.append(b, O::DemoteToHelperInvocation, &[], 0).unwrap();
    let mut p = Program::new(Stage::Fragment);
    p.blocks = vec![b];
    p.syntax = vec![N::Block(b), N::Return];
    p.info.uses_demote_to_helper_invocation = true;
    let a = emit_spirv(
        &mut ir,
        p,
        &profile(),
        &RuntimeInfo::default(),
        &CompilerSettings::default(),
        &mut Bindings::default(),
    )
    .unwrap();
    validate(&a, "demote");
    assert!(operations(&a.words).iter().any(|(op, _)| *op == Op::Kill));
}
#[test]
fn dual_vertex_and_graphics_linkage_use_one_owned_vertex_artifact() {
    let mut compiler = Compiler::new().unwrap();
    let settings = CompilerSettings {
        renderer_debug: true,
        ..Default::default()
    };
    let host = HostTranslateInfo::default();
    let mut a = environment(Stage::VertexA);
    let mut b = environment(Stage::VertexB);
    let mut f = environment(Stage::Fragment);
    let va = compiler.translate(&mut a, &host, &settings).unwrap();
    let vb = compiler.translate(&mut b, &host, &settings).unwrap();
    let vertex = compiler.merge_dual_vertex(va, vb, &b, &settings).unwrap();
    let fragment = compiler.translate(&mut f, &host, &settings).unwrap();
    let output = compiler
        .emit_graphics(
            &[vertex, fragment],
            &profile(),
            &GraphicsRuntimeState::default(),
            &settings,
            &mut Bindings::default(),
        )
        .unwrap();
    assert_eq!(output.len(), 2);
    assert_eq!(
        output[0].execution.provenance,
        StageProvenance::MergedVertex
    );
    for a in &output {
        validate(a, "graphics");
    }
    compiler.reset().unwrap();
    assert!(!output[0].words.is_empty());
}
#[test]
fn buffer_cas_and_sampled_image_modules_validate() {
    use nixe_shader_recompiler::{
        artifact::{StorageBufferDescriptor, TextureDescriptor},
        ir::TextureInstInfo,
        types::{ByteOffset, CbufIndex, TextureType},
    };
    let mut ir = Ir::new().unwrap();
    let b = ir.new_block().unwrap();
    let mut p = Program::new(Stage::Compute);
    p.execution.workgroup_size = [1, 1, 1];
    p.execution.shared_memory_size = 64;
    p.blocks = vec![b];
    p.syntax = vec![N::Block(b), N::Return];
    p.info.uses_int8 = true;
    p.info.used_storage_buffer_types = Type::U8 | Type::U32;
    p.info
        .storage_buffers_descriptors
        .push(StorageBufferDescriptor {
            cbuf_index: CbufIndex(0),
            cbuf_offset: ByteOffset(0),
            count: 1,
            is_written: true,
        })
        .unwrap();
    ir.append(b, O::WriteSharedU8, &[V::U32(3), V::U32(17)], 0)
        .unwrap();
    let value = ir.append(b, O::LoadSharedU8, &[V::U32(3)], 0).unwrap();
    ir.append(
        b,
        O::WriteStorageU8,
        &[V::U32(0), V::U32(5), V::Inst(value)],
        0,
    )
    .unwrap();
    let a = emit_spirv(
        &mut ir,
        p,
        &profile(),
        &RuntimeInfo::default(),
        &CompilerSettings::default(),
        &mut Bindings::default(),
    )
    .unwrap();
    validate(&a, "memory-cas");
    let mut ir = Ir::new().unwrap();
    let b = ir.new_block().unwrap();
    let mut p = Program::new(Stage::Fragment);
    p.blocks = vec![b];
    p.syntax = vec![N::Block(b), N::Return];
    p.info.stores_frag_color[0] = true;
    p.info.texture_descriptors.push(TextureDescriptor {
        texture_type: TextureType::Color2D,
        count: 1,
        ..Default::default()
    });
    let coords = ir
        .append(
            b,
            O::CompositeConstructF32x2,
            &[V::f32(0.25), V::f32(0.5)],
            0,
        )
        .unwrap();
    let mut flags = TextureInstInfo::default();
    flags.set_texture_type(TextureType::Color2D);
    let sample = ir
        .append(
            b,
            O::ImageSampleImplicitLod,
            &[V::U32(0), V::Inst(coords), V::None, V::None],
            flags.0,
        )
        .unwrap();
    for i in 0..4 {
        let c = ir
            .append(
                b,
                O::CompositeExtractF32x4,
                &[V::Inst(sample), V::U32(i)],
                0,
            )
            .unwrap();
        ir.append(b, O::SetFragColor, &[V::U32(0), V::U32(i), V::Inst(c)], 0)
            .unwrap();
    }
    let a = emit_spirv(
        &mut ir,
        p,
        &profile(),
        &RuntimeInfo::default(),
        &CompilerSettings::default(),
        &mut Bindings::default(),
    )
    .unwrap();
    validate(&a, "sample");
}
