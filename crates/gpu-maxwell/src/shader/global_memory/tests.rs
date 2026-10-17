use super::*;
use nixe_gpu::ShaderSourceLocation;

const STORE: u64 = 0xeedc_2000_0007_00ff;

fn r(index: u32) -> ShaderRegister {
    ShaderRegister::new(index)
}

fn immediate(destination: u32, bits: u32) -> ShaderOperation {
    ShaderOperation::MoveImmediate32 {
        destination: r(destination),
        bits,
        scalar_type: ShaderScalarType::Unsigned32,
    }
}

fn observe(
    memory: &mut GlobalMemory,
    ops: Vec<ShaderOperation>,
    next: &mut u32,
) -> Vec<ShaderInstruction> {
    let mut instructions = ops
        .into_iter()
        .map(|op| ShaderInstruction::new(ShaderSourceLocation::new(8), ShaderPredicate::Always, op))
        .collect();
    memory.observe(&mut instructions, 0, 0, next).unwrap();
    instructions
}

fn pointer(relative: u32, high_offset: u32) -> (GlobalMemory, u32, Vec<ShaderInstruction>) {
    let mut memory = GlobalMemory::default();
    let mut next = 20;
    let instructions = observe(
        &mut memory,
        vec![
            immediate(0, relative),
            immediate(2, 0),
            ShaderOperation::LoadConstantBuffer32 {
                destination: r(3),
                binding: 2,
                byte_offset: 0x140,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::LoadConstantBuffer32 {
                destination: r(4),
                binding: 2,
                byte_offset: high_offset,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::AddCarry32 {
                destination: r(0),
                carry_out: r(5),
                left: r(3),
                right: r(0),
                carry_in: None,
            },
            ShaderOperation::AddCarry32 {
                destination: r(1),
                carry_out: r(5),
                left: r(4),
                right: r(2),
                carry_in: Some(r(5)),
            },
        ],
        &mut next,
    );
    (memory, next, instructions)
}

#[test]
fn aliased_pointer_is_frozen_before_add_and_repeated_stores_share_a_binding() {
    let (mut memory, mut next, instructions) = pointer(32, 0x144);
    assert!(
        matches!(instructions[4].operation(), ShaderOperation::Move32 { destination, source, .. }
        if *destination == r(20) && *source == r(0))
    );
    assert!(
        matches!(instructions[5].operation(), ShaderOperation::AddCarry32 { destination, .. } if *destination == r(0))
    );
    for displacement in [0, 4, 12] {
        let operations = memory
            .store(48, STORE | (displacement << 20), 6, &mut next)
            .unwrap();
        assert!(matches!(
            operations.last().unwrap(),
            ShaderOperation::StoreStorageBuffer32 { binding: 32, .. }
        ));
        assert!(
            matches!(&operations[1], ShaderOperation::ShiftRight32 { value, .. } if *value == r(20))
        );
    }
    assert_eq!(memory.bindings.buffers.len(), 1);
    let buffer = &memory.bindings.buffers[0];
    assert_eq!((buffer.constant_buffer, buffer.byte_offset), (2, 0x140));
    assert_eq!(memory.bindings.byte_extents([1; 3], [1; 3]), [48]);
}

#[test]
fn displacement_keeps_address_carry_instead_of_wrapping_a_byte_offset() {
    let (mut memory, mut next, _) = pointer(0xffff_fffc, 0x144);
    let operations = memory.store(48, STORE | (4 << 20), 6, &mut next).unwrap();
    assert_eq!(
        memory.bindings.byte_extents([1; 3], [1; 3]),
        [0x1_0000_0004]
    );
    assert!(matches!(
        operations[1],
        ShaderOperation::ShiftRight32 { .. }
    ));
    assert!(matches!(
        operations[2],
        ShaderOperation::MoveImmediate32 { bits: 1, .. }
    ));
    assert!(matches!(operations[3], ShaderOperation::Add32 { .. }));
}

#[test]
fn mismatched_high_half_unaligned_offset_and_clobbered_pointer_are_rejected() {
    for (relative, high) in [(0, 0x148), (2, 0x144)] {
        let (mut memory, mut next, _) = pointer(relative, high);
        assert!(memory.store(48, STORE, 6, &mut next).is_err());
    }
    for predicate in [
        ShaderPredicate::Always,
        ShaderPredicate::Register {
            register: 0,
            inverted: false,
        },
    ] {
        for index in [0, 1] {
            let (mut memory, mut next, _) = pointer(0, 0x144);
            let mut write = vec![ShaderInstruction::new(
                ShaderSourceLocation::new(40),
                predicate,
                immediate(index, 0),
            )];
            memory.observe(&mut write, 0, 0, &mut next).unwrap();
            assert!(memory.store(48, STORE, 6, &mut next).is_err());
        }
    }
}

#[test]
fn invalid_encodings_still_fail() {
    for encoding in [
        STORE & !(1 << 45),
        STORE | (1 << 46),
        STORE | (1 << 44),
        STORE | (1 << 43),
        STORE | (2 << 20),
        STORE ^ (4 << 48),
    ] {
        let (mut memory, mut next, _) = pointer(0, 0x144);
        assert!(
            memory.store(48, encoding, 6, &mut next).is_err(),
            "{encoding:#x}"
        );
    }
}

#[test]
fn wrapped_bound_expands_conservatively_and_dispatch_geometry_is_not_cached() {
    let bindings = GlobalBufferBindings {
        bounds: vec![
            Bound::Group(0),
            Bound::Shift(0, 5),
            Bound::Local(0),
            Bound::Add(1, 2),
            Bound::Shift(3, 5),
            Bound::Constant(u32::MAX),
            Bound::Add(4, 5),
        ],
        buffers: vec![
            GlobalBufferBinding {
                constant_buffer: 0,
                byte_offset: 0,
                binding: 32,
                read: false,
                write: true,
                runtime_extent: false,
                accesses: vec![(4, 28)],
            },
            GlobalBufferBinding {
                constant_buffer: 0,
                byte_offset: 8,
                binding: 33,
                read: false,
                write: true,
                runtime_extent: false,
                accesses: vec![(6, 0)],
            },
        ],
    };
    assert_eq!(
        bindings.byte_extents([8, 1, 1], [32, 1, 1]),
        [8192, 0x1_0000_0003]
    );
    assert_eq!(
        bindings.byte_extents([16, 1, 1], [32, 1, 1]),
        [16384, 0x1_0000_0003]
    );
}

#[test]
fn loads_and_stores_share_proven_pointers_and_accumulate_access_permissions() {
    let (mut memory, mut next, _) = pointer(32, 0x144);
    let load = 0xeed4_a000_0047_0002;
    let operations = memory.load(48, load, 6, &mut next).unwrap();
    assert!(matches!(operations.last().unwrap(),
        ShaderOperation::LoadStorageBuffer32 { destination, binding: 32, .. } if *destination == r(2)));
    assert!(memory.bindings.buffers[0].read);
    assert!(!memory.bindings.buffers[0].write);
    memory.store(56, STORE | (12 << 20), 6, &mut next).unwrap();
    assert_eq!(memory.bindings.buffers.len(), 1);
    assert!(memory.bindings.buffers[0].read && memory.bindings.buffers[0].write);
    assert_eq!(memory.bindings.byte_extents([1; 3], [1; 3]), [48]);
}

#[test]
fn direct_constant_buffer_pointer_can_load_into_its_own_low_register() {
    let mut memory = GlobalMemory::default();
    let mut next = 6;
    observe(
        &mut memory,
        (0..2)
            .map(|i| ShaderOperation::LoadConstantBuffer32 {
                destination: r(i),
                binding: 2,
                byte_offset: 0x140 + i * 4,
                scalar_type: ShaderScalarType::Unsigned32,
            })
            .collect(),
        &mut next,
    );
    let operations = memory
        .load(32, 0xeed4_a000_0007_0000, 6, &mut next)
        .unwrap();
    assert_eq!(memory.bindings.byte_extents([1; 3], [1; 3]), [4]);
    assert!(matches!(operations.last().unwrap(),
        ShaderOperation::LoadStorageBuffer32 { destination, binding: 32, .. } if *destination == r(0)));
    observe(&mut memory, operations, &mut next);
    // Loading arbitrary memory destroys pointer provenance; another access
    // must not accidentally use the old base after the destination aliases it.
    assert!(
        memory
            .load(40, 0xeed4_a000_0007_0000, 6, &mut next)
            .is_err()
    );
}

#[test]
fn global_loads_reject_unproven_addresses_and_unsupported_widths() {
    for encoding in [
        0xeed4_a000_0007_0000 ^ (4 << 48),
        0xeed4_a000_0007_0000 & !(1 << 45),
        0xeed4_a000_0007_0000 | (2 << 20),
        0xeed4_a000_0007_0000 | (1 << 44),
    ] {
        let (mut memory, mut next, _) = pointer(0, 0x144);
        assert!(memory.load(48, encoding, 6, &mut next).is_err());
    }
    let mut memory = GlobalMemory::default();
    assert!(memory.load(48, 0xeed4_a000_0007_0000, 6, &mut 6).is_err());
}

#[test]
fn wide_global_accesses_expand_consecutive_words_with_one_frozen_address() {
    for (width, words) in [(4_u64, 1), (5, 2), (6, 4), (7, 4)] {
        let (mut memory, mut next, _) = pointer(32, 0x144);
        let word = 0xeed0_a000_00c7_0000 | (width << 48);
        let ops = memory.load(48, word, 8, &mut next).unwrap();
        let outputs = ops
            .iter()
            .filter_map(|op| match op {
                ShaderOperation::LoadStorageBuffer32 { destination, .. } => {
                    Some(destination.index())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs, (0..words).collect::<Vec<_>>());
        assert_eq!(
            memory.bindings.byte_extents([1; 3], [1; 3]),
            [32 + 12 + 4 * u64::from(words)]
        );
        assert!(!memory.bindings.buffers[0].runtime_extent);
        assert!(
            ops.iter()
                .filter(|op| matches!(op, ShaderOperation::ShiftRight32 { .. }))
                .count()
                == 1
        );
    }
}

#[test]
fn unknown_loop_indices_use_nvn_runtime_extent_instead_of_a_fictitious_static_bound() {
    let mut memory = GlobalMemory::default();
    let mut next = 20;
    observe(
        &mut memory,
        vec![
            immediate(2, 0),
            ShaderOperation::LoadConstantBuffer32 {
                destination: r(3),
                binding: 0,
                byte_offset: 0x140,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::LoadConstantBuffer32 {
                destination: r(4),
                binding: 0,
                byte_offset: 0x144,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::AddCarry32 {
                destination: r(0),
                carry_out: r(5),
                left: r(3),
                right: r(0),
                carry_in: None,
            },
            ShaderOperation::AddCarry32 {
                destination: r(1),
                carry_out: r(5),
                left: r(4),
                right: r(2),
                carry_in: Some(r(5)),
            },
        ],
        &mut next,
    );
    memory
        .load(48, 0xeed4_a000_0007_0002, 8, &mut next)
        .unwrap();
    assert!(memory.bindings.buffers[0].runtime_extent);
    assert_eq!(memory.bindings.buffers[0].byte_offset, 0x140);
}

fn cfg_relocation(
    branch_clobber: bool,
    loop_clobber: bool,
    bypass: bool,
) -> Result<(GlobalBufferBindings, Vec<ShaderInstruction>), MaxwellShaderTranslationError> {
    use crate::shader::{
        binary::{MaxwellShaderBinary, MaxwellShaderInstructionBundle, MaxwellShaderMetadata},
        control_flow::ShaderControlFlow,
    };
    let binary = MaxwellShaderBinary {
        address: 0,
        metadata: MaxwellShaderMetadata::Compute {
            workgroup_size: [1; 3],
            register_count: 8,
        },
        bundles: (0..3)
            .map(|i| MaxwellShaderInstructionBundle {
                offset: i * 32,
                control: 0,
                instructions: [0; 3],
            })
            .collect(),
        indirect_tables: Box::default(),
        source_cpu_writes: Box::default(),
        source_mappings: Box::default(),
    };
    let mut flow = ShaderControlFlow {
        reachable: [8, 16, 24, 40, 48, 56, 72].into_iter().collect(),
        successors: [
            (8, vec![16]),
            (16, vec![24, 40]),
            (24, vec![48]),
            (40, vec![48]),
            (48, vec![56]),
            (56, vec![48, 72]),
        ]
        .into_iter()
        .map(|(pc, next)| (pc, next.into_iter().collect()))
        .collect(),
        stack_targets: BTreeMap::new(),
    };
    let mut instructions = Vec::new();
    let mut add = |pc, op| {
        instructions.push(ShaderInstruction::new(
            ShaderSourceLocation::new(pc),
            ShaderPredicate::Always,
            op,
        ))
    };
    if bypass {
        flow.successors.insert(8, [16, 40].into_iter().collect());
    }
    let definition = if bypass { 16 } else { 8 };
    for op in [
        immediate(0, 32),
        immediate(2, 0),
        ShaderOperation::LoadConstantBuffer32 {
            destination: r(3),
            binding: 2,
            byte_offset: 0x140,
            scalar_type: ShaderScalarType::Unsigned32,
        },
        ShaderOperation::LoadConstantBuffer32 {
            destination: r(4),
            binding: 2,
            byte_offset: 0x144,
            scalar_type: ShaderScalarType::Unsigned32,
        },
        ShaderOperation::AddCarry32 {
            destination: r(0),
            carry_out: r(5),
            left: r(3),
            right: r(0),
            carry_in: None,
        },
        ShaderOperation::AddCarry32 {
            destination: r(1),
            carry_out: r(5),
            left: r(4),
            right: r(2),
            carry_in: Some(r(5)),
        },
    ] {
        add(definition, op);
    }
    add(
        24,
        if branch_clobber {
            immediate(0, 7)
        } else {
            ShaderOperation::Nop
        },
    );
    add(40, ShaderOperation::Nop);
    // A repeated read through the pointer is valid even when the original
    // relative-offset register was overwritten by its low address half.
    add(
        48,
        decode_access(MaxwellShaderStage::Compute, 48, 0xeed4_2000_0007_0006, 8)?,
    );
    add(
        56,
        if loop_clobber {
            immediate(1, 9)
        } else {
            ShaderOperation::Nop
        },
    );
    add(72, ShaderOperation::Exit);
    let bindings = flow::lower(&binary, &flow, &mut instructions, 8, &mut 8)?;
    Ok((bindings, instructions))
}

#[test]
fn cfg_join_and_loop_retain_an_unchanged_dominating_pointer() {
    let (bindings, instructions) = cfg_relocation(false, false, false).unwrap();
    assert_eq!(bindings.buffers.len(), 1);
    assert_eq!(bindings.byte_extents([1; 3], [1; 3]), [36]);
    assert!(bindings.buffers[0].read);
    assert!(!bindings.buffers[0].write);
    assert!(
        !instructions
            .iter()
            .any(|i| matches!(i.operation(), ShaderOperation::GlobalMemory { .. }))
    );
    assert!(instructions.iter().any(|i| matches!(i.operation(), ShaderOperation::LoadStorageBuffer32 { destination, .. } if *destination == r(6))));
}

#[test]
fn cfg_rejects_branch_clobbers_late_backedge_clobbers_and_bypassed_definitions() {
    for (branch, backedge, bypass) in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
    ] {
        assert!(
            cfg_relocation(branch, backedge, bypass).is_err(),
            "branch={branch} backedge={backedge} bypass={bypass}"
        );
    }
}

#[test]
fn predicated_pointer_loads_relocate_only_under_their_original_guard() {
    let predicate = ShaderPredicate::Register {
        register: 1,
        inverted: false,
    };
    let mut memory = GlobalMemory::default();
    let mut next = 32;
    let mut definitions = [0x140, 0x144]
        .into_iter()
        .enumerate()
        .map(|(i, byte_offset)| {
            ShaderInstruction::new(
                ShaderSourceLocation::new(8 + i as u32 * 8),
                predicate,
                ShaderOperation::LoadConstantBuffer32 {
                    destination: r(12 + i as u32),
                    binding: 0,
                    byte_offset,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
            )
        })
        .collect();
    memory.observe(&mut definitions, 0, 0, &mut next).unwrap();
    let encoding = 0xeed6_2000_0001_0c0c;
    let ops = memory.load(40, encoding, 32, &mut next).unwrap();
    assert_eq!(
        ops.iter()
            .filter(|op| matches!(op, ShaderOperation::LoadStorageBuffer32 { .. }))
            .count(),
        4
    );
    assert_eq!(memory.bindings.buffers[0].byte_offset, 0x140);
    assert!(memory.load(40, encoding | 7 << 16, 32, &mut next).is_err());
    assert!(memory.load(40, encoding | 1 << 19, 32, &mut next).is_err());
    memory
        .values
        .invalidate_predicate_writes(&ShaderOperation::SetPredicateBoolean {
            destinations: [Some(1), None],
            inputs: [ShaderPredicate::Always; 3],
            operations: [nixe_gpu::ShaderPredicateSetOperation::And; 2],
        });
    assert!(memory.load(40, encoding, 32, &mut next).is_err());
}
