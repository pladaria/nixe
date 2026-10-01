use super::*;

const STORE: u64 = 0xeedc_2000_0007_00ff;

fn r(index: u16) -> ShaderRegister {
    ShaderRegister::new(index)
}

fn immediate(destination: u16, bits: u32) -> ShaderOperation {
    ShaderOperation::MoveImmediate32 {
        destination: r(destination),
        bits,
        scalar_type: ShaderScalarType::Unsigned32,
    }
}

fn observe(
    memory: &mut GlobalMemory,
    ops: Vec<ShaderOperation>,
    next: &mut u16,
) -> Vec<ShaderInstruction> {
    let mut instructions = ops
        .into_iter()
        .map(|op| ShaderInstruction::new(ShaderSourceLocation::new(8), ShaderPredicate::Always, op))
        .collect();
    memory.observe(&mut instructions, 0, 0, next).unwrap();
    instructions
}

fn pointer(relative: u32, high_offset: u32) -> (GlobalMemory, u16, Vec<ShaderInstruction>) {
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
            matches!(&operations[1], ShaderOperation::ShiftRightLogical32 { value, .. } if *value == r(20))
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
        ShaderOperation::ShiftRightLogical32 { .. }
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
fn store_rejects_control_flow_in_both_orders_and_unsupported_encoding_fields() {
    for branch_first in [false, true] {
        let (mut memory, mut next, _) = pointer(0, 0x144);
        if branch_first {
            memory.observe_control_flow(40, 0).unwrap();
            assert!(memory.store(48, STORE, 6, &mut next).is_err());
        } else {
            memory.store(48, STORE, 6, &mut next).unwrap();
            assert!(memory.observe_control_flow(56, 0).is_err());
        }
    }
    for encoding in [
        STORE & !(1 << 45),
        STORE | (1 << 46),
        STORE | (1 << 44),
        STORE | (1 << 43),
        STORE | (2 << 20),
        STORE ^ (1 << 48),
        STORE & !(7 << 16),
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
                accesses: vec![(4, 28)],
            },
            GlobalBufferBinding {
                constant_buffer: 0,
                byte_offset: 8,
                binding: 33,
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
