use super::*;
use nixe_gpu::ShaderSourceLocation;

fn access(slot: u16, write: bool) -> ShaderOperation {
    let (location, component) = patch_location(slot * 4).unwrap();
    if write {
        ShaderOperation::StoreOutput {
            sources: vec![ShaderRegister::new(0)].into(),
            location,
            first_component: component,
            scalar_type: ShaderScalarType::Float32,
        }
    } else {
        ShaderOperation::LoadPatchOutput {
            destination: ShaderRegister::new(0),
            location,
            component,
        }
    }
}

fn ordered(operations: Vec<ShaderOperation>) -> Vec<ShaderInstruction> {
    let mut code = operations
        .into_iter()
        .enumerate()
        .map(|(i, op)| {
            ShaderInstruction::new(
                ShaderSourceLocation::new(i as u32 * 8),
                ShaderPredicate::Always,
                op,
            )
        })
        .collect();
    order_patch_outputs(&mut code);
    code
}

fn barrier_offsets(code: &[ShaderInstruction]) -> Vec<u32> {
    code.iter()
        .filter(|i| matches!(i.operation(), ShaderOperation::PatchBarrier))
        .map(|i| i.source().byte_offset())
        .collect()
}

#[test]
fn every_patch_slot_orders_raw_war_waw_but_not_rar() {
    for slot in (0..6).chain(8..256) {
        for (first, second, expected) in [
            (true, false, 1),
            (false, true, 1),
            (true, true, 1),
            (false, false, 0),
        ] {
            let code = ordered(vec![access(slot, first), access(slot, second)]);
            assert_eq!(
                barrier_offsets(&code).len(),
                expected,
                "slot={slot} writes={first}/{second}"
            );
        }
    }
}

#[test]
fn independent_components_do_not_synchronize_and_one_barrier_orders_all_slots() {
    let code = ordered(vec![
        access(8, true),
        access(9, true),
        access(63, true),
        access(64, true),
        access(8, false),
        access(9, false),
        access(63, false),
        access(64, false),
        access(255, true),
        access(8, false),
        access(255, false),
    ]);
    assert_eq!(barrier_offsets(&code), [32, 80]);
}

#[test]
fn vector_access_checks_all_components() {
    let mut vector = access(60, true);
    if let ShaderOperation::StoreOutput { sources, .. } = &mut vector {
        *sources = vec![ShaderRegister::new(0); 4].into();
    }
    let code = ordered(vec![
        access(63, false),
        vector,
        access(62, false),
        access(61, false),
    ]);
    assert_eq!(barrier_offsets(&code), [8, 16]);
}

#[test]
fn conditional_patch_writer_needs_unconditional_war_barrier() {
    let mut code = ordered(vec![access(12, false)]);
    code.push(ShaderInstruction::new(
        ShaderSourceLocation::new(8),
        ShaderPredicate::Register {
            register: 0,
            inverted: false,
        },
        access(12, true),
    ));
    order_patch_outputs(&mut code);
    assert_eq!(barrier_offsets(&code), [8]);
    assert_eq!(code[1].predicate(), ShaderPredicate::Always);
    assert_eq!(
        code[2].predicate(),
        ShaderPredicate::Register {
            register: 0,
            inverted: false
        }
    );
}
