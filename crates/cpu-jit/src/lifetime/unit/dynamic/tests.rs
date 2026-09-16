use super::*;
use crate::abi::{FpSpecialization, GuestValue, RegisterClass, ValueBinding};
use crate::lifetime::unit::tests::{input, key, process, publish};
use cranelift_codegen::nixe::StateMap;

pub(super) fn source(
    process: &Lifetime,
    cursor: &AtomicU64,
    pc: u64,
    kind: EdgeKind,
) -> UnitHandle {
    source_with_binding(process, cursor, pc, kind, false)
}

fn source_with_binding(
    process: &Lifetime,
    cursor: &AtomicU64,
    pc: u64,
    kind: EdgeKind,
    constant_x0: bool,
) -> UnitHandle {
    let mut candidate = input(process, &[pc], Tier::Lcq);
    candidate.faults = Box::new([]);
    candidate.code.metadata.faults = Box::new([]);
    let width = if candidate.code.metadata.abi == HostAbi::X86_64 {
        8
    } else {
        4
    };
    candidate.code.metadata.states = (0..2)
        .map(|id| StateMap {
            id,
            offset: 8,
            entry: false,
            patch_bytes: width,
            fault_bytes: 0,
            poll: None,
            values: Vec::new(),
        })
        .collect();
    let initial = &candidate.states[0].state;
    candidate.states = (0..2)
        .map(|id| {
            let mut state = initial.clone();
            state.site.state_map = id;
            if constant_x0 {
                state.live.integer.x[0] = true;
                state.bindings = Box::new([ValueBinding {
                    value: GuestValue::General(0),
                    location: ValueLocation::Constant(17),
                }]);
            }
            StateRecord {
                native_offset: 8,
                state,
                exit: Some(GuestExit {
                    pc: key(pc).pc,
                    kind,
                }),
                transfer: Some(Box::new(TerminalTransfer {
                    destination: ValueLocation::Register {
                        class: RegisterClass::Integer,
                        index: 0,
                    },
                    static_target: None,
                    completed: 1,
                    patch_bytes: width,
                    fallback_offset: 0,
                    poll_offset: None,
                })),
            }
        })
        .collect();
    // These fixtures are never entered at the synthetic terminal. Only their
    // immutable source contracts are used to emit/execute a transfer to a leaf.
    process
        .prepare_unit(&[process.reserve(key(pc)).unwrap()], candidate, cursor)
        .unwrap()
        .publish()
        .unwrap()
}

fn bridge(process: &Lifetime, source: UnitHandle, map: u32) -> PreparedBridge<'_> {
    process
        .prepare_dynamic_bridge(source, map, key(4))
        .unwrap()
        .unwrap()
}

#[test]
fn dynamic_resolution_keys_exact_source_maps_versions_and_preferred_target() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let a = source(&process, &cursor, 0, EdgeKind::Indirect);
    assert!(
        process
            .prepare_dynamic_bridge(a, 0, key(4))
            .unwrap()
            .is_none()
    );
    let target = publish(&process, &cursor, &[4, 8], Tier::Lcq);
    let b = source(&process, &cursor, 12, EdgeKind::Call);
    let c = source(&process, &cursor, 16, EdgeKind::Return);
    let first = bridge(&process, a, 0);
    assert_eq!(first.key(), bridge(&process, a, 0).key());
    assert_ne!(first.key(), bridge(&process, a, 1).key());
    assert_ne!(first.key(), bridge(&process, b, 0).key());
    assert_ne!(first.key(), bridge(&process, c, 0).key());
    let other_entry = process
        .prepare_dynamic_bridge(a, 0, key(8))
        .unwrap()
        .unwrap();
    assert_eq!(first.target, other_entry.target);
    assert_ne!(first.key(), other_entry.key());
    assert_eq!(first.target, target);
    assert_eq!(other_entry.target_entry, 1);
    let hcq = publish(&process, &cursor, &[4, 8], Tier::Hcq);
    assert!(process.try_service_links().unwrap());
    let promoted = bridge(&process, a, 0);
    assert_eq!(promoted.target, hcq);
    assert_ne!(promoted.key(), first.key());
    assert!(matches!(first.emit(), Err(Error::StalePublication)));
}

#[test]
fn dynamic_resolution_rejects_wrong_execution_keys_and_nondynamic_sources() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let a = source(&process, &cursor, 0, EdgeKind::Indirect);
    let mut wrong = key(4);
    wrong.fp = FpSpecialization::Exact(0);
    assert!(matches!(
        process.prepare_dynamic_bridge(a, 0, wrong),
        Err(Error::InvalidUnit(_))
    ));
    wrong = key(4);
    wrong.address_space = nixe_memory::AddressSpaceId::new(2);
    assert!(matches!(
        process.prepare_dynamic_bridge(a, 0, wrong),
        Err(Error::InvalidUnit(_))
    ));
    wrong = key(4);
    wrong.pc = nixe_memory::GuestVirtualAddress::new(5);
    assert!(matches!(
        process.prepare_dynamic_bridge(a, 0, wrong),
        Err(Error::InvalidUnit(_))
    ));
    assert!(matches!(
        process.prepare_dynamic_bridge(a, 2, key(4)),
        Err(Error::InvalidUnit(_))
    ));
    let observation = source(&process, &cursor, 12, EdgeKind::SupervisorCall(0));
    assert!(matches!(
        process.prepare_dynamic_bridge(observation, 0, key(4)),
        Err(Error::InvalidUnit(_))
    ));
    let other_process = super::super::tests::process();
    assert!(matches!(
        other_process.prepare_dynamic_bridge(a, 0, key(4)),
        Err(Error::StaleUnit)
    ));
}

#[test]
fn dynamic_empty_transfer_owns_units_without_allocating_code_or_islands() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    let a = source(&process, &cursor, 0, EdgeKind::Return);
    let before = process.cache.usage().unwrap();
    let transfer = bridge(&process, a, 0).emit().unwrap();
    assert!(transfer.code.is_none());
    assert_eq!(process.cache.usage().unwrap(), before);
    {
        let state = process.lock();
        transfer.validate(&state).unwrap();
        assert_eq!(
            Arc::strong_count(&state.units.records.get(a.0).unwrap().code),
            2
        );
        let target = &state.units.records.get(target.0).unwrap().code;
        assert_eq!(Arc::strong_count(target), 2);
        assert_eq!(transfer.address(), target.code.allocation.address());
        assert_eq!(
            transfer.prepared.source_code.code.allocation.island_count(),
            0
        );
    }
    // The synthetic target is a System-ABI leaf with no physical inputs. The
    // preparation's strong reference keeps its RX address alive through call.
    let call: unsafe extern "C" fn() -> u32 = unsafe { std::mem::transmute(transfer.address()) };
    assert_eq!(unsafe { call() }, 42);
}

#[test]
fn dynamic_nonempty_transfer_is_charged_reusable_and_has_no_static_islands() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut candidate = input(&process, &[4], Tier::Lcq);
    // A constant source X0 must become physical integer register zero. Remove
    // the synthetic leaf's MOV so its return exposes the transferred value.
    // Reinstall unpublished bytes, retaining the landing and unreachable maps.
    let old = candidate.code;
    let mut bytes = unsafe {
        std::slice::from_raw_parts(old.allocation.address() as *const u8, old.allocation.len())
    }
    .to_vec();
    if cfg!(target_arch = "x86_64") {
        bytes[4..9].fill(0x90);
    } else {
        bytes[4..8].copy_from_slice(&0xd503201f_u32.to_le_bytes());
    }
    candidate.code = process
        .cache
        .install(
            crate::executable::output::Output {
                bytes: bytes.into_boxed_slice(),
                alignment: 16,
                metadata: old.metadata,
            },
            Tier::Lcq,
            |_| None,
        )
        .unwrap();
    candidate.entries[0].contract.live_in.integer.x[0] = true;
    candidate.entries[0].contract.bindings = Box::new([ValueBinding {
        value: GuestValue::General(0),
        location: ValueLocation::Register {
            class: RegisterClass::Integer,
            index: 0,
        },
    }]);
    process
        .prepare_unit(&[process.reserve(key(4)).unwrap()], candidate, &cursor)
        .unwrap()
        .publish()
        .unwrap();
    let a = source_with_binding(&process, &cursor, 0, EdgeKind::Indirect, true);
    let prepared = bridge(&process, a, 0);
    let before = process.cache.usage().unwrap();
    let transfer = prepared.emit().unwrap();
    let code = transfer.code.as_ref().unwrap();
    assert_eq!(code.allocation.island_count(), 0);
    assert!(process.cache.usage().unwrap().metadata > before.metadata);
    let landing: &[u8] = if cfg!(target_arch = "x86_64") {
        &[0xf3, 0x0f, 0x1e, 0xfa]
    } else {
        &[0x5f, 0x24, 0x03, 0xd5]
    };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(transfer.address() as *const u8, 4) },
        landing
    );
    let address = transfer.address();
    // The bridge uses only a constant and caller-saved register zero; neither
    // it nor the synthetic target requires a pinned NativeFrame here.
    let call: unsafe extern "C" fn() -> u32 = unsafe { std::mem::transmute(address) };
    assert_eq!(unsafe { call() }, 17);
    drop(transfer);
    assert_eq!(process.cache.usage().unwrap(), before);
    let again = bridge(&process, a, 0).emit().unwrap();
    assert_eq!(again.address(), address);
}

#[test]
fn dynamic_preparations_do_not_survive_admission_changes_as_callable_entries() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let a = source(&process, &cursor, 0, EdgeKind::Indirect);
    let transfer = bridge(&process, a, 0).emit().unwrap();
    process.request(Reason::LinkPatch).unwrap();
    assert_eq!(transfer.validate(&process.lock()), Err(Error::Closed));
    assert!(matches!(
        process.prepare_dynamic_bridge(a, 0, key(4)),
        Err(Error::Closed)
    ));
    assert!(process.try_service_links().unwrap());
    assert_eq!(
        transfer.validate(&process.lock()),
        Err(Error::StalePublication)
    );
    bridge(&process, a, 0)
        .emit()
        .unwrap()
        .validate(&process.lock())
        .unwrap();
}

#[test]
fn dynamic_preparation_roots_delay_reclamation_but_not_safety_withdrawal() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let target = publish(&process, &cursor, &[4], Tier::Lcq);
    let a = source(&process, &cursor, 0, EdgeKind::Indirect);
    let transfer = bridge(&process, a, 0).emit().unwrap();
    process.retire_unit(target).unwrap();
    process.retire_unit(a).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(transfer.validate(&process.lock()).is_err());
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert!(!process.try_shutdown().unwrap());
    drop(transfer);
    assert!(process.try_shutdown().unwrap());
}
