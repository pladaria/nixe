use super::*;
use nixe_cpu::{
    exclusive::ExclusiveMonitorState,
    execution::{ArchitecturalTimer, TimerSnapshot, VcpuEventState},
    memory::ProcessMemory,
};
use nixe_cpu_interpreter::{InstructionStep, InterpreterContext, execute_one_with_context};
use nixe_memory::{
    CanonicalBackingPage, CanonicalBackingStore, ContentGeneration, DirectArena, DirectMapRequest,
    DirectProtection,
};
use std::cell::RefCell;

mod atomic;
mod authority;
mod casp;
mod cold;
mod exclusive;
mod invocation;

struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 1,
        }
    }
}

const DATA: usize = 0x2000;
const ARENA: usize = 0x4000;

fn run(words: &[u32], state: &mut A64State, arena: &mut [u8]) {
    run_with_fault(words, state, arena, None);
}

// `fault` selects a protected page and the subaccess/commit stage expected at
// its first access. Authority/invocation tests exercise real memory policy.
fn run_with_fault(
    words: &[u32],
    state: &mut A64State,
    arena: &mut [u8],
    fault: Option<(usize, u16, u16)>,
) {
    run_memory_case(words, state, arena, fault, false, false);
}

fn run_memory_case(
    words: &[u32],
    state: &mut A64State,
    arena: &mut [u8],
    fault: Option<(usize, u16, u16)>,
    escape: bool,
    inherited_fp: bool,
) -> Option<(
    crate::lcq::fault::Reconstructed,
    crate::lcq::fault::access::Access,
)> {
    let backing = CanonicalBackingStore::allocate().unwrap();
    let pages: Vec<_> = arena
        .chunks_exact(4096)
        .enumerate()
        .map(|(i, bytes)| {
            CanonicalBackingPage::initialized(
                &backing,
                GuestPhysicalPageId::new(i as u64 + 1),
                bytes,
                ContentGeneration::INITIAL,
            )
            .unwrap()
        })
        .collect();
    let backing_pages: Vec<_> = pages
        .iter()
        .map(|page| page.direct_backing().unwrap())
        .collect();
    let direct = DirectArena::new(arena.len()).unwrap();
    let requests: Vec<_> = backing_pages
        .iter()
        .enumerate()
        .map(|(i, backing)| DirectMapRequest {
            guest_address: (i * 4096) as u64,
            backing,
            protection: if fault.is_some_and(|(page, _, _)| i * 4096 == page) {
                DirectProtection::None
            } else if i == 1 {
                DirectProtection::Read
            } else {
                DirectProtection::ReadWrite
            },
        })
        .collect();
    direct.map_pages(&requests).unwrap();
    let memory = super::memory(words);
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    let mut reader = process.register().unwrap();
    let Request::Owner(claim) = reader.claim(key()).unwrap() else {
        panic!()
    };
    let compilation = Compilation::capture(claim, &memory).unwrap();
    let mut compiler = Compiler::for_arena(native_abi(), arena.len()).unwrap();
    let handle = compiler
        .publish(compilation, &process, &cache, &memory)
        .unwrap();
    let unit = process.snapshot(handle).unwrap();
    let mut frame = NativeFrame::new(state, PollBudget::new(4096, 1000).unwrap());
    let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
    for record in &unit.faults {
        unit.states[record.state_map as usize]
            .state
            .validate()
            .unwrap();
        let pc = unit.code.allocation.address() + record.native_start as usize;
        let found = invocation.fault(pc).unwrap();
        assert_eq!(found.record.native_end, record.native_end);
    }
    let entry = invocation.payload().preferred().unwrap();
    let mut reconstructed = None;
    let result = if let Some((page, subaccess, stage)) = fault {
        use nixe_cpu_direct_memory::{InvocationOutcome, NativeInvocation, WorkerFaultContext};
        let mut worker = WorkerFaultContext::register().unwrap();
        let (frame, lookup) = invocation.frame_and_faults();
        let mut call = CapturedEntry {
            frame,
            arena: direct.view().base as *mut u8,
            result: None,
        };
        let mut repair = Repair {
            lookup,
            frame: std::ptr::from_ref(call.frame).cast(),
            arena: &direct,
            page,
            subaccess,
            stage,
            seen: 0,
            escape,
            access: None,
            caller_fp: [
                call.frame.host_fp.saved_control,
                call.frame.host_fp.saved_status,
            ],
        };
        let outcome = unsafe {
            worker.invoke_captured(
                direct.view(),
                [
                    call.frame.host_fp.saved_control,
                    call.frame.host_fp.saved_status,
                ],
                repair_mapping,
                std::ptr::from_mut(&mut repair).cast(),
                NativeInvocation {
                    gateway: if inherited_fp {
                        captured_fp_entry
                    } else {
                        captured_entry
                    },
                    context: std::ptr::from_mut(&mut call).cast(),
                    entry: entry.canonical.get(),
                },
            )
        }
        .unwrap();
        assert_eq!(repair.seen, 1);
        if escape {
            assert_eq!(outcome, InvocationOutcome::Escaped);
            assert!(call.result.is_none());
            let captured = worker.escaped_fault().unwrap();
            let found = repair.lookup.find(captured.native_pc()).unwrap();
            reconstructed = Some((
                unsafe { crate::lcq::fault::reconstruct(call.frame, &captured, &found) }.unwrap(),
                repair.access.unwrap(),
            ));
            assert!(
                unsafe { crate::lcq::fault::reconstruct(call.frame, &captured, &found) }.is_err(),
                "an escaped invocation must not merge captured status twice"
            );
            None
        } else {
            assert_eq!(outcome, InvocationOutcome::Returned);
            assert!(worker.escaped_fault().is_err());
            Some(call.result.unwrap().unwrap())
        }
    } else {
        Some(
            unsafe {
                if inherited_fp {
                    invocation.frame().ensure_fp().unwrap();
                    crate::fp_env::tests::divide_by_zero();
                }
                crate::native::enter_protected(
                    invocation.frame(),
                    direct.view().base as *mut u8,
                    entry.canonical.get() as *const u8,
                )
            }
            .unwrap(),
        )
    };
    if let Some(result) = result {
        assert_eq!(result.reason, NativeExitReason::Architectural);
    }
    // All pages in this test are mapped and readable; the real arena's guard
    // remains inaccessible. No native invocation survives this copy or unmap.
    drop(invocation);
    if escape && fault.unwrap().0 < arena.len() {
        // Readback of the fixture only, after leaving native execution. The
        // failed guest operation is never retried or completed by this mapping.
        direct
            .protect_ranges(&[nixe_memory::DirectProtectRequest {
                guest_address: fault.unwrap().0 as u64,
                size: 4096,
                protection: DirectProtection::ReadWrite,
            }])
            .unwrap();
    }
    arena.copy_from_slice(unsafe {
        std::slice::from_raw_parts(direct.view().base as *const u8, arena.len())
    });
    reconstructed
}

struct CapturedEntry<'a, 's> {
    frame: &'a mut NativeFrame<'s>,
    arena: *mut u8,
    result: Option<Result<crate::native::NativeReturn, crate::native::NativeReturnError>>,
}

unsafe extern "C" fn captured_entry(opaque: *mut libc::c_void, entry: usize) {
    let call = unsafe { &mut *opaque.cast::<CapturedEntry<'_, '_>>() };
    call.result =
        Some(unsafe { crate::native::enter_protected(call.frame, call.arena, entry as *const u8) });
}

// Seed the invocation as if a previous native unit activated FP. Do this only
// after fault capture is installed and before entering the integer-only unit.
unsafe extern "C" fn captured_fp_entry(opaque: *mut libc::c_void, entry: usize) {
    unsafe {
        let call = &mut *opaque.cast::<CapturedEntry<'_, '_>>();
        call.frame.ensure_fp().unwrap();
        crate::fp_env::tests::divide_by_zero();
        captured_entry(opaque, entry);
    }
}

#[test]
fn integer_only_unit_keeps_inherited_fpsr_on_return_retry_and_escape() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    // Deliberately nonzero caller status, distinct from guest divide-by-zero.
    // Inactive guest FP must never import those caller bits at a memory fault.
    let caller = crate::fp_env::tests::distinct_caller();
    for inherited_fp in [false, true] {
        for mode in 0..3 {
            let escape = mode == 2;
            let mut state = A64State::default();
            state.set_pc(PC);
            state.set_fpsr(1 << 27);
            state.general_register_storage_mut()[0] = 0xdead_beef;
            state.general_register_storage_mut()[1] = 0x3000;
            state.general_register_storage_mut()[5] = 5;
            let mut expected = state.clone();
            expected.general_register_storage_mut()[5] = 6;
            expected.set_pc(PC + if escape { 4 } else { 8 });
            expected.set_fpsr((1 << 27) | if inherited_fp { 2 } else { 0 });
            if !escape {
                expected.general_register_storage_mut()[0] = 0x1212_1212_1212_1212;
            }
            let mut arena = vec![0x12; ARENA];
            let result = run_memory_case(
                &[0x9100_04a5, 0xf940_0020, 0xd420_0000], // ADD X5; LDR X0,[X1]; BRK
                &mut state,
                &mut arena,
                (mode != 0).then_some((0x3000, 0, 0)),
                escape,
                inherited_fp,
            );
            assert_eq!(result.is_some(), escape);
            assert_eq!(state, expected, "inherited={inherited_fp}, mode={mode}");
            let mut restored = crate::abi::HostFpState::default();
            unsafe {
                restored.begin();
                restored.finish();
            }
            assert_eq!((restored.saved_control, restored.saved_status), caller);
        }
    }
}

struct Repair<'a> {
    lookup: crate::lifetime::FaultLookup<'a>,
    frame: *const libc::c_void,
    arena: &'a DirectArena,
    page: usize,
    subaccess: u16,
    stage: u16,
    seen: usize,
    escape: bool,
    access: Option<crate::lcq::fault::access::Access>,
    caller_fp: [u64; 2],
}

unsafe extern "C" fn repair_mapping(
    opaque: *mut libc::c_void,
    captured: *mut nixe_cpu_direct_memory::CapturedFault,
) -> nixe_cpu_direct_memory::FaultDisposition {
    use nixe_cpu_direct_memory::FaultDisposition;
    let repair = unsafe { &mut *opaque.cast::<Repair<'_>>() };
    let mut host = crate::abi::HostFpState::default();
    unsafe { host.begin() };
    assert_eq!(
        [host.saved_control, host.saved_status],
        repair.caller_fp,
        "the landing leaf must restore caller FP before running the dispatcher"
    );
    let captured = unsafe { &*captured };
    let Some(found) = repair.lookup.find(captured.native_pc()) else {
        return FaultDisposition::Fatal;
    };
    // Native execution is suspended. Inspection borrows its frame without
    // canonicalizing or disturbing registers/FPSR needed by exact retry.
    let access = unsafe {
        crate::lcq::fault::access::inspect(
            &*repair.frame.cast::<NativeFrame<'_>>(),
            captured,
            &found,
            repair.arena.view(),
        )
    }
    .unwrap();
    assert_eq!(access.size.bytes(), usize::from(found.record.bytes));
    repair.access = Some(access);
    assert_eq!(
        captured.native_pc(),
        found.unit.code.allocation.address() + found.record.native_start as usize
    );
    assert_eq!(found.record.subaccess, repair.subaccess);
    assert_eq!(found.record.commit_stage, repair.stage);
    assert_eq!(repair.seen, 0, "a repaired page must not fault again");
    assert_eq!(
        (captured.fault_address() - repair.arena.view().base) & !4095,
        repair.page
    );
    // The physical prefault map and bytes are borrowed through the active
    // Invocation, without Arc acquisition or legacy registry publication.
    let state = &found.unit.states[found.record.state_map as usize].state;
    state.validate().unwrap();
    repair.seen += 1;
    if repair.escape {
        return FaultDisposition::Escape;
    }
    repair
        .arena
        .protect_ranges(&[nixe_memory::DirectProtectRequest {
            guest_address: repair.page as u64,
            size: 4096,
            protection: DirectProtection::ReadWrite,
        }])
        .unwrap();
    FaultDisposition::Retry
}

#[test]
fn delivered_lcq_faults_recover_original_addresses_before_repair() {
    for (word, base, offset, address, bytes) in [
        (0xf940_0420, 0x2ff8, 0, 0x3000, 8), // unsigned +8
        (0xf85f_8020, 0x3008, 0, 0x3000, 8), // unscaled -8
        (0xf840_8c20, 0x2ff8, 0, 0x3000, 8), // pre-index +8
        (0xf840_8420, 0x3000, 0, 0x3000, 8), // post-index +8
        (0xf862_4820, 0x2ff8, 0xffff_ffff_0000_0008, 0x3000, 8), // UXTW
        (0xf862_d820, 0x3008, 0xffff_ffff, 0x3000, 8), // SXTW scaled
        (0xf862_7820, 0x2ff8, 1, 0x3000, 8), // LSL scaled
        (0xf862_e820, 0x3008, u64::MAX - 7, 0x3000, 8), // SXTX
        (0xf87f_6820, 0x3000, 123, 0x3000, 8), // XZR offset
        (0x5801_0000, 0x3000, 0, PC + 4 + 0x2000, 8), // literal uses fault PC
        (0xf940_0020, 0x2ffc, 0, 0x2ffc, 8), // Linux reports the second page
        (0x3dc0_0420, 0x2ff0, 0, 0x3000, 16), // Q unsigned +16
        (0x3cdf_0020, 0x3010, 0, 0x3000, 16), // Q unscaled -16
        (0x3cc1_0c20, 0x2ff0, 0, 0x3000, 16), // Q pre-index
        (0x3cc1_0420, 0x3000, 0, 0x3000, 16), // Q post-index
        (0x3ce2_d820, 0x3010, 0xffff_ffff, 0x3000, 16), // Q SXTW scaled
        (0x3dc0_0020, 0x2ff8, 0, 0x2ff8, 16), // Q crosses page
        (0xf940_0020, ARENA as u64 - 4, 0, ARENA as u64 - 4, 8), // crosses guard
        (0xf940_0020, ARENA as u64 + 32, 0, ARENA as u64 + 32, 8), // confined start
        (0xf940_0020, u64::MAX - 3, 0, u64::MAX - 3, 8), // extent overflow
        (0xf862_6820, u64::MAX - 7, 0x3008, 0x3000, 8), // wrapping address arithmetic
    ] {
        for sp in [false, true] {
            let word = if sp && word != 0x5801_0000 {
                (word & !(31 << 5)) | (31 << 5)
            } else {
                word
            };
            let access = check_address(word, base, offset, sp, address, bytes, false);
            let fault = access.guest_fault(SPACE);
            if address.checked_add(bytes as u64 - 1).is_none() {
                assert_eq!(
                    fault.unwrap().reason,
                    nixe_cpu::memory::DataAccessFaultReason::AddressOverflow
                );
            } else {
                assert!(fault.is_none());
            }
        }
    }
}

#[test]
fn delivered_lcq_ordered_alignment_faults_are_not_mapping_repairs() {
    for size in 1..4 {
        for load in [false, true] {
            for sp in [false, true] {
                let word = (if load { 0x08df_fc00 } else { 0x089f_fc00 })
                    | (size << 30)
                    | ((if sp { 31 } else { 1 }) << 5);
                let bytes = 1 << size;
                for address in [0x2001, 0x3001, u64::MAX] {
                    let access = check_address(word, address, 0, sp, address, bytes, true);
                    let fault = access.guest_fault(SPACE).unwrap();
                    assert_eq!(fault.address.get(), address);
                    assert_eq!(
                        fault.reason,
                        nixe_cpu::memory::DataAccessFaultReason::Misaligned {
                            required_alignment: bytes as u8,
                        }
                    );
                    assert_eq!(
                        fault.kind,
                        if load {
                            nixe_cpu::memory::DataAccessKind::Read
                        } else {
                            nixe_cpu::memory::DataAccessKind::Write
                        }
                    );
                }
            }
        }
    }
}

fn check_address(
    word: u32,
    base: u64,
    offset: u64,
    sp: bool,
    address: u64,
    bytes: usize,
    ordered: bool,
) -> crate::lcq::fault::access::Access {
    let mut state = A64State::default();
    state.set_pc(PC);
    state.general_register_storage_mut()[1] = base.wrapping_sub(1);
    state.general_register_storage_mut()[2] = offset;
    *state.stack_pointer_storage_mut() = base.wrapping_sub(1);
    let mut expected = state.clone();
    let add = if sp {
        *expected.stack_pointer_storage_mut() = base;
        0x9100_07ff // ADD SP,SP,#1
    } else {
        expected.general_register_storage_mut()[1] = base;
        0x9100_0421 // ADD X1,X1,#1; address comes from dirty state
    };
    expected.set_pc(PC + 4);
    let page = if address.saturating_add(bytes as u64 - 1) >= ARENA as u64
        || (ordered && address & (bytes as u64 - 1) != 0)
    {
        ARENA
    } else {
        0x3000
    };
    let mut arena = vec![0x12; ARENA];
    let (_, access) = run_memory_case(
        &[add, word, 0xd420_0000],
        &mut state,
        &mut arena,
        Some((page, 0, 0)),
        true,
        false,
    )
    .unwrap();
    assert_eq!(access.address.get(), address, "word={word:08x}");
    assert_eq!(access.size.bytes(), bytes);
    assert_eq!(access.alignment, if ordered { bytes as u8 } else { 1 });
    assert_eq!(
        state, expected,
        "failed instruction must preserve PRE state: {word:08x}"
    );
    assert!(
        arena.iter().all(|byte| *byte == 0x12),
        "failed access must not commit a store"
    );
    access
}

#[test]
fn escaped_lcq_faults_reconstruct_the_architectural_prefix() {
    for (word, base, subaccess, stage) in [
        (0xf840_8420, 0x3000, 0, 0), // LDR X0,[X1],#8: no destination/writeback
        (0x3cc1_0420, 0x3000, 0, 0), // LDR Q0,[X1],#16
        (0xa8c1_0c20, 0x2ff8, 1, 0), // LDP X0,X3,[X1],#16
        (0xa881_0c20, 0x2ff8, 1, 1), // STP X0,X3,[X1],#16
        (0xacc1_0c20, 0x2ff0, 1, 0), // LDP Q0,Q3,[X1],#32
        (0x0cdf_883f, 0x2ff4, 3, 3), // LD2 V31,V0: completed lanes, zero upper halves
        (0x0c9f_883f, 0x2ff4, 3, 3), // ST2: completed stores
    ] {
        check_escape(&[0xf100_04a5], word, base, subaccess, stage);
    }
}

#[test]
fn contiguous_structure_faults_reconstruct_grouped_pre_state_and_partial_vectors() {
    for full in [false, true] {
        for size in 0..4 {
            if !full && size == 3 {
                continue;
            }
            let lanes = (if full { 16u16 } else { 8 }) >> size;
            for load in [false, true] {
                let word =
                    0x0c9f_203f | (u32::from(full) << 30) | (u32::from(load) << 22) | (size << 10);
                for prefix in [0, 1, lanes + 1] {
                    check_escape(
                        &[0xf100_04a5],
                        word,
                        0x3000 - (u64::from(prefix) << size),
                        prefix,
                        prefix,
                    );
                }
            }
        }
    }
}

#[test]
fn escaped_lcq_faults_reconstruct_lazy_flag_recipes() {
    for producer in [
        0xab07_00c5,
        0xeb07_00c5,
        0xba07_00c5,
        0xfa07_00c5,
        0xea07_00c5,
    ] {
        for word in [producer, producer & !(1 << 31)] {
            check_escape(&[word], 0xf940_0020, 0x3000, 0, 0);
            check_escape(&[0xf100_08a5, word], 0xf940_0020, 0x3000, 0, 0); // initial C=0
        }
    }
    // CCMP/CCMN with both predicate outcomes, following a dirty producer.
    for conditional in [0xfa47_00c5, 0xba47_00c5, 0x7a47_00c5, 0x3a47_00c5] {
        for condition in [0, 1] {
            check_escape(
                &[0xf100_04a5, conditional | (condition << 12)],
                0xf940_0020,
                0x3000,
                0,
                0,
            );
        }
    }
    check_escape(&[], 0xf940_0020, 0x3000, 0, 0); // canonical NZCV
    check_escape(&[0xea07_03e5], 0xf940_0020, 0x3000, 0, 0); // ANDS X5,XZR,X7
    check_escape(&[0xd51b_4206], 0xf940_0020, 0x3000, 0, 0); // MSR NZCV,X6: packed
}

#[test]
fn escaped_lcq_fault_reconstructs_spills_and_merges_captured_fpsr_once() {
    let mut prefix = Vec::new();
    for register in 2..31 {
        prefix.push(0x9100_0400 | (register << 5) | register);
    }
    for register in 0..32 {
        prefix.push(0x4e20_8400 | (register << 16) | (register << 5) | register);
    }
    prefix.extend([
        0x9e67_0121, // FMOV D1,X9
        0x9e67_0142, // FMOV D2,X10
        0xf100_04a5, // SUBS
        0x1e62_2820, // FADD D0,D1,D2: pending inexact
    ]);
    check_escape(&prefix, 0xf900_0023, 0x3000, 0, 0);
}

fn check_escape(prefix: &[u32], word: u32, base: u64, subaccess: u16, stage: u16) {
    let mut words = prefix.to_vec();
    words.extend([word, 0xd420_0000]);
    let mut memory = super::memory(&words);
    let page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(page));
    assert!(memory.initialize_ram(page, 0, &[0x12; 4096]));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA as u64),
        page,
        MemoryPermissions::READ_WRITE
    ));
    let mut expected = A64State::default();
    expected.set_pc(PC);
    expected.set_fpcr(1 << 22);
    expected.set_fpsr(1 << 27);
    expected.set_nzcv(Nzcv::from_bits(Nzcv::C | Nzcv::V));
    expected.general_register_storage_mut()[1] = base;
    expected.general_register_storage_mut()[5] = 1;
    expected.general_register_storage_mut()[6] = 0x7fff_ffff_7fff_ffff;
    expected.general_register_storage_mut()[7] = 1;
    expected.general_register_storage_mut()[9] = 1.0_f64.to_bits() - 1;
    expected.general_register_storage_mut()[10] = 0x3ca0_0000_0000_0000 - 1;
    for register in 0..32 {
        assert!(expected.set_vector(register, u128::MAX));
    }
    let mut actual = expected.clone();
    let monitor = RefCell::new(ExclusiveMonitorState::default());
    let events = VcpuEventState::default();
    for &instruction in prefix {
        let context = InterpreterContext::new(
            ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
            &memory,
            &monitor,
            &Timer,
            &events,
        );
        assert_eq!(
            execute_one_with_context(context, &mut expected, instruction).unwrap(),
            InstructionStep::Continue
        );
    }
    let context = InterpreterContext::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
        &memory,
        &monitor,
        &Timer,
        &events,
    );
    assert!(matches!(
        execute_one_with_context(context, &mut expected, word).unwrap(),
        InstructionStep::Exit(nixe_cpu::execution::CpuExit::DataFault { .. })
    ));
    let mut actual_bytes = vec![0x12; ARENA];
    let (recovered, _) = run_memory_case(
        &words,
        &mut actual,
        &mut actual_bytes,
        Some((0x3000, subaccess, stage)),
        true,
        false,
    )
    .unwrap();
    assert_eq!(actual, expected, "{word:08x} prefix={prefix:x?}");
    let mut expected_bytes = [0; 4096];
    memory
        .read_bytes(
            SPACE,
            GuestVirtualAddress::new(DATA as u64),
            &mut expected_bytes,
        )
        .unwrap();
    assert_eq!(actual_bytes[DATA..DATA + 4096], expected_bytes);
    assert_eq!(
        actual_bytes[0x3000..],
        [0x12; 4096],
        "faulting store must not execute"
    );
    if let Some(value) = recovered.completed_read {
        let bytes = if word == 0xacc1_0c20 { 16 } else { 8 };
        let mut expected = [0u8; 16];
        expected[..bytes].fill(0x12);
        assert_eq!(value, u128::from_le_bytes(expected));
    }
    assert_eq!(
        recovered.completed_read.is_some(),
        matches!(word, 0xa8c1_0c20 | 0xacc1_0c20)
    );
    assert!(recovered.poll_remaining <= 1000);
}

#[test]
fn delivered_lcq_faults_use_the_live_directory_and_retry_exact_subaccess() {
    // First load, first store, Q load, second pair read/write and interleaved
    // first/second-lane reads. A successful prefix must survive exact-PC retry.
    for (word, base, subaccess, stage) in [
        (0xf940_0020, 0x3000, 0, 0), // LDR X0,[X1]
        (0xf900_0023, 0x3000, 0, 0), // STR X3,[X1]
        (0x3dc0_0020, 0x3000, 0, 0), // LDR Q0,[X1]
        (0xa940_0c20, 0x2ff8, 1, 0), // LDP X0,X3,[X1]
        (0xa900_0c20, 0x2ff8, 1, 1), // STP X0,X3,[X1]
        (0x0cdf_883f, 0x2ffc, 1, 1), // LD2 {V31.2S,V0.2S},[X1],#16
        (0x0cdf_883f, 0x2ff4, 3, 3),
        (0x0c9f_883f, 0x2ff4, 3, 3), // ST2, completed first three elements
        (0x4cdf_203f, 0x3000, 0, 0), // grouped LD1 .16B
        (0x4c9f_203f, 0x3000, 0, 0), // grouped ST1 .16B
        (0x4cdf_283f, 0x2ff4, 3, 3), // LD1 .4S, partial vector before retry
        (0x4c9f_283f, 0x2ff4, 3, 3), // ST1 .4S, partial store prefix
        (0x4cdf_2c3f, 0x2ff8, 1, 1), // LD1 .2D, first half retained
    ] {
        let words = [0xf100_04a5, word, 0x9a1f_00c6, 0xd420_0000];
        let mut expected = A64State::default();
        expected.set_pc(PC);
        expected.general_register_storage_mut()[1] = base;
        expected.general_register_storage_mut()[3] = 0xfedc_ba98_7654_3210;
        expected.general_register_storage_mut()[5] = 1;
        for register in 0..32 {
            assert!(expected.set_vector(register, u128::MAX));
        }
        let mut actual = expected.clone();
        let mut expected_bytes = vec![0x12; ARENA];
        let mut actual_bytes = expected_bytes.clone();
        run(&words, &mut expected, &mut expected_bytes);
        run_with_fault(
            &words,
            &mut actual,
            &mut actual_bytes,
            Some((0x3000, subaccess, stage)),
        );
        assert_eq!(actual, expected, "{word:08x} subaccess={subaccess}");
        assert_eq!(
            actual_bytes, expected_bytes,
            "{word:08x} subaccess={subaccess}"
        );
    }
}

#[test]
fn delivered_lcq_retry_preserves_dirty_spills_lazy_flags_and_pending_fp() {
    let mut words = Vec::new();
    for register in 2..31 {
        words.push(0x9100_0400 | (register << 5) | register); // ADD Xn,Xn,#1
    }
    for register in 0..32 {
        words.push(0x4e20_8400 | (register << 16) | (register << 5) | register); // ADD Vn.16B,Vn,Vn
    }
    words.extend([
        0x9e67_0121, // FMOV D1,X9
        0x9e67_0142, // FMOV D2,X10
        0xf100_04a5, // SUBS X5,X5,#1
        0x1e62_2820, // FADD D0,D1,D2, inexact, guest round toward +infinity
        0xf900_0023, // STR X3,[X1] faults, with pending FPSR and dirty spills
        0x1e62_2803, // FADD D3,D0,D2, same guest environment after retry
        0x9a1f_00c6, // ADC X6,X6,XZR consumes the lazy carry
        0xd420_0000,
    ]);
    let fragment = Fragment::capture(&super::memory(&words), key()).unwrap();
    let lowered = Compiler::for_arena(native_abi(), ARENA)
        .unwrap()
        .lower(&fragment, CodeVersion::new(1).unwrap())
        .unwrap();
    let map = &lowered.states[lowered.faults[0].state_map as usize].state;
    assert!(map.host_fpsr_pending);
    assert!(
        map.bindings
            .iter()
            .any(|binding| matches!(binding.location, crate::abi::ValueLocation::Spill { .. }))
    );
    let mut expected = A64State::default();
    expected.set_pc(PC);
    expected.set_fpcr(1 << 22);
    expected.set_fpsr(1 << 27);
    expected.general_register_storage_mut()[1] = 0x3000;
    // The preceding ADDs and FMOVs produce exactly 1.0 and 2^-53.
    expected.general_register_storage_mut()[9] = 1.0_f64.to_bits() - 1;
    expected.general_register_storage_mut()[10] = 0x3ca0_0000_0000_0000 - 1;
    let mut actual = expected.clone();
    let mut expected_bytes = vec![0; ARENA];
    let mut actual_bytes = expected_bytes.clone();
    run(&words, &mut expected, &mut expected_bytes);
    run_with_fault(&words, &mut actual, &mut actual_bytes, Some((0x3000, 0, 0)));
    assert_eq!(actual, expected);
    assert_eq!(actual_bytes, expected_bytes);
    assert_eq!(actual.vector(0), Some(u128::from(1.0_f64.to_bits() + 1)));
    assert_eq!(actual.vector(3), Some(u128::from(1.0_f64.to_bits() + 2)));
    assert_eq!(actual.fpsr(), (1 << 27) | (1 << 4));
}

#[test]
fn scalar_ram_addressing_and_extensions_match_interpreter() {
    for word in [
        0xf940_0020,
        0xf900_0023,
        0xb940_0020,
        0xb900_0023,
        0x3940_0020,
        0x3900_0023,
        0x7940_0020,
        0x7900_0023,
        0x3980_0020,
        0x39c0_0020,
        0x7980_0020,
        0x79c0_0020,
        0xb980_0020,
        0xf840_0020,
        0xf800_0023,
        0xf85f_8020,
        0xf81f_8023,
        0xf840_8420,
        0xf800_8423,
        0xf840_8c20,
        0xf800_8c23,
        0xf862_6820,
        0xf822_6823,
        0xf862_5820,
        0xf862_d820,
        0xf862_f820,
        0xf940_03e0,
        0xf900_03ff, // SP base, XZR store
        0xf940_003f, // discarded load still accesses memory
        0x5800_0040,
        0x1800_0040,
        0x9800_0040, // literals X/W/LDRSW
    ] {
        check_transfer(word);
    }
}

fn check_transfer(word: u32) {
    check_transfer_at(word, &[0, 1, 7, 4060]);
}

fn check_transfer_at(word: u32, offsets: &[usize]) {
    for &unaligned in offsets {
        // A dirty flags producer precedes memory; ADC after it requires C.
        let words = [0xf100_04a5, word, 0x9a1f_00c6, 0xd420_0000];
        let mut memory = super::memory(&words);
        let mut arena = vec![0; ARENA];
        for (i, word) in words.iter().enumerate() {
            arena[PC as usize + i * 4..PC as usize + i * 4 + 4]
                .copy_from_slice(&word.to_le_bytes());
        }
        for (i, byte) in arena[DATA..].iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(17) ^ 0x89;
        }
        for page in 0..2 {
            let id = GuestPhysicalPageId::new(2 + page as u64);
            assert!(memory.add_ram_page(id));
            assert!(memory.initialize_ram(
                id,
                0,
                &arena[DATA + page * 4096..DATA + (page + 1) * 4096]
            ));
            assert!(memory.map_page(
                SPACE,
                GuestVirtualAddress::new((DATA + page * 4096) as u64),
                id,
                MemoryPermissions::READ_WRITE
            ));
        }
        let mut actual = A64State::default();
        actual.set_pc(PC);
        actual.set_fpcr(0x00c0_9f00);
        actual.set_fpsr((1 << 27) | 0x9f);
        for i in 0..32 {
            assert!(actual.set_vector(
                i,
                0xff80_0001_7f80_0001_807f_0100_fedc_ba98u128 ^ (u128::from(i) << 112)
            ));
        }
        actual.general_register_storage_mut()[1] = (DATA + 32 + unaligned) as u64;
        actual.general_register_storage_mut()[2] = 8;
        actual.general_register_storage_mut()[3] = 0x8877_6655_4433_2211;
        actual.general_register_storage_mut()[5] = 1;
        actual.general_register_storage_mut()[6] = 19;
        *actual.stack_pointer_storage_mut() = (DATA + 32 + unaligned) as u64;
        let mut expected = actual.clone();
        let monitor = RefCell::new(ExclusiveMonitorState::default());
        let events = VcpuEventState::default();
        for word in &words[..3] {
            let context = InterpreterContext::new(
                ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
                &memory,
                &monitor,
                &Timer,
                &events,
            );
            assert_eq!(
                execute_one_with_context(context, &mut expected, *word).unwrap(),
                InstructionStep::Continue
            );
        }
        run(&words, &mut actual, &mut arena);
        assert_eq!(actual, expected, "{word:08x}, unaligned {unaligned}");
        let mut expected_bytes = vec![0; ARENA - DATA];
        memory
            .read_bytes(
                SPACE,
                GuestVirtualAddress::new(DATA as u64),
                &mut expected_bytes,
            )
            .unwrap();
        assert_eq!(arena[DATA..], expected_bytes, "{word:08x}");
    }
}

#[test]
fn multiple_structures_match_interpreter() {
    for opcode in [8, 4, 0, 7, 10, 6, 2] {
        // LD1/2/3/4 and ST1/2/3/4, including every contiguous arrangement.
        for full in [false, true] {
            for size in 0..4 {
                let contiguous = matches!(opcode, 7 | 10 | 6 | 2);
                if !contiguous && !full && size == 3 {
                    continue;
                } // .1D is reserved.
                for load in [false, true] {
                    for (post, rm, rn, rt) in [
                        (false, 0, 1, 31),
                        (true, 31, 31, 30),
                        (true, 2, 1, 0),
                        (true, 1, 1, 31), // Rm aliases Rn.
                    ] {
                        let word = 0x0c00_0000
                            | (u32::from(full) << 30)
                            | (u32::from(post) << 23)
                            | (u32::from(load) << 22)
                            | (rm << 16)
                            | (opcode << 12)
                            | (size << 10)
                            | (rn << 5)
                            | rt;
                        check_transfer_at(word, &[0, 1, 4060]);
                    }
                }
            }
        }
    }
}

#[test]
fn contiguous_structure_fault_maps_cover_grouped_and_element_paths() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for (opcode, count) in [(7, 1), (10, 2), (6, 3), (2, 4)] {
            for full in [false, true] {
                for size in 0..4 {
                    if !full && size == 3 {
                        continue;
                    } // existing .1D path
                    for load in [false, true] {
                        let transfer = 0x0c9f_003f
                            | (u32::from(full) << 30)
                            | (u32::from(load) << 22)
                            | (opcode << 12)
                            | (size << 10);
                        let memory = super::memory(&[0xf100_04a5, transfer, 0xd420_0000]);
                        let fragment = Fragment::capture(&memory, key()).unwrap();
                        let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
                        let lowered = compiler
                            .lower(&fragment, CodeVersion::new(1).unwrap())
                            .unwrap();
                        let bytes = if full { 16 } else { 8 };
                        let lanes = bytes >> size;
                        assert_eq!(lowered.faults.len(), count * (lanes + 1));
                        assert_eq!(
                            lowered
                                .faults
                                .iter()
                                .filter(|f| f.bytes == bytes as u8)
                                .count(),
                            count
                        );
                        for fault in &lowered.faults {
                            let index = usize::from(fault.subaccess);
                            assert!(index < count * lanes);
                            assert_eq!(fault.commit_stage, fault.subaccess);
                            assert!(fault.completed_read.is_none());
                            if fault.bytes == bytes as u8 {
                                assert_eq!(index % lanes, 0);
                            } else {
                                assert_eq!(fault.bytes, 1 << size);
                            }
                            let state = &lowered.states[fault.state_map as usize].state;
                            assert!(state.dirty_live.integer.x[1]); // inherited pre-writeback base
                            for register in 0..32 {
                                assert_eq!(
                                    state.dirty_live.vector[(31 + register) & 31],
                                    lowered.entry.live_in.vector[(31 + register) & 31]
                                        || (load && register < count && register * lanes < index)
                                );
                            }
                            state.validate().unwrap();
                        }
                        let clif = compiler.context.func.display().to_string();
                        assert_eq!(clif.matches("brif").count(), 1, "{clif}");
                        assert_eq!(
                            clif.matches("nixe_fault_start").count(),
                            count * (lanes + 1)
                        );
                        assert!(!clif.contains("call"), "{clif}");
                    }
                }
            }
        }
    }
}

#[test]
fn multiple_structure_fault_maps_preserve_each_committed_lane() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for (opcode, count) in [(8, 2), (4, 3), (0, 4), (7, 1), (10, 2), (6, 3), (2, 4)] {
            for full in [false, true] {
                for size in 0..4 {
                    let contiguous = matches!(opcode, 7 | 10 | 6 | 2);
                    if (contiguous && (full || size != 3)) || (!contiguous && !full && size == 3) {
                        continue;
                    }
                    for load in [false, true] {
                        // SUBS; LD/STn {V31,...},[X1],#bytes; BRK.
                        let transfer = 0x0c9f_003f
                            | (u32::from(full) << 30)
                            | (u32::from(load) << 22)
                            | (opcode << 12)
                            | (size << 10);
                        let memory = super::memory(&[0xf100_04a5, transfer, 0xd420_0000]);
                        let fragment = Fragment::capture(&memory, key()).unwrap();
                        let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
                        let lowered = compiler
                            .lower(&fragment, CodeVersion::new(1).unwrap())
                            .unwrap();
                        let accesses = count * ((if full { 16 } else { 8 }) >> size);
                        assert_eq!(lowered.faults.len(), accesses);
                        for (index, fault) in lowered.faults.iter().enumerate() {
                            assert_eq!(fault.subaccess, index as u16);
                            assert_eq!(fault.commit_stage, index as u16);
                            assert_eq!(fault.bytes, 1 << size);
                            assert!(fault.completed_read.is_none());
                            let state = &lowered.states[fault.state_map as usize].state;
                            assert!(state.dirty_live.integer.x[1]);
                            assert!(matches!(state.nzcv, NzcvLocation::Deferred(_)));
                            assert!(state.host_fpsr_pending && state.dirty_live.fpsr);
                            for offset in 0..32 {
                                assert_eq!(
                                    state.dirty_live.vector[(31 + offset) & 31],
                                    lowered.entry.live_in.vector[(31 + offset) & 31]
                                        || (load && offset < count && offset < index)
                                );
                            }
                            if index > 0 {
                                assert!(lowered.faults[index - 1].native_end <= fault.native_start);
                            }
                            let backend = &lowered.output.metadata.faults[index];
                            assert_eq!(
                                fault.native_end - fault.native_start,
                                u32::from(backend.fault_bytes)
                            );
                            state.validate().unwrap();
                        }
                        let exit = &lowered
                            .states
                            .iter()
                            .find(|map| map.exit.is_some())
                            .unwrap()
                            .state;
                        assert!(exit.dirty_live.integer.x[1]);
                        let clif = compiler.context.func.display().to_string();
                        assert_eq!(clif.matches("nixe_fault_start").count(), accesses);
                        assert_eq!(
                            clif.matches("store").count(),
                            if load { 0 } else { accesses }
                        );
                        assert!(!clif.contains("call"));
                        assert!(!clif.contains("i128"));
                    }
                }
            }
        }
    }
}

#[test]
fn single_structure_lanes_and_replication_match_interpreter() {
    for count in 1..=4 {
        for size in 0..4 {
            for (post, rm, rn, rt) in [(false, 0, 1, 31), (true, 31, 31, 30), (true, 2, 1, 0)] {
                let address = (u32::from(post) << 23) | (rm << 16) | (rn << 5) | rt;
                for lane in [0, (16 >> size) - 1] {
                    for load in [false, true] {
                        let word =
                            single_structure_word(count, size, Some(lane), load, false) | address;
                        check_transfer_at(word, &[0, 1, 4060]);
                    }
                }
                for full in [false, true] {
                    let word = single_structure_word(count, size, None, true, full) | address;
                    check_transfer_at(word, &[0, 1, 4060]);
                }
            }
        }
    }
    // The post-index register may equal the base; both are read before writeback.
    check_transfer(
        single_structure_word(4, 2, Some(3), true, false) | (1 << 23) | (1 << 16) | (1 << 5),
    );
}

// Independent A64 single-structure encoding: lane=None selects LDnR.
fn single_structure_word(count: u32, size: u32, lane: Option<u32>, load: bool, full: bool) -> u32 {
    let group = (count - 1) / 2;
    let (opcode, s, element_size, q) = match lane {
        Some(lane) => match size {
            0 => (group, (lane >> 2) & 1, lane & 3, lane >> 3),
            1 => (2 | group, (lane >> 1) & 1, (lane & 1) << 1, lane >> 2),
            2 => (4 | group, lane & 1, 0, lane >> 1),
            3 => (4 | group, 0, 1, lane),
            _ => unreachable!(),
        },
        None => (6 | group, 0, size, u32::from(full)),
    };
    0x0d00_0000
        | (q << 30)
        | (u32::from(load) << 22)
        | (((count - 1) & 1) << 21)
        | (opcode << 13)
        | (s << 12)
        | (element_size << 10)
}

#[test]
fn single_structure_fault_maps_preserve_completed_elements_and_defer_writeback() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for size in 0..4 {
            for (lane, load) in [
                (Some(0), false),
                (Some((16 >> size) - 1), true),
                (None, true),
            ] {
                // SUBS; LD4/ST4 lane or LD4R, starting at V31, post-index; BRK.
                let transfer = single_structure_word(4, size, lane, load, false)
                    | (1 << 23)
                    | (31 << 16)
                    | (1 << 5)
                    | 31;
                let memory = super::memory(&[0xf100_04a5, transfer, 0xd420_0000]);
                let fragment = Fragment::capture(&memory, key()).unwrap();
                let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
                let lowered = compiler
                    .lower(&fragment, CodeVersion::new(1).unwrap())
                    .unwrap();
                assert_eq!(lowered.faults.len(), 4);
                for (index, fault) in lowered.faults.iter().enumerate() {
                    assert_eq!(fault.subaccess, index as u16);
                    assert_eq!(fault.commit_stage, index as u16);
                    assert_eq!(fault.bytes, 1 << size);
                    assert!(fault.completed_read.is_none());
                    let state = &lowered.states[fault.state_map as usize].state;
                    assert!(state.dirty_live.integer.x[1]);
                    assert!(matches!(state.nzcv, NzcvLocation::Deferred(_)));
                    assert!(state.host_fpsr_pending && state.dirty_live.fpsr);
                    for (element, register) in [31, 0, 1, 2].into_iter().enumerate() {
                        assert_eq!(
                            state.dirty_live.vector[register],
                            lowered.entry.live_in.vector[register] || (load && element < index)
                        );
                    }
                    if index > 0 {
                        assert!(lowered.faults[index - 1].native_end <= fault.native_start);
                    }
                    state.validate().unwrap();
                }
                let exit = &lowered
                    .states
                    .iter()
                    .find(|map| map.exit.is_some())
                    .unwrap()
                    .state;
                assert!(exit.dirty_live.integer.x[1]);
                for register in [31, 0, 1, 2] {
                    assert_eq!(
                        exit.dirty_live.vector[register],
                        lowered.entry.live_in.vector[register] || load
                    );
                }
                let clif = compiler.context.func.display().to_string();
                assert_eq!(clif.matches("nixe_fault_start").count(), 4);
                assert_eq!(clif.matches("store").count(), if load { 0 } else { 4 });
                assert!(!clif.contains("call"));
                assert!(!clif.contains("i128"));
            }
        }
    }
}

#[test]
fn ordered_scalar_ram_transfers_match_interpreter() {
    for size in 0..4 {
        for form in [0x08df_fc00, 0x089f_fc00] {
            for rn in [1, 31] {
                for rt in [0, 1, 31] {
                    let word = form | (size << 30) | (rn << 5) | rt;
                    // Naturally aligned, ending at and starting after a page boundary.
                    check_transfer_at(word, &[0, 8, 4056, 4064]);
                }
            }
        }
    }
}

#[test]
fn ordered_fault_maps_name_only_the_access_and_keep_rcsc_ordering() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for size in 0..4 {
            // SUBS; STLR X/W3,[X1]; LDAR X/W0,[X1]; ADC; BRK.
            let words = [
                0xf100_04a5,
                0x089f_fc23 | (size << 30),
                0x08df_fc20 | (size << 30),
                0x9a1f_00c6,
                0xd420_0000,
            ];
            let memory = super::memory(&words);
            let fragment = Fragment::capture(&memory, key()).unwrap();
            let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
            let lowered = compiler
                .lower(&fragment, CodeVersion::new(1).unwrap())
                .unwrap();
            assert_eq!(lowered.faults.len(), 2);
            for (i, fault) in lowered.faults.iter().enumerate() {
                assert_eq!(fault.bytes, 1 << size);
                assert_eq!(fault.subaccess, 0);
                assert_eq!(fault.commit_stage, 0);
                assert!(fault.completed_read.is_none());
                assert_eq!(
                    fault.access,
                    if i == 0 {
                        crate::lifetime::unit::Access::Write
                    } else {
                        crate::lifetime::unit::Access::Read
                    }
                );
                let state = &lowered.states[fault.state_map as usize].state;
                assert!(state.dirty_live.integer.x[5]);
                assert!(!state.dirty_live.integer.x[0]);
                assert!(matches!(state.nzcv, NzcvLocation::Deferred(_)));
            }
            let store = &lowered.faults[0];
            let load = &lowered.faults[1];
            let bytes = &lowered.output.bytes;
            if abi == HostAbi::X86_64 {
                // Cranelift emits MOV followed by MFENCE for an atomic_store.
                // The immutable fault interval must exclude the non-faulting fence.
                // Allocator spill/reload moves may occur between them.
                assert!(
                    bytes[store.native_end as usize..load.native_start as usize]
                        .windows(3)
                        .any(|inst| inst == [0x0f, 0xae, 0xf0]),
                    "missing store/load fence: store {}..{}, load {}, bytes {bytes:02x?}",
                    store.native_start,
                    store.native_end,
                    load.native_start,
                );
            } else {
                let instruction =
                    |start| u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
                assert_eq!(
                    instruction(store.native_start as usize) & 0xffff_fc00,
                    0x089f_fc00 | (size << 30)
                ); // STLR[B/H]
                assert_eq!(
                    instruction(load.native_start as usize) & 0xffff_fc00,
                    0x08df_fc00 | (size << 30)
                ); // LDAR[B/H]
                assert_eq!(store.native_end - store.native_start, 4);
                assert_eq!(load.native_end - load.native_start, 4);
            }
            let clif = compiler.context.func.display().to_string();
            assert_eq!(clif.matches("atomic_store").count(), 1);
            assert_eq!(clif.matches("atomic_load").count(), 1);
            assert_eq!(
                clif.matches("store").count(),
                1,
                "no eager checkpoints: {clif}"
            );
            assert_eq!(clif.matches("nixe_fault_start").count(), 2);
            assert!(!clif.contains("call"));
            if size != 0 {
                assert!(
                    clif.matches(" = band ").count() >= 4,
                    "ordered alignment must not depend on host SCTLR: {clif}"
                );
            }
        }
    }
}

#[test]
fn single_vector_ram_transfers_match_interpreter_without_fp_effects() {
    for (size, opc) in [(0, 0), (1, 0), (2, 0), (3, 0), (0, 2)] {
        for load in [0, 1] {
            let transfer = (size << 30) | ((opc + load) << 22);
            for register in [0, 31] {
                for form in [
                    0x3d00_0000 | (1 << 10),                         // unsigned scaled immediate
                    0x3c00_0000 | (0x1f0 << 12),                     // unscaled -16
                    0x3c00_0400 | (16 << 12),                        // post-index +16
                    0x3c00_0c00 | (16 << 12),                        // pre-index +16
                    0x3c20_0800 | (2 << 16) | (2 << 13),             // UXTW
                    0x3c20_0800 | (2 << 16) | (3 << 13) | (1 << 12), // LSL scaled
                    0x3c20_0800 | (2 << 16) | (6 << 13) | (1 << 12), // SXTW scaled
                    0x3c20_0800 | (2 << 16) | (7 << 13),             // SXTX
                ] {
                    check_transfer(form | transfer | (1 << 5) | register);
                }
                check_transfer(0x3d00_0000 | transfer | (31 << 5) | register); // SP base
            }
        }
    }
}

#[test]
fn scalar_and_vector_pairs_match_interpreter() {
    for vector in [false, true] {
        for size in 0..3 {
            for load in [0, 1] {
                if !vector && size == 1 && load == 0 {
                    continue;
                }
                for mode in 0..4 {
                    if !vector && size == 1 && mode == 0 {
                        continue;
                    }
                    for immediate in [2, 0x7e] {
                        let form = (if vector { 0x2c00_0000 } else { 0x2800_0000 })
                            | (size << 30)
                            | (load << 22)
                            | (mode << 23)
                            | (immediate << 15);
                        for (rn, rt, rt2) in [(1, 0, 3), (31, 31, 0), (1, 0, 31)] {
                            check_transfer(form | (rn << 5) | rt | (rt2 << 10));
                        }
                    }
                }
            }
        }
    }
    // Offset addressing permits a load destination to alias the base. Both
    // addresses must have been computed from the PRE base, not the first load.
    check_transfer(0xa940_0021); // LDP X1,X0,[X1]
}

#[test]
fn pair_fault_maps_retain_uncommitted_reads_and_completed_store_stages() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for vector in [false, true] {
            for size in 0..3 {
                for load in [0, 1] {
                    if !vector && size == 1 && load == 0 {
                        continue;
                    }
                    let pair = (if vector { 0x2c00_0000 } else { 0x2800_0000 })
                        | (size << 30)
                        | (load << 22)
                        | (3 << 23)
                        | (2 << 15)
                        | (3 << 10)
                        | (1 << 5);
                    // Dirty both destinations before the pair; faults must
                    // reconstruct those old values, not a partial pair load.
                    let words = if vector {
                        vec![0x4f00_e400, 0x4f00_e403, pair, 0xd420_0000]
                    } else {
                        vec![0xd280_00e0, 0xd280_0103, pair, 0xd420_0000]
                    };
                    let memory = super::memory(&words);
                    let fragment = Fragment::capture(&memory, key()).unwrap();
                    let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
                    let lowered = compiler
                        .lower(&fragment, CodeVersion::new(1).unwrap())
                        .unwrap();
                    assert_eq!(lowered.faults.len(), 2);
                    for (i, fault) in lowered.faults.iter().enumerate() {
                        assert_eq!(fault.subaccess, i as u16);
                        assert_eq!(fault.commit_stage, if load == 0 { i as u16 } else { 0 });
                        assert_eq!(fault.completed_read.is_some(), load == 1 && i == 1);
                        let state = &lowered.states[fault.state_map as usize].state;
                        assert!(state.dirty_live.integer.x[1]);
                        let dirty = if vector {
                            &state.dirty_live.vector[..]
                        } else {
                            &state.dirty_live.integer.x[..]
                        };
                        assert!(dirty[0] && dirty[3]);
                        assert!(state.host_fpsr_pending && state.dirty_live.fpsr);
                        if let Some(location) = fault.completed_read {
                            assert!(location.valid(abi, fault.bytes));
                            let backend = &lowered.output.metadata.faults[i];
                            let value = backend.values.last().unwrap();
                            assert_eq!(value.ty.bytes(), u32::from(fault.bytes));
                            let expected = match value.location {
                                Location::Register { index, vector } => {
                                    crate::abi::ValueLocation::Register {
                                        class: if vector {
                                            crate::abi::RegisterClass::Vector
                                        } else {
                                            crate::abi::RegisterClass::Integer
                                        },
                                        index,
                                    }
                                }
                                Location::Spill { offset } => crate::abi::ValueLocation::Spill {
                                    offset,
                                    bytes: fault.bytes,
                                },
                                Location::Unused => panic!("first read lost at second access"),
                            };
                            assert_eq!(location, expected);
                        }
                    }
                    assert!(lowered.faults[0].native_end <= lowered.faults[1].native_start);
                    assert_eq!(lowered.faults[0].instruction, lowered.faults[1].instruction);
                    let clif = compiler.context.func.display().to_string();
                    assert_eq!(clif.matches("nixe_arena_addr").count(), 2);
                    assert_eq!(clif.matches("nixe_fault_start").count(), 2);
                    assert_eq!(clif.matches("store").count(), if load == 0 { 2 } else { 0 });
                    assert!(!clif.contains("i128") || !vector);
                    assert!(!clif.contains("call"));
                }
            }
        }
    }
}

#[test]
fn pair_prefault_maps_keep_dirty_spills_and_the_first_read_under_pressure() {
    for pair in [0xa940_0c20, 0xad40_0c20] {
        // LDP X0,X3 / Q0,Q3,[X1]
        let mut words = Vec::new();
        for register in 0..31 {
            words.push(0x9100_0400 | (register << 5) | register); // ADD Xn,Xn,#1
        }
        for register in 0..32 {
            words.push(0x4e20_8400 | (register << 16) | (register << 5) | register);
        }
        words.extend([pair, 0xd420_0000]);
        let memory = super::memory(&words);
        let fragment = Fragment::capture(&memory, key()).unwrap();
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
            let lowered = compiler
                .lower(&fragment, CodeVersion::new(1).unwrap())
                .unwrap();
            assert_eq!(lowered.faults.len(), 2);
            let second = &lowered.faults[1];
            let state = &lowered.states[second.state_map as usize].state;
            assert!(state.dirty_live.integer.x.iter().all(|dirty| *dirty));
            assert!(state.dirty_live.vector.iter().all(|dirty| *dirty));
            assert!(state.bindings.iter().any(|binding| matches!(
                binding.location,
                crate::abi::ValueLocation::Spill { .. }
            )));
            assert!(second.completed_read.unwrap().valid(abi, second.bytes));
            state.validate().unwrap();
        }
    }
}

#[test]
fn vector_prefault_maps_preserve_previous_vector_and_pre_writeback_base() {
    // MOVI V0.16B,#0; LDR Q0,[X1],#16; STR Q0,[X1,#16]!; BRK.
    let memory = super::memory(&[0x4f00_e400, 0x3cc1_0420, 0x3c81_0c20, 0xd420_0000]);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
        let lowered = compiler
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(lowered.faults.len(), 2);
        for (i, fault) in lowered.faults.iter().enumerate() {
            assert_eq!(fault.bytes, 16);
            let state = &lowered.states[fault.state_map as usize].state;
            assert!(state.dirty_live.vector[0]);
            assert!(state.dirty_live.integer.x[1]);
            assert!(state.host_fpsr_pending && state.dirty_live.fpsr);
            let map = &lowered.output.metadata.faults[i];
            assert_eq!(
                fault.native_end - fault.native_start,
                u32::from(map.fault_bytes)
            );
            assert!(map.values.iter().all(|value| value.ty != types::I128));
        }
        let clif = compiler.context.func.display().to_string();
        assert_eq!(clif.matches("nixe_fault_start").count(), 2);
        assert_eq!(clif.matches("store").count(), 1, "{clif}");
        assert!(!clif.contains("call"));
        assert!(!clif.contains("i128"));
    }
}

#[test]
fn scalar_prefault_maps_keep_pre_writeback_state_and_lazy_flags() {
    let words = [0xf100_04a5, 0xf840_8420, 0xf800_8c23, 0xd420_0000];
    let memory = super::memory(&words);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
        let lowered = compiler
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(lowered.faults.len(), 2);
        for (i, fault) in lowered.faults.iter().enumerate() {
            assert_eq!(
                fault.instruction,
                InstructionKey::new(
                    key()
                        .at(GuestVirtualAddress::new(PC + 4 + i as u64 * 4))
                        .unwrap()
                )
                .unwrap()
            );
            assert_eq!(fault.bytes, 8);
            assert_eq!(fault.subaccess, 0);
            assert_eq!(fault.commit_stage, 0);
            let state = &lowered.states[fault.state_map as usize].state;
            assert!(state.dirty_live.integer.x[5]);
            assert_eq!(state.dirty_live.integer.x[0], i == 1);
            assert!(state.dirty_live.integer.x[1]);
            assert!(matches!(state.nzcv, NzcvLocation::Deferred(_)));
            state.validate().unwrap();
        }
        let clif = compiler.context.func.display().to_string();
        assert_eq!(clif.matches("nixe_arena_addr").count(), 2);
        assert_eq!(clif.matches("nixe_fault_start").count(), 2);
        assert_eq!(
            clif.matches("store").count(),
            1,
            "no eager canonical checkpoint: {clif}"
        );
        assert!(!clif.contains("call"));
    }
}

#[test]
fn scalar_prefault_maps_include_prior_native_fp_effects() {
    let memory = super::memory(&[0x1e62_2820, 0xf840_8420, 0xd420_0000]);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let mut compiler = Compiler::for_arena(abi, ARENA).unwrap();
        let lowered = compiler
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(lowered.faults.len(), 1);
        let state = &lowered.states[lowered.faults[0].state_map as usize].state;
        assert!(state.host_fpsr_pending);
        assert!(state.dirty_live.fpsr);
        assert!(state.dirty_live.vector[0]);
        assert!(!state.dirty_live.integer.x[0]);
        assert!(state.dirty_live.integer.x[1]);
    }
}
