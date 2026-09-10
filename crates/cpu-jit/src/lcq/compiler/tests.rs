use super::*;
use crate::abi::{BlockKey, FpSpecialization, NativeFrame, PollBudget};
use crate::lifetime::compile::Request;
use nixe_cpu::{
    memory::{MemoryPermissions, SyntheticMemory},
    platform::TargetPlatform,
    profile::ProcessCpuContext,
    state::a64::{A64State, Nzcv},
};
use nixe_memory::{AddressSpaceId, GuestPhysicalPageId};

mod fp;
mod fp_add;
mod fp_divide;
mod fp_effects;
mod fp_fused;
mod fp_multiply;
mod fp_to_integer;
mod fp_unary;
mod fp_value;
mod integer;
mod integer_to_fp;
mod memory;
mod shape;
mod simd;
mod system;
mod vector_fp_divide;
mod vector_fp_multiply_element;
mod vector_integer_to_fp;

const PC: u64 = 0x1000;
const SPACE: AddressSpaceId = AddressSpaceId::new(1);
fn key() -> BlockKey {
    BlockKey::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
        GuestVirtualAddress::new(PC),
        FpSpecialization::Dynamic,
    )
    .unwrap()
}
fn memory(words: &[u32]) -> SyntheticMemory {
    let mut memory = SyntheticMemory::new();
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    assert!(memory.initialize_ram(page, 0, &bytes));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(PC),
        page,
        MemoryPermissions::READ_EXECUTE
    ));
    memory
}
fn native_abi() -> HostAbi {
    if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    }
}

fn execute(words: &[u32], state: &mut A64State) -> (NativeExitReason, GuestExit) {
    let memory = memory(words);
    execute_memory(&memory, words.len(), state)
}

fn execute_memory(
    memory: &SyntheticMemory,
    count: usize,
    state: &mut A64State,
) -> (NativeExitReason, GuestExit) {
    execute_compiler(memory, count, state, Compiler::new(native_abi()).unwrap())
}

fn execute_compiler(
    memory: &SyntheticMemory,
    count: usize,
    state: &mut A64State,
    compiler: Compiler,
) -> (NativeExitReason, GuestExit) {
    execute_with_fp(memory, count, state, compiler, false)
}

fn execute_with_fp(
    memory: &SyntheticMemory,
    count: usize,
    state: &mut A64State,
    mut compiler: Compiler,
    seed_host_status: bool,
) -> (NativeExitReason, GuestExit) {
    let demanded = key().at(GuestVirtualAddress::new(state.pc())).unwrap();
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    let mut reader = process.register().unwrap();
    let Request::Owner(claim) = reader.claim(demanded).unwrap() else {
        panic!()
    };
    let compilation = Compilation::capture(claim, memory).unwrap();
    let bits: Vec<_> = compilation
        .fragment
        .image
        .words()
        .iter()
        .map(|word| word.bits)
        .collect();
    let handle = compiler
        .publish(compilation, &process, &cache, memory)
        .unwrap_or_else(|error| panic!("LCQ compilation of {bits:08x?}: {error:?}"));
    let snapshot = process.snapshot(handle).unwrap();
    assert_eq!(snapshot.instructions.len(), count);
    assert!(
        (crate::abi::TRANSFER_BYTES..=crate::abi::SPILL_BYTES)
            .contains(&snapshot.code.metadata.frame_extent)
    );
    let mut frame = NativeFrame::new(state, PollBudget::new(4096, 1000).unwrap());
    let mut invocation = unsafe { reader.admit(&mut frame, demanded) }
        .unwrap()
        .unwrap();
    let entry = invocation.payload().preferred().unwrap();
    let reason = unsafe {
        if seed_host_status {
            invocation.frame().ensure_fp().unwrap();
            crate::fp_env::tests::divide_by_zero();
        }
        crate::native::enter_protected(
            invocation.frame(),
            std::ptr::null_mut(),
            entry.canonical.get() as *const u8,
        )
    }
    .unwrap()
    .reason;
    assert_eq!(invocation.frame().host_fp.active, 0);
    assert_eq!(invocation.frame().host_fp.saved, 0);
    let exit = snapshot.states[invocation.frame().exit_state_map as usize]
        .exit
        .unwrap();
    assert_eq!(
        invocation.frame().exit_source_version,
        snapshot.version.get()
    );
    drop(invocation);
    assert_eq!(frame.execution_epoch, 0);
    (reason, exit)
}

#[test]
fn bitfield_destination_is_an_input_only_for_bfm() {
    for (word, result) in [
        (0x9340_7c65, 0xffff_ffff_8765_4380), // SXTW X5, W3 (es2gears).
        (0x1300_1c65, 0x0000_0000_ffff_ff80), // SXTB W5, W3.
        (0xd340_7c65, 0x0000_0000_8765_4380), // UBFM X5, X3, #0, #31.
        (0x5300_1c65, 0x0000_0000_0000_0080), // UXTB W5, W3.
        (0xb378_1c65, 0x0123_4567_89ab_80ef), // BFI X5, X3, #8, #8.
        (0x3318_1c65, 0x0000_0000_89ab_80ef), // BFI W5, W3, #8, #8.
    ] {
        let mut expected = A64State::default();
        expected.set_pc(PC);
        expected.general_register_storage_mut()[3] = 0xfedc_ba98_8765_4380;
        expected.general_register_storage_mut()[5] = 0x0123_4567_89ab_cdef;
        expected.set_nzcv(Nzcv::from_bits(0xb000_0000));
        let mut actual = expected.clone();
        nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
        assert_eq!(expected.general_register_storage_mut()[5], result);
        // X5 has not been defined in this fragment. SBFM/UBFM must compile
        // without a live-in X5; BFM must preserve its unaffected destination bits.
        let (reason, exit) = execute(&[word, 0xd420_0000], &mut actual);
        assert_eq!(reason, NativeExitReason::Architectural);
        assert_eq!(exit.kind, EdgeKind::Breakpoint(0));
        assert_eq!(actual, expected, "bitfield {word:08x}");
    }
}

#[test]
fn decoded_integer_fragment_uses_real_cache_gateway_and_dirty_writeback() {
    let words = [
        0x1100_0400,
        0x9100_23e2,
        0x9100_43ff,
        0xaa01_03e3,
        0xaa02_003f,
        0xd420_0000,
    ];
    let mut expected = A64State::default();
    expected.set_pc(PC);
    expected
        .general_register_storage_mut()
        .fill(0x1234_5678_ffff_ffff);
    expected.set_nzcv(Nzcv::from_bits(0xb000_0000));
    let mut actual = expected.clone();
    for word in &words[..words.len() - 1] {
        nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, *word).unwrap();
    }
    let (reason, exit) = execute(&words, &mut actual);
    assert_eq!(reason, NativeExitReason::Architectural);
    assert_eq!(exit.kind, EdgeKind::Breakpoint(0));
    assert_eq!(actual, expected);
}

#[test]
fn deferred_flags_and_conditional_exits_match_interpreter() {
    for initial in [0, 1, u64::MAX, 0x7fff_ffff_ffff_ffff, 0x8000_0000_0000_0000] {
        for nzcv in [0, 0x2000_0000, 0xf000_0000] {
            for arithmetic in [
                0xb100_0400u32,
                0xf100_0400,
                0xba01_0000,
                0xfa01_0000,
                0xea01_0000,
                0xfa41_0800,
            ] {
                let words = [arithmetic, 0x5400_0040]; // B.EQ +8
                let mut expected = A64State::default();
                expected.set_pc(PC);
                expected.general_register_storage_mut()[0] = initial;
                expected.general_register_storage_mut()[1] = 1;
                expected.set_nzcv(Nzcv::from_bits(nzcv));
                let mut actual = expected.clone();
                for word in words {
                    nixe_cpu_interpreter::execute_one(
                        &TargetPlatform::Switch1,
                        &mut expected,
                        word,
                    )
                    .unwrap();
                }
                let (reason, edge) = execute(&words, &mut actual);
                assert_eq!(reason, NativeExitReason::Dispatch);
                assert!(matches!(edge.kind, EdgeKind::Taken | EdgeKind::NotTaken));
                assert_eq!(
                    actual, expected,
                    "{arithmetic:08x}, X0={initial:x}, NZCV={nzcv:x}"
                );
            }
        }
    }
}

#[test]
fn calls_and_indirect_returns_preserve_guest_lr_and_target() {
    for word in [
        0x9400_0002u32,
        0xd63f_03c0,
        0xd65f_03c0,
        0xd61f_03c0,
        0xb400_005e,
        0xb600_005e,
    ] {
        let mut expected = A64State::default();
        expected.set_pc(PC);
        expected.general_register_storage_mut()[30] = 0x1234;
        let mut actual = expected.clone();
        nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, word).unwrap();
        execute(&[word], &mut actual);
        assert_eq!(actual, expected, "{word:08x}");
    }
}

#[test]
fn both_encoders_produce_final_maps_and_canonical_ingress_without_jitmodule() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let memory = memory(&[0xb100_0400, 0x5400_0040]);
        let fragment = Fragment::capture(&memory, key()).unwrap();
        let lowered = Compiler::new(abi)
            .unwrap()
            .lower(&fragment, CodeVersion::new(7).unwrap())
            .unwrap();
        assert_eq!(lowered.states.len(), 2);
        assert!(lowered.output.metadata.faults.is_empty());
        assert!(lowered.output.metadata.frame_extent >= crate::abi::TRANSFER_BYTES);
        assert_eq!(
            &lowered.output.bytes[lowered.canonical as usize..lowered.canonical as usize + 4],
            landing(abi)
        );
        for state in &lowered.states {
            state.state.validate().unwrap();
            assert!(matches!(state.state.nzcv, NzcvLocation::Deferred(_)));
            assert_eq!(state.state.site.source.get(), 7);
        }
    }
}

#[test]
fn staged_installation_preserves_capacity_error_and_releases_the_compile_claim() {
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    let mut reader = process.register().unwrap();
    let mut compiler = Compiler::new(native_abi()).unwrap();
    let memory = memory(&[0x14000000]);
    let Request::Owner(claim) = reader.claim(key()).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    let charge = cache
        .charge_metadata(
            crate::executable::HARD_BYTES - cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    let error = compiler
        .publish(captured, &process, &cache, &memory)
        .err()
        .unwrap();
    assert!(matches!(
        error,
        PublishError::Storage(crate::executable::Error::Capacity(_))
    ));
    assert_eq!(error.capacity(), Some("640 MiB code+metadata hard limit"));
    assert_eq!(cache.usage().unwrap().committed, 0);
    drop(charge);
    let Request::Owner(claim) = reader.claim(key()).unwrap() else {
        panic!("failed publication retained its claim")
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    compiler
        .publish(captured, &process, &cache, &memory)
        .unwrap();
}

#[test]
fn stale_or_unported_work_is_never_published_and_compiler_is_reusable() {
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    let mut reader = process.register().unwrap();
    let mut compiler = Compiler::new(native_abi()).unwrap();
    for words in [&[0x1400_0000u32][..], &[0xf940_0000, 0x1400_0000][..]] {
        let mut memory = memory(words);
        let Request::Owner(claim) = reader.claim(key()).unwrap() else {
            panic!()
        };
        let captured = Compilation::capture(claim, &memory).unwrap();
        if words.len() == 1 {
            assert!(memory.initialize_ram(
                GuestPhysicalPageId::new(1),
                0,
                &0xd420_0000u32.to_le_bytes(),
            ));
        }
        let result = compiler.publish(captured, &process, &cache, &memory);
        if words.len() == 1 {
            assert!(matches!(result, Err(PublishError::StaleCapture)));
        } else {
            assert!(matches!(result, Err(PublishError::Lowering(_))));
        }
        assert_eq!(cache.usage().unwrap().committed, 0);
    }
    let memory = memory(&[0x1400_0000]);
    let Request::Owner(claim) = reader.claim(key()).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    compiler
        .publish(captured, &process, &cache, &memory)
        .unwrap();
}

#[test]
fn unreadable_successor_does_not_prevent_execution_of_the_valid_prefix() {
    let mut memory = memory(&[0x9100_0400]);
    memory.inject_instruction_fault(
        SPACE,
        GuestVirtualAddress::new(PC + 4),
        "unavailable successor",
    );
    let mut state = A64State::default();
    state.set_pc(PC);
    let (reason, edge) = execute_memory(&memory, 1, &mut state);
    assert_eq!(reason, NativeExitReason::Dispatch);
    assert_eq!(edge.kind, EdgeKind::Static);
    assert_eq!(state.general_register_storage_mut()[0], 1);
    assert_eq!(state.pc(), PC + 4);
    let key = key().at(GuestVirtualAddress::new(PC + 4)).unwrap();
    let fragment = Fragment::capture(&memory, key).unwrap();
    let error = Compiler::new(native_abi())
        .unwrap()
        .lower(&fragment, CodeVersion::new(1).unwrap())
        .err()
        .unwrap();
    assert!(error.to_string().contains("unavailable successor"));
}

#[test]
fn non_memory_catalog_lowers_with_final_maps_on_both_encoders() {
    use nixe_cpu::decode::a64::fp_simd::Instruction as Fp;
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let mut compiler = Compiler::new(abi).unwrap();
        let mut covered = 0;
        for pattern in decode::a64::patterns() {
            let Some(fixture) = pattern.regression_fixture else {
                continue;
            };
            let bits = fixture.encoding.bits();
            let DecodeResult::Decoded(decoded) = decode::decode(
                TargetPlatform::Switch1,
                nixe_cpu::location::LocationDescriptor::new(
                    GuestVirtualAddress::new(PC),
                    key().profile,
                ),
                bits.into(),
            ) else {
                continue;
            };
            match decode::a64::normalize(&decoded.instruction, decoded.encoding) {
                A64Instruction::Memory(_) => continue,
                A64Instruction::FpSimd(
                    Fp::MemoryUnsigned(_)
                    | Fp::MemoryUnscaled(_)
                    | Fp::MemoryPostIndex(_)
                    | Fp::MemoryPreIndex(_)
                    | Fp::MemoryRegister(_)
                    | Fp::MemoryPair(_)
                    | Fp::MemoryMultipleStructures(_)
                    | Fp::MemoryMultipleStructuresPostIndex(_)
                    | Fp::MemorySingleStructure(_)
                    | Fp::MemorySingleStructurePostIndex(_),
                ) => continue,
                _ => {}
            }
            let memory = memory(&[bits, 0xd420_0000]);
            let fragment = Fragment::capture(&memory, key()).unwrap();
            let lowered = compiler
                .lower(&fragment, CodeVersion::new(1).unwrap())
                .unwrap_or_else(|error| panic!("{abi:?}, {} ({bits:08x}): {error}", pattern.name));
            for state in &lowered.states {
                state.state.validate().unwrap();
            }
            assert!(
                !lowered.states.is_empty(),
                "{} has no attributed exit",
                pattern.name
            );
            covered += 1;
        }
        assert!(covered > 0);
    }
}

#[test]
fn integer_catalog_executes_through_new_abi_with_shared_semantics() {
    let mut covered = 0;
    for pattern in nixe_cpu::decode::a64::patterns() {
        let Some(fixture) = pattern.regression_fixture else {
            continue;
        };
        let bits = fixture.encoding.bits();
        let DecodeResult::Decoded(decoded) = decode::decode(
            TargetPlatform::Switch1,
            nixe_cpu::location::LocationDescriptor::new(
                GuestVirtualAddress::new(PC),
                key().profile,
            ),
            bits.into(),
        ) else {
            continue;
        };
        if !matches!(
            decode::a64::normalize(&decoded.instruction, decoded.encoding),
            A64Instruction::Integer(_)
        ) {
            continue;
        }
        let mut expected = A64State::default();
        expected.set_pc(PC);
        for (index, register) in expected
            .general_register_storage_mut()
            .iter_mut()
            .enumerate()
        {
            *register = 0x8000_0123_4567_89ab ^ (index as u64 * 0x1111_1111);
        }
        expected.set_nzcv(Nzcv::from_bits(0xb000_0000));
        let mut actual = expected.clone();
        nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, bits).unwrap();
        execute(&[bits, 0xd420_0000], &mut actual);
        assert_eq!(actual, expected, "{} ({bits:08x})", pattern.name);
        covered += 1;
    }
    assert!(covered >= 17);
}

#[test]
fn register_simd_catalog_matches_interpreter_without_fp_activation() {
    let mut covered = 0;
    for pattern in nixe_cpu::decode::a64::patterns() {
        let Some(fixture) = pattern.regression_fixture else {
            continue;
        };
        let bits = fixture.encoding.bits();
        let DecodeResult::Decoded(decoded) = decode::decode(
            TargetPlatform::Switch1,
            nixe_cpu::location::LocationDescriptor::new(
                GuestVirtualAddress::new(PC),
                key().profile,
            ),
            bits.into(),
        ) else {
            continue;
        };
        let A64Instruction::FpSimd(instruction) =
            decode::a64::normalize(&decoded.instruction, decoded.encoding)
        else {
            continue;
        };
        if !is_register_simd(instruction) {
            continue;
        }
        for seed in [0u128, u128::MAX, 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210] {
            let mut expected = A64State::default();
            expected.set_pc(PC);
            for (index, register) in expected
                .general_register_storage_mut()
                .iter_mut()
                .enumerate()
            {
                *register = seed as u64 ^ (index as u64 * 0x1111_1111);
            }
            for (index, register) in expected
                .vector_register_storage_mut()
                .iter_mut()
                .enumerate()
            {
                *register = seed ^ u128::from_le_bytes([index as u8 * 7; 16]);
            }
            expected.set_nzcv(Nzcv::from_bits(0xb000_0000));
            // Bit/lane operations also work with controls which native FP
            // arithmetic cannot activate, and must preserve sticky status.
            expected.set_fpcr((3 << 22) | (1 << 8));
            expected.set_fpsr(0x0800_009f);
            let mut actual = expected.clone();
            nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, bits)
                .unwrap();
            execute(&[bits, 0xd420_0000], &mut actual);
            assert_eq!(
                actual, expected,
                "{} ({bits:08x}), seed {seed:032x}",
                pattern.name
            );
        }
        // The public x86 host requirements do not require SSSE3/SSE4/AVX.
        // Check the conservative encoder too, including its inline shuffles.
        let mut baseline = Compiler::new(HostAbi::X86_64).unwrap();
        baseline.isa = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap())
            .unwrap()
            .finish(baseline.isa.flags().clone())
            .unwrap();
        let memory = memory(&[bits, 0xd420_0000]);
        let fragment = Fragment::capture(&memory, key()).unwrap();
        baseline
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap_or_else(|error| panic!("baseline x86 {}: {error:?}", pattern.name));
        if cfg!(target_arch = "x86_64") {
            let mut expected = A64State::default();
            expected.set_pc(PC);
            expected
                .vector_register_storage_mut()
                .fill(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);
            let mut actual = expected.clone();
            nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, bits)
                .unwrap();
            execute_compiler(&memory, 2, &mut actual, baseline);
            assert_eq!(actual, expected, "baseline x86 {}", pattern.name);
        }
        covered += 1;
    }
    assert!(covered >= 34, "only {covered} SIMD fixtures covered");
}

#[test]
fn mixed_vector_spills_preserve_lazy_flags_and_partial_register_writes() {
    let mut words: Vec<u32> = (0..31).map(|reg| 0x9100_0400 | (reg << 5) | reg).collect();
    words.push(0xf100_041f); // CMP X0, #1: a lazy recipe spanning the SIMD work.
    words.extend((0..32).map(|reg| 0x6e3f_1c00 | (reg << 5) | reg)); // EOR Vd.16B,Vd.16B,V31.16B
    // FCSEL consumes Z without modifying flags; INS preserves unselected lanes.
    words.extend([0x1e62_0c20, 0x4e18_1c20, 0xd420_0000]);
    let memory = memory(&words);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let mut compiler = Compiler::new(abi).unwrap();
        if abi == HostAbi::X86_64 {
            compiler.isa = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap())
                .unwrap()
                .finish(compiler.isa.flags().clone())
                .unwrap();
        }
        let lowered = compiler
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        let exit = &lowered.states[0].state;
        assert!(matches!(exit.nzcv, NzcvLocation::Deferred(_)));
        assert!(exit.dirty_live.vector.iter().all(|dirty| *dirty));
        assert!(
            abi == HostAbi::Aarch64
                || exit.bindings.iter().any(|binding| matches!(
                    binding.value,
                    GuestValue::Vector(_)
                ) && matches!(
                    binding.location,
                    crate::abi::ValueLocation::Spill { .. }
                ))
        );
    }
    for x0 in [0, 7] {
        let mut expected = A64State::default();
        expected.set_pc(PC);
        expected.general_register_storage_mut().fill(x0);
        for (index, vector) in expected
            .vector_register_storage_mut()
            .iter_mut()
            .enumerate()
        {
            *vector = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128.rotate_left(index as u32);
        }
        expected.set_fpcr(1 << 8);
        expected.set_fpsr(0x0800_009f);
        let mut actual = expected.clone();
        let initial = expected.clone();
        for bits in &words[..words.len() - 1] {
            nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, *bits)
                .unwrap();
        }
        execute(&words, &mut actual);
        assert_eq!(actual, expected);
        if cfg!(target_arch = "x86_64") {
            let mut compiler = Compiler::new(HostAbi::X86_64).unwrap();
            compiler.isa = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap())
                .unwrap()
                .finish(compiler.isa.flags().clone())
                .unwrap();
            let mut actual = initial;
            execute_compiler(&memory, words.len(), &mut actual, compiler);
            assert_eq!(actual, expected, "baseline x86 vector spill execution");
        }
    }
}

#[test]
fn pressure_uses_final_spill_maps_on_both_encoders() {
    let mut words: Vec<u32> = (0..31)
        .map(|register| 0x9100_0400 | (register << 5) | register)
        .collect();
    words.push(0xd420_0000);
    let memory = memory(&words);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let lowered = Compiler::new(abi)
            .unwrap()
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        assert!(lowered.output.metadata.frame_extent > crate::abi::TRANSFER_BYTES);
        assert!(
            lowered.states[0]
                .state
                .bindings
                .iter()
                .any(|binding| matches!(binding.location, crate::abi::ValueLocation::Spill { .. }))
        );
    }
    let mut expected = A64State::default();
    expected.set_pc(PC);
    for (index, register) in expected
        .general_register_storage_mut()
        .iter_mut()
        .enumerate()
    {
        *register = index as u64;
    }
    let mut actual = expected.clone();
    for word in &words[..31] {
        nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, *word).unwrap();
    }
    execute(&words, &mut actual);
    assert_eq!(actual, expected);
}

#[test]
fn unsupported_identity_is_retained_without_executing_it() {
    let pattern = nixe_cpu::decode::a64::patterns()
        .iter()
        .find(|pattern| pattern.decoder == decode::DecodeSupport::RecognizedUnimplemented)
        .unwrap();
    let bits = pattern.regression_fixture.unwrap().encoding.bits();
    let mut state = A64State::default();
    state.set_pc(PC);
    state.general_register_storage_mut().fill(0x1234);
    let before = state.clone();
    let (reason, exit) = execute(&[bits], &mut state);
    assert_eq!(reason, NativeExitReason::Unsupported);
    assert_eq!(exit.kind, EdgeKind::Unsupported);
    assert_eq!(exit.pc.get(), PC);
    assert_eq!(state, before);
}
