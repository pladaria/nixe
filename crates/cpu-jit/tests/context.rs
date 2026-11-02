//! Native process execution and complete architectural context ownership.
use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State};
use nixe_cpu_jit::{JitProcess, JitRunRequest, JitThread, ThreadState};
use nixe_memory::{
    AddressSpace, AddressWidth, AllocationRange, GuestVirtualAddress, Machine, MemoryPermissions,
    PAGE_SIZE,
};
use std::sync::Arc;
const CODE: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);
struct FixedTimer;
impl ArchitecturalTimer for FixedTimer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}
fn process_fixture(code: &[u32]) -> (Arc<AddressSpace>, Arc<JitProcess>) {
    let code_bytes = (code.len() as u64 * 4).div_ceil(PAGE_SIZE).max(1) * PAGE_SIZE;
    let machine = Machine::new(code_bytes + PAGE_SIZE).unwrap();
    let account = machine.account(code_bytes + PAGE_SIZE).unwrap();
    let executable = account.allocate(code_bytes).unwrap();
    let data = account.allocate(PAGE_SIZE).unwrap();
    let bytes: Vec<_> = code.iter().flat_map(|word| word.to_le_bytes()).collect();
    machine
        .write_resolved(
            &[AllocationRange::new(executable.pin(), 0, bytes.len() as u64).unwrap()],
            &bytes,
        )
        .unwrap();
    let memory = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
    memory
        .map(
            CODE,
            executable,
            0,
            code_bytes,
            MemoryPermissions::READ_EXECUTE,
        )
        .unwrap();
    memory
        .map(
            GuestVirtualAddress::new(CODE.get() + code_bytes),
            data,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let process = JitProcess::new(
        ProcessCpuContext::for_platform(TargetPlatform::Switch1, memory.identity()),
        memory.clone(),
        19_200_000,
    )
    .unwrap();
    (memory, process)
}
fn fixture(words: &[u32]) -> (Arc<AddressSpace>, JitThread) {
    let (memory, process) = process_fixture(words);
    (memory, process.create_thread().unwrap())
}
fn x(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}
fn a64_state() -> A64State {
    let mut state = A64State::default();
    state.set_pc(CODE.get());
    state
}

fn pattern(seed: u64) -> A64State {
    let mut state = a64_state();
    for i in 0..31 {
        state.write_x(
            x(i),
            seed.wrapping_mul(0x123456789abcdef)
                .rotate_left(u32::from(i)),
        );
    }
    state.write_x(A64Register::StackPointer, 0x8000 + seed * 16);
    for i in 0..32 {
        state.set_vector(
            i,
            (u128::from(seed + u64::from(i)) << 96) | u128::from(0xfedcba9876543210_u64 ^ seed),
        );
    }
    state.set_tpidr_el0(0xabc000 + seed);
    state.set_tpidrro_el0_from_runtime(0xdef000 + seed);
    state.set_nzcv(nixe_cpu::state::a64::Nzcv::from_bits((seed as u32) << 28));
    state.set_fpcr(((seed as u32) & 3) << 22);
    state.set_fpsr(0x08000000 | ((seed as u32) & 0x1f));
    state
}

fn run(jit: &mut JitThread, state: &mut ThreadState) -> nixe_cpu::execution::ExecutionReport {
    jit.run_slice(
        &mut nixe_cpu_direct_memory::NativeWorker::default(),
        JitRunRequest {
            state,
            timer: &FixedTimer,
        },
    )
    .unwrap()
}

#[test]
fn overlapping_decoder_patterns_preserve_simd_immediates_and_bitfields() {
    // SIMD modified-immediate encodings overlap wider unallocated patterns.
    // Exercise both priority exceptions and the split scalar immediate fields.
    let (_memory, mut jit) = fixture(&[
        0x4f02e440, // MOVI V0.16B,#0x42
        0x6f002641, // MVNI V1.4S,#0x12,LSL#8
        0x4f003641, // ORR V1.4S,#0x12,LSL#8
        0x6f043401, // BIC V1.4S,#0x80,LSL#8
        0x4f03f602, // FMOV V2.4S,#1.0
        0x6f04f403, // FMOV V3.2D,#-2.0
        0x92a24684, // MOVN X4,#0x1234,LSL#16
        0xd3507c85, // UBFM X5,X4,#16,#31
        0xd4000841, // SVC #0x42
    ]);
    let mut state: ThreadState = a64_state().into();
    assert!(matches!(
        run(&mut jit, &mut state).stop,
        CpuExit::SupervisorCall {
            immediate: 0x42,
            ..
        }
    ));
    assert_eq!(state.vector(0), Some(u128::from_le_bytes([0x42; 16])));
    assert_eq!(state.vector(1), Some(0xffff7fff_ffff7fff_ffff7fff_ffff7fff));
    assert_eq!(state.vector(2), Some(0x3f800000_3f800000_3f800000_3f800000));
    assert_eq!(state.vector(3), Some(0xc000000000000000_c000000000000000));
    assert_eq!(state.read_x(x(4)), !(0x1234_u64 << 16));
    assert_eq!(state.read_x(x(5)), 0xedcb);
}

#[test]
fn direct_memory_exclusives_simd_fp_tls_and_system_registers() {
    // STR/LDR, LDXR/ADD/STXR, FP and SIMD, TLS/cache/timer registers, SVC #0x42.
    let code = [
        0xf9000020, 0xf9400022, 0xc85f7c23, 0x91000463, 0xc8047c23, 0xf9400025, 0x1e6e1000,
        0x1e602801, 0x4f02e442, 0xd53bd046, 0xd53bd067, 0xd53be008, 0xd53be029, 0xd53b002a,
        0xd53b00eb, 0xd4000841,
    ];
    let (_memory, mut jit) = fixture(&code);
    let mut state: ThreadState = a64_state().into();
    state.write_x(x(0), 41);
    state.write_x(x(1), 0x2000);
    state.set_tpidr_el0(0xabcdef);
    state.set_tpidrro_el0_from_runtime(0x123456);
    let report = run(&mut jit, &mut state);
    assert!(matches!(
        report.stop,
        CpuExit::SupervisorCall {
            immediate: 0x42,
            ..
        }
    ));
    assert_eq!(state.pc(), CODE.get() + (code.len() as u64 - 1) * 4);
    assert_eq!(state.read_x(x(2)), 41);
    assert_eq!(state.read_x(x(4)), 0);
    assert_eq!(state.read_x(x(5)), 42);
    assert_eq!(state.vector(1), Some(u128::from(2.0_f64.to_bits())));
    assert_eq!(state.vector(2), Some(u128::from_le_bytes([0x42; 16])));
    assert_eq!(state.read_x(x(6)), 0xabcdef);
    assert_eq!(state.read_x(x(7)), 0x123456);
    assert_eq!(state.read_x(x(8)), 19_200_000);
    assert_eq!(state.read_x(x(9)), 0);
    assert_eq!(state.read_x(x(10)), 0x8444c004);
    assert_eq!(state.read_x(x(11)), 4);
}

#[test]
fn memory_widths_keep_values_and_observers_across_fault_recompilation() {
    // STR/LDR and STLR/LDAR at every scalar width, plus 128-bit SIMD transfers.
    // Reuse each block through observed writes, reopened RAM and invalid addresses.
    for (store, load, width) in [
        (0x39000020, 0x39400022, 1),
        (0x79000020, 0x79400022, 2),
        (0xb9000020, 0xb9400022, 4),
        (0xf9000020, 0xf9400022, 8),
        (0x089ffc20, 0x08dffc22, 1),
        (0x489ffc20, 0x48dffc22, 2),
        (0x889ffc20, 0x88dffc22, 4),
        (0xc89ffc20, 0xc8dffc22, 8),
        (0x3d800020, 0x3dc00022, 16),
    ] {
        let (memory, mut jit) = fixture(&[store, load, 0xd4000841]);
        let address = GuestVirtualAddress::new(0x2000);
        for (observed, valid) in [(true, true), (true, true), (false, true), (false, false)] {
            memory
                .write_value(address, 16, u128::from_le_bytes([0x73; 16]))
                .unwrap()
                .into_ready()
                .unwrap();
            let observation = nixe_memory::Observation::default();
            if observed {
                let ranges = memory
                    .capture(address, 16, MemoryPermissions::READ)
                    .unwrap();
                memory
                    .machine()
                    .access_captured(
                        &ranges,
                        nixe_memory::AccessOperation::ReadObserved(observation.clone()),
                    )
                    .unwrap()
                    .into_ready()
                    .unwrap();
            }
            let value = 0xfedc_ba98_7654_3210_1234_5678_9abc_def0_u128;
            let mut state: ThreadState = a64_state().into();
            state.write_x(x(0), value as u64);
            state.set_vector(0, value);
            state.write_x(x(1), if valid { address.get() } else { 0x1_0000_2000 });
            assert!(matches!(
                run(&mut jit, &mut state).stop,
                CpuExit::SupervisorCall {
                    immediate: 0x42,
                    ..
                }
            ));
            let mut expected = [0x73; 16];
            if valid {
                expected[..width].copy_from_slice(&value.to_le_bytes()[..width]);
            }
            assert_eq!(
                memory
                    .read_value(address, 16)
                    .unwrap()
                    .into_ready()
                    .unwrap(),
                u128::from_le_bytes(expected)
            );
            let loaded = if valid {
                value & (u128::MAX >> ((16 - width) * 8))
            } else {
                0
            };
            if width == 16 {
                assert_eq!(state.vector(2), Some(loaded));
            } else {
                assert_eq!(state.read_x(x(2)), loaded as u64);
            }
            assert_eq!(observation.is_invalid(), observed);
        }
    }
}

#[test]
fn branch_terminals_select_leaf_exits_for_all_flags_and_register_widths() {
    // B.cond, CBZ/CBNZ and TBZ/TBNZ choose leaf exits; SVC then returns through
    // CheckHalt. Reuse the emitted blocks with new flags and operand values.
    // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/B-cond--Branch-conditionally-
    // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/CBZ--Compare-and-Branch-on-Zero-
    // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/TBZ--Test-bit-and-Branch-if-Zero-
    let mut branches: Vec<u32> = (0..16).map(|condition| 0x5400_0060 | condition).collect();
    branches.extend([0x3400_0060, 0x3500_0060, 0xb400_0060, 0xb500_0060]);
    for bit in [0_u32, 31, 32, 63] {
        let branch = 0x3600_0060 | ((bit >> 5) << 31) | ((bit & 31) << 19);
        branches.extend([branch, branch | 0x0100_0000]);
    }
    let code: Vec<u32> = branches
        .iter()
        .flat_map(|&branch| [branch, 0xd280_0161, 0x1400_0002, 0xd280_02c1, 0xd400_0841])
        .collect();
    let (_memory, mut jit) = fixture(&code);
    for (index, branch) in branches.iter().enumerate() {
        for flags in 0_u32..16 {
            let [n, z, c, v] = [8, 4, 2, 1].map(|bit| flags & bit != 0);
            let conditions = [
                z,
                !z,
                c,
                !c,
                n,
                !n,
                v,
                !v,
                c && !z,
                !c || z,
                n == v,
                n != v,
                !z && n == v,
                z || n != v,
                true,
                true,
            ];
            for operand in [0_u64, 1, 1 << 31, 1 << 32, 1 << 63, u64::MAX] {
                let taken = match index {
                    0..16 => conditions[index],
                    16..20 => {
                        let zero = if index < 18 {
                            operand as u32 == 0
                        } else {
                            operand == 0
                        };
                        zero ^ (index & 1 != 0)
                    }
                    _ => {
                        let bit = [0, 31, 32, 63][(index - 20) / 2];
                        (operand & (1 << bit) == 0) ^ (index & 1 != 0)
                    }
                };
                let mut state = a64_state();
                state.set_pc(CODE.get() + index as u64 * 20);
                state.set_nzcv(nixe_cpu::state::a64::Nzcv::from_bits(flags << 28));
                state.write_x(x(0), operand);
                let mut state: ThreadState = state.into();
                assert!(matches!(
                    run(&mut jit, &mut state).stop,
                    CpuExit::SupervisorCall {
                        immediate: 0x42,
                        ..
                    }
                ));
                assert_eq!(
                    state.read_x(x(1)),
                    if taken { 22 } else { 11 },
                    "branch={branch:#010x} flags={flags:#x} operand={operand:#x}"
                );
            }
        }
    }
}

#[test]
fn deferred_floating_point_branches_keep_nan_payloads_across_blocks() {
    // FSQRT's NaN path and each STR retain forward branches until block emission.
    // https://developer.arm.com/documentation/ddi0602/latest/SIMD-FP-Instructions/FSQRT--scalar---Floating-point-square-root--scalar--
    for (sqrt, store, width, finite, result, negative, qnan, snan, quiet, default_nan) in [
        (
            0x1e21c002,
            0xbd000022,
            4,
            u64::from(9_f32.to_bits()),
            u64::from(3_f32.to_bits()),
            u64::from((-9_f32).to_bits()),
            0x7fc00041,
            0x7f800042,
            0x00400000,
            0x7fc00000,
        ),
        (
            0x1e61c002,
            0xfd000022,
            8,
            9_f64.to_bits(),
            3_f64.to_bits(),
            (-9_f64).to_bits(),
            0x7ff8000000000041,
            0x7ff0000000000042,
            0x0008000000000000,
            0x7ff8000000000000,
        ),
    ] {
        let mut code = Vec::new();
        for index in 0..64 {
            code.extend_from_slice(&[sqrt, store | index << 10]);
        }
        code.push(0xd4000841);
        let (memory, mut jit) = fixture(&code);
        // Alternate FPCR-specialized native blocks, reusing them with new data.
        for default_mode in [false, true, false] {
            for (input, expected, invalid) in [
                (finite, result, false),
                (negative, default_nan, true),
                (qnan, if default_mode { default_nan } else { qnan }, false),
                (
                    snan,
                    if default_mode {
                        default_nan
                    } else {
                        snan | quiet
                    },
                    true,
                ),
            ] {
                let mut state: ThreadState = a64_state().into();
                state.set_fpcr(u32::from(default_mode) << 25);
                state.set_vector(0, u128::from(input));
                state.write_x(x(1), 0x2000);
                assert!(matches!(
                    run(&mut jit, &mut state).stop,
                    CpuExit::SupervisorCall {
                        immediate: 0x42,
                        ..
                    }
                ));
                assert_eq!(state.vector(2), Some(u128::from(expected)));
                assert_eq!(state.fpsr() & 1 != 0, invalid);
                for index in 0..64 {
                    assert_eq!(
                        memory
                            .read_value(
                                GuestVirtualAddress::new(0x2000 + (index * width) as u64),
                                width,
                            )
                            .unwrap()
                            .into_ready()
                            .unwrap(),
                        u128::from(expected)
                    );
                }
            }
        }
    }
}

#[test]
fn live_scalar_and_vector_values_survive_spills_fixed_registers_and_host_calls() {
    // Retain more scalar and vector inputs than the host register file can hold,
    // then call the architectural timer and use fixed-register arithmetic before
    // storing every result. This also exercises moves between register classes
    // and spill storage; all inputs remain observable after the call.
    let mut code = Vec::new();
    for register in 0..29 {
        code.push(0xf9400000 | register << 10 | 30 << 5 | register); // LDR Xn,[X30,#n*8]
    }
    for register in 0..32 {
        code.push(0x3dc00000 | (16 + register) << 10 | 30 << 5 | register); // LDR Qn
    }
    code.extend([
        0xaa0703e6, // MOV X6,X7 (both names retain the same value)
        0x9e660108, // FMOV X8,D8
        0x9e670129, // FMOV D9,X9
        0xd53be03d, // MRS X29,CNTPCT_EL0 (host callback)
        0x9ac10800, // UDIV X0,X0,X1
        0x9ac32042, // LSLV X2,X2,X3
        0x9bc57c84, // UMULH X4,X4,X5
        0x6e3f1c00, // EOR V0.16B,V0.16B,V31.16B
    ]);
    for register in 0..30 {
        code.push(0xf9000000 | register << 10 | 30 << 5 | register); // STR Xn
    }
    for register in 0..32 {
        code.push(0x3d800000 | (16 + register) << 10 | 30 << 5 | register); // STR Qn
    }
    code.push(0xd4000cc1); // SVC #0x66
    let block_words = code.len();
    // Compile distinct blocks through the reusable IR storage, then revisit a
    // cached block. No assignment from a previous emission may survive reset.
    code.extend_from_within(0..block_words);
    code.extend_from_within(0..block_words);
    let (memory, mut jit) = fixture(&code);
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    for (block, seed) in [(0, 3u64), (1, 11), (2, 29), (0, 37)] {
        let mut input = [0u8; 768];
        let scalar: [u64; 29] = std::array::from_fn(|index| {
            seed.wrapping_mul(0xfedcba987654321)
                .rotate_left(index as u32)
                | 1
        });
        for (index, value) in scalar.iter().enumerate() {
            input[index * 8..index * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }
        let vector: [u128; 32] = std::array::from_fn(|index| {
            (u128::from(seed) << 96 | 0x0123456789abcdef_fedcba9876543210).rotate_left(index as u32)
        });
        for (index, value) in vector.iter().enumerate() {
            input[256 + index * 16..272 + index * 16].copy_from_slice(&value.to_le_bytes());
        }
        memory
            .write(GuestVirtualAddress::new(0x2000), &input)
            .unwrap()
            .into_ready()
            .unwrap();
        let mut expected = input;
        expected[0..8].copy_from_slice(&(scalar[0] / scalar[1]).to_le_bytes());
        expected[16..24].copy_from_slice(&scalar[2].wrapping_shl(scalar[3] as u32).to_le_bytes());
        expected[32..40].copy_from_slice(
            &(((u128::from(scalar[4]) * u128::from(scalar[5])) >> 64) as u64).to_le_bytes(),
        );
        expected[48..56].copy_from_slice(&scalar[7].to_le_bytes());
        expected[64..72].copy_from_slice(&(vector[8] as u64).to_le_bytes());
        expected[256..272].copy_from_slice(&(vector[0] ^ vector[31]).to_le_bytes());
        expected[400..416].copy_from_slice(&u128::from(scalar[9]).to_le_bytes());
        let mut state: ThreadState = a64_state().into();
        state.set_pc(CODE.get() + (block * block_words * 4) as u64);
        state.write_x(x(30), 0x2000);
        let report = jit
            .run_slice(
                &mut worker,
                JitRunRequest {
                    state: &mut state,
                    timer: &FixedTimer,
                },
            )
            .unwrap();
        assert!(
            matches!(
                report.stop,
                CpuExit::SupervisorCall {
                    immediate: 0x66,
                    ..
                }
            ),
            "{:?}",
            report.stop
        );
        let mut output = [0u8; 768];
        memory
            .read_into(GuestVirtualAddress::new(0x2000), &mut output)
            .unwrap()
            .into_ready()
            .unwrap();
        assert_eq!(output, expected, "seed {seed}");
    }
}

#[test]
fn invalid_data_reads_return_zero_and_continue_with_native_register_optimization() {
    let (_memory, mut jit) = fixture(&[0xd2800020, 0xf9400022, 0xd2800040, 0xd4000001]);
    let mut state: ThreadState = pattern(5).into();
    state.write_x(x(1), 0x3000);
    state.write_x(x(2), 0xdead);
    let mut expected = state.clone();
    expected.write_x(x(0), 2);
    expected.write_x(x(2), 0);
    expected.set_pc(CODE.get() + 12);
    let report = run(&mut jit, &mut state);
    assert!(matches!(report.stop, CpuExit::SupervisorCall { .. }));
    assert!(report.context.is_none());
    assert_eq!(state, expected);
}

fn fallback_site_words(sites: usize, reads: usize) -> Vec<u32> {
    (0..2 * sites)
        .flat_map(|_| {
            (0..reads)
                .flat_map(|_| [0xf9400022, 0x8b020000]) // LDR X2,[X1]; ADD X0,X0,X2
                .chain([0xd4000001]) // SVC #0
        })
        .collect()
}

fn run_fallback_site(
    jit: &mut JitThread,
    worker: &mut nixe_cpu_direct_memory::NativeWorker,
    state: &mut ThreadState,
    pc: u64,
    address: u64,
    expected: u64,
) {
    state.set_pc(pc);
    state.write_x(x(0), 0);
    state.write_x(x(1), address);
    let report = jit
        .run_slice(
            worker,
            JitRunRequest {
                state,
                timer: &FixedTimer,
            },
        )
        .unwrap();
    assert!(matches!(
        report.stop,
        CpuExit::SupervisorCall { immediate: 0, .. }
    ));
    assert_eq!(state.read_x(x(0)), expected);
}

#[test]
fn fallback_sites_keep_pc_and_fpcr_identity_when_addresses_become_readable() {
    let sites = 32;
    let reads = 3;
    let stride = (reads * 2 + 1) as u64 * 4;
    let words = fallback_site_words(sites, reads);
    let (memory, mut jit) = fixture(&words);
    let data = CODE.get() + (words.len() as u64 * 4).div_ceil(PAGE_SIZE) * PAGE_SIZE;
    memory
        .write_value(GuestVirtualAddress::new(data), 8, 7)
        .unwrap()
        .into_ready()
        .unwrap();
    let mut state: ThreadState = a64_state().into();
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    for fpcr in [0, 1 << 22] {
        state.set_fpcr(fpcr);
        for site in 0..sites {
            run_fallback_site(
                &mut jit,
                &mut worker,
                &mut state,
                CODE.get() + site as u64 * stride,
                0x8000_0000,
                0,
            );
        }
    }
    for fpcr in [0, 1 << 22] {
        state.set_fpcr(fpcr);
        for site in 0..sites {
            run_fallback_site(
                &mut jit,
                &mut worker,
                &mut state,
                CODE.get() + site as u64 * stride,
                data,
                reads as u64 * 7,
            );
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
#[ignore = "controlled compilation with populated native fastmem fallback sites"]
fn populated_fastmem_compilation_throughput() {
    let sites = 256;
    let reads = 8;
    let stride = (reads * 2 + 1) as u64 * 4;
    let words = fallback_site_words(sites, reads);
    let (memory, mut jit) = fixture(&words);
    let data = CODE.get() + (words.len() as u64 * 4).div_ceil(PAGE_SIZE) * PAGE_SIZE;
    memory
        .write_value(GuestVirtualAddress::new(data), 8, 7)
        .unwrap()
        .into_ready()
        .unwrap();
    let mut state: ThreadState = a64_state().into();
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    for site in 0..sites {
        run_fallback_site(
            &mut jit,
            &mut worker,
            &mut state,
            CODE.get() + site as u64 * stride,
            0x8000_0000,
            0,
        );
    }
    let start = GuestVirtualAddress::new(CODE.get() + sites as u64 * stride);
    let bytes: Vec<u8> = words[sites * (reads * 2 + 1)..]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let ranges = memory
        .capture(start, bytes.len() as u64, MemoryPermissions::EXECUTE)
        .unwrap();
    let rounds = std::env::var("NIXE_FALLBACK_BENCH_ROUNDS")
        .ok()
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(32);
    let cpu_ns = || {
        std::fs::read_to_string("/proc/thread-self/schedstat")
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse::<u64>()
            .unwrap()
    };
    let before = cpu_ns();
    let wall = std::time::Instant::now();
    for _ in 0..rounds {
        // Publish the same bytes through the real code-write invalidation path.
        // Only the second half recompiles; the fault-site history stays populated.
        memory.machine().write_resolved(&ranges, &bytes).unwrap();
        for site in 0..sites {
            run_fallback_site(
                &mut jit,
                &mut worker,
                &mut state,
                start.get() + site as u64 * stride,
                data,
                reads as u64 * 7,
            );
        }
    }
    let wall_ns = wall.elapsed().as_nanos();
    let elapsed = cpu_ns() - before;
    let blocks = rounds * sites;
    let marked = sites * reads;
    println!(
        "FALLBACK_BENCH {{\"rounds\":{rounds},\"cold_blocks\":{blocks},\"populated_sites\":{marked},\"reads_per_block\":{reads},\"cpu_ns\":{elapsed},\"wall_ns\":{wall_ns}}}"
    );
}

#[test]
fn address_range_proofs_preserve_masking_overflow_and_invalid_accesses() {
    // STR X3,[X4]; address expression; LDR X2,[X1]; STR X5,[X1];
    // LDR X6,[X4]; SVC. Invalid accesses preserve the mapped sentinel.
    let cases: &[(&[u32], u64, bool)] = &[
        (&[0xd2840001], 0, true),                             // MOV X1,#0x2000
        (&[0x92403c01, 0x91002021], u64::MAX - 0xe007, true), // AND; ADD #8
        (&[0x2a0003e1], 0xffff_ffff_0000_2000, true),         // MOV W1,W0
        (&[0x2a0003e1, 0xd1002021], 4, false),                // zero extension; SUB #8
        (&[0xd2840001, 0xf2c00021], 0, false),                // constant outside Bits32
        (&[0xaa0003e1], 0x2000, true),                        // runtime MOV X1,X0
        (&[0xaa0003e1], 0x1_0000_2000, false),
    ];
    for &(expression, input, mapped) in cases {
        let mut code = vec![0xf9000083];
        code.extend_from_slice(expression);
        code.extend_from_slice(&[0xf9400022, 0xf9000025, 0xf9400086, 0xd4000841]);
        let (_memory, mut jit) = fixture(&code);
        // Reuse the compiled block with different runtime inputs too. A value
        // loaded from an X register is not a compilation-time address proof.
        for input in [input, input ^ (1 << 48), input] {
            let mut state: ThreadState = a64_state().into();
            state.write_x(x(0), input);
            state.write_x(x(3), 0x1122_3344_5566_7788);
            state.write_x(x(4), 0x2000);
            state.write_x(x(5), 0x8877_6655_4433_2211);
            let mapped = mapped && (expression != [0xaa0003e1] || input == 0x2000);
            assert!(matches!(
                run(&mut jit, &mut state).stop,
                CpuExit::SupervisorCall {
                    immediate: 0x42,
                    ..
                }
            ));
            assert_eq!(
                state.read_x(x(2)),
                if mapped { state.read_x(x(3)) } else { 0 }
            );
            assert_eq!(
                state.read_x(x(6)),
                state.read_x(x(if mapped { 5 } else { 3 }))
            );
        }
    }
}

#[test]
fn data_observation_keeps_native_execution_running_until_preemption() {
    use std::sync::mpsc;
    use std::time::{Duration, Instant};
    // ADD X0,#1; STR X0,[X1]; B back to ADD. The guest publishes readiness.
    let (memory, process) = process_fixture(&[0x91000400, 0xf9000020, 0x17fffffe]);
    let mut state: ThreadState = a64_state().into();
    state.write_x(x(1), 0x2008);
    let ranges = memory
        .capture(GuestVirtualAddress::new(0x2000), 4, MemoryPermissions::READ)
        .unwrap();
    let (ready, native_ready) = mpsc::channel();
    let (send, completed) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let mut jit = process.create_thread().unwrap();
        ready.send(jit.control()).unwrap();
        let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
        let report = jit.run_slice(
            &mut worker,
            JitRunRequest {
                state: &mut state,
                timer: &FixedTimer,
            },
        );
        state.leave_core();
        drop(jit);
        send.send((report, state)).unwrap();
        worker.finish().unwrap();
    });
    let control = native_ready.recv_timeout(Duration::from_secs(5)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while memory
        .read_value(GuestVirtualAddress::new(0x2008), 8)
        .unwrap()
        .into_ready()
        .unwrap()
        == 0
        && Instant::now() < deadline
    {
        std::thread::yield_now();
    }
    let running = memory
        .read_value(GuestVirtualAddress::new(0x2008), 8)
        .unwrap()
        .into_ready()
        .unwrap()
        != 0;
    let observation = nixe_memory::Observation::default();
    let captured = memory.machine().access_captured(
        &ranges,
        nixe_memory::AccessOperation::ReadObserved(observation.clone()),
    );
    let premature = completed.recv_timeout(Duration::from_millis(50));
    control.request_preemption();
    let (returned_for_memory, (report, state)) = match premature {
        Ok(result) => (true, result),
        Err(mpsc::RecvTimeoutError::Timeout) => (
            false,
            completed.recv_timeout(Duration::from_secs(5)).unwrap(),
        ),
        Err(error) => panic!("native worker disconnected: {error}"),
    };
    worker.join().unwrap();
    assert!(running);
    assert!(matches!(
        captured,
        Ok(nixe_memory::Access::Ready(nixe_memory::AccessResult::Read(
            _
        )))
    ));
    assert!(
        !returned_for_memory,
        "memory quiescence is not a scheduler exit"
    );
    assert!(!observation.is_invalid());
    assert_eq!(report.unwrap().stop, CpuExit::Safepoint);
    assert!(state.read_x(x(0)) > 0);
}

#[test]
fn scheduler_interrupt_exits_direct_and_indirect_native_loops() {
    for branch in [0x17fffffe, 0xd61f0040] {
        // B -8 / BR X2.
        let (memory, process) = process_fixture(&[0x91000400, 0xf9000020, branch]);
        let (ready, native_ready) = std::sync::mpsc::channel();
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let mut jit = process.create_thread().unwrap();
            ready.send(jit.control()).unwrap();
            let mut state: ThreadState = pattern(5).into();
            state.write_x(x(1), 0x2000);
            state.write_x(x(2), CODE.get());
            let expected = state.clone();
            let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
            let report = jit
                .run_slice(
                    &mut worker,
                    JitRunRequest {
                        state: &mut state,
                        timer: &FixedTimer,
                    },
                )
                .unwrap();
            state.leave_core();
            drop(jit);
            tx.send((report, state, expected)).unwrap();
            worker.finish().unwrap();
        });
        let control = native_ready
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while memory
            .read_value(GuestVirtualAddress::new(0x2000), 8)
            .unwrap()
            .into_ready()
            .unwrap()
            == 0
            && std::time::Instant::now() < deadline
        {
            std::thread::yield_now();
        }
        let running = memory
            .read_value(GuestVirtualAddress::new(0x2000), 8)
            .unwrap()
            .into_ready()
            .unwrap()
            != 0;
        control.request_preemption();
        let (report, state, mut expected) =
            rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        worker.join().unwrap();
        assert!(running);
        assert_eq!(report.stop, CpuExit::Safepoint);
        expected.write_x(x(0), state.read_x(x(0)));
        expected.set_pc(state.pc());
        assert_eq!(state, expected);
    }
}

#[test]
fn complete_context_survives_alternating_guests_and_migrating_between_cores() {
    let (memory, initial) = fixture(&[0xd4000001]);
    drop(initial);
    // One process/monitor, three host-owned native cores; none is a guest thread.
    let process = JitProcess::new(
        ProcessCpuContext::for_platform(TargetPlatform::Switch1, memory.identity()),
        memory,
        19_200_000,
    )
    .unwrap();
    let mut first = process.create_thread().unwrap();
    let mut second = process.create_thread().unwrap();
    let mut third = process.create_thread().unwrap();
    let mut states = [
        ThreadState::from(pattern(5)),
        ThreadState::from(pattern(10)),
    ];
    let expected = states.clone();
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    for turn in 0..24 {
        let guest = turn % 2;
        states[1 - guest].materialize();
        let core = match (turn / 2) % 3 {
            0 => &mut first,
            1 => &mut second,
            _ => &mut third,
        };
        let report = core
            .run_slice(
                &mut worker,
                JitRunRequest {
                    state: &mut states[guest],
                    timer: &FixedTimer,
                },
            )
            .unwrap();
        assert!(matches!(report.stop, CpuExit::SupervisorCall { .. }));
        assert_eq!(states[guest], expected[guest]);
        assert!(report.context.is_none());
    }
}

#[test]
fn failed_native_invocation_has_no_fabricated_context_and_cannot_resume() {
    let (_memory, mut jit) = fixture(&[0xd4000001]);
    let mut state: ThreadState = a64_state().into();
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    jit.run_slice(
        &mut worker,
        JitRunRequest {
            state: &mut state,
            timer: &FixedTimer,
        },
    )
    .unwrap();
    // Registering a second native fault context on this TID is a real error.
    let failure = jit
        .run_slice(
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            JitRunRequest {
                state: &mut state,
                timer: &FixedTimer,
            },
        )
        .unwrap_err();
    assert!(failure.message.contains("already registered"));
    assert!(failure.context.is_none());
    assert!(failure.to_string().contains("registers=[unavailable]"));
    let failure = jit
        .run_slice(
            &mut worker,
            JitRunRequest {
                state: &mut state,
                timer: &FixedTimer,
            },
        )
        .unwrap_err();
    assert!(failure.message.contains("unavailable"));
    assert!(!state.is_available());
    assert!(failure.context.is_none());
}

#[test]
fn ordinary_svc_keeps_complete_context_native_and_snapshots_are_owned() {
    let (_memory, mut jit) = fixture(&[0x91000400, 0xd4000001]); // ADD X0,#1; SVC
    let mut state: ThreadState = pattern(5).into();
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();

    let mut retained = None;
    for turn in 0..24 {
        state.set_pc(CODE.get());
        state.write_x(x(0), turn);
        state.set_tpidr_el0(0xabc000 + turn);
        let report = jit
            .run_slice(
                &mut worker,
                JitRunRequest {
                    state: &mut state,
                    timer: &FixedTimer,
                },
            )
            .unwrap();
        assert!(matches!(report.stop, CpuExit::SupervisorCall { .. }));
        assert!(report.context.is_none());
        assert!(state.is_resident());
        assert_eq!(state.read_x(x(0)), turn + 1);
        assert_eq!(state.tpidr_el0(), 0xabc000 + turn);
        if turn == 0 {
            retained = Some(state.snapshot());
        }
    }
    let snapshot = retained.unwrap();
    assert_eq!(snapshot.read_x(x(0)), 1);
    assert_eq!(snapshot.tpidr_el0(), 0xabc000);
    for index in 0..32 {
        assert_eq!(state.vector(index), snapshot.vector(index));
    }
    state.materialize();
    assert!(!state.is_resident());
    assert_eq!(state.read_x(x(0)), 24);
}

#[test]
fn replacing_an_unexported_owner_is_rejected_without_corrupting_either_guest() {
    let (_memory, mut jit) = fixture(&[0xd4000001]);
    let mut first: ThreadState = pattern(5).into();
    let mut second: ThreadState = pattern(10).into();
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();

    let mut run = |state: &mut ThreadState| {
        jit.run_slice(
            &mut worker,
            JitRunRequest {
                state,
                timer: &FixedTimer,
            },
        )
    };
    run(&mut first).unwrap();
    let expected_first = first.snapshot();
    let expected_second = second.snapshot();
    let failure = run(&mut second).unwrap_err();
    assert!(failure.message.contains("another guest context"));
    assert_eq!(first.snapshot(), expected_first);
    assert_eq!(second.snapshot(), expected_second);
    first.leave_core();
    run(&mut second).unwrap();
    assert_eq!(second.snapshot(), expected_second);
}

#[test]
fn stopped_native_owner_retains_core_monitor_and_backing_until_materialized() {
    let (memory, mut jit) = fixture(&[0xd4000001]);
    let weak_memory = Arc::downgrade(&memory);
    let mut state: ThreadState = pattern(5).into();
    let expected = state.snapshot();
    jit.run_slice(
        &mut nixe_cpu_direct_memory::NativeWorker::default(),
        JitRunRequest {
            state: &mut state,
            timer: &FixedTimer,
        },
    )
    .unwrap();
    drop(jit);
    drop(memory);
    assert!(weak_memory.upgrade().is_some());
    assert_eq!(state.snapshot(), expected);
    state.leave_core();
    assert!(weak_memory.upgrade().is_none());
    assert_eq!(state.snapshot(), expected);
}
