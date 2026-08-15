use std::{
    collections::BTreeMap,
    env, fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Mutex, MutexGuard},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use nixe_cpu::{
    decode::{DecodeResult, a64::A64Instruction},
    location::{ExecutionState, InstructionEncoding, LocationDescriptor},
    profile::GuestCpuProfile,
    state::{
        A32State, A64State, ThreadCpuState,
        a32::{A32GeneralRegister, Cpsr},
        a64::{A64GeneralRegister, A64Register, Nzcv},
    },
};
use nixe_cpu_engine_interpreter::{InterpreterOutcome, execute_one, has_semantics};

const RUNNER_SOURCE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/oracle/arm_oracle_runner.c"
);
static QEMU_GDB_TEST_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug)]
struct OracleMetadata {
    emulator: String,
    version: String,
    profile: &'static str,
    execution_state: ExecutionState,
}

#[derive(Clone, Copy)]
struct OracleConfiguration {
    state: ExecutionState,
    compiler_default: &'static str,
    compiler_environment: &'static str,
    emulator_default: &'static str,
    emulator_environment: &'static str,
}

#[test]
#[ignore = "requires the optional QEMU user-mode and Arm cross-toolchain dependencies"]
fn qemu_user_mode_matches_adds_for_a64_a32_and_t32() {
    for configuration in configurations() {
        run_configuration(configuration);
    }
}

#[test]
#[ignore = "requires the optional QEMU user-mode and AArch64 cross-toolchain dependencies"]
fn qemu_a64_single_step_oracle_preserves_bfm_destination_bits() {
    let _serial = qemu_gdb_test_guard();
    let configuration = configurations()[0];
    let compiler = configured_tool(
        configuration.compiler_environment,
        configuration.compiler_default,
    );
    let emulator = configured_tool(
        configuration.emulator_environment,
        configuration.emulator_default,
    );
    let temporary = TestDirectory::new(ExecutionState::A64);
    let runner = temporary.path().join("a64-state-oracle-runner");
    compile_runner(&compiler, ExecutionState::A64, &runner);
    let slot = symbol_address(&runner, "nixe_oracle_slot");

    let mut oracle = QemuA64Oracle::start(&emulator, &runner);
    oracle.write_instruction(slot, 0x331b_0c20); // BFI W0,W1,#5,#4
    oracle.write_register(0, 0xa5a5_a5a5);
    oracle.write_register(1, 0xf);
    oracle.write_register(A64_PC_REGISTER, slot);
    oracle.step("BFI W0,W1,#5,#4", 0x331b_0c20);

    assert_eq!(oracle.read_register(0), 0xa5a5_a5e5);
    assert_eq!(oracle.read_register(A64_PC_REGISTER), slot + 4);
}

#[test]
#[ignore = "requires the optional QEMU user-mode and AArch64 cross-toolchain dependencies"]
fn qemu_a64_matches_control_and_stateless_system_semantics() {
    let _serial = qemu_gdb_test_guard();
    let configuration = configurations()[0];
    let compiler = configured_tool(
        configuration.compiler_environment,
        configuration.compiler_default,
    );
    let emulator = configured_tool(
        configuration.emulator_environment,
        configuration.emulator_default,
    );
    let temporary = TestDirectory::new(ExecutionState::A64);
    let runner = temporary.path().join("a64-control-oracle-runner");
    compile_runner(&compiler, ExecutionState::A64, &runner);
    let slot = symbol_address(&runner, "nixe_oracle_slot");
    let mut oracle = QemuA64Oracle::start(&emulator, &runner);
    let cases = [
        (0xd503_201f, "NOP"),
        (0x1400_0002, "B +8"),
        (0x9400_0002, "BL +8"),
        (0xd61f_0000, "BR X0"),
        (0xd63f_0000, "BLR X0"),
        (0xd65f_03c0, "RET X30"),
        (0x5400_0041, "B.NE +8"),
        (0xb500_0040, "CBNZ X0,+8"),
        (0x3600_0040, "TBZ X0,#0,+8"),
        (0xd503_3f9f, "DSB SY"),
    ];
    for (index, (encoding, _)) in cases.iter().enumerate() {
        oracle.write_instruction(slot + index as u64 * 4, *encoding);
    }
    let profile = GuestCpuProfile::switch_1();
    for (index, (encoding, name)) in cases.into_iter().enumerate() {
        let pc = slot + index as u64 * 4;
        let target = pc + 8;
        let mut expected = initial_oracle_a64_state(pc);
        let ThreadCpuState::A64(state) = &mut expected else {
            unreachable!()
        };
        state.write_x(x(0), target);
        state.write_x(x(30), target);
        oracle.write_register(0, target);
        oracle.write_register(30, target);
        oracle.write_raw_register(33, &state.nzcv().bits().to_le_bytes());
        oracle.write_register(A64_PC_REGISTER, pc);

        let outcome = execute_one(&profile, &mut expected, encoding.into())
            .unwrap_or_else(|error| panic!("{name} ({encoding:#010x}) failed in Nixe: {error}"));
        assert!(matches!(outcome, InterpreterOutcome::Resume(_)));
        oracle.step(name, encoding);

        let ThreadCpuState::A64(expected) = expected else {
            unreachable!()
        };
        assert_eq!(
            oracle.read_register(30),
            expected.read_x(x(30)),
            "{name} X30"
        );
        assert_eq!(
            oracle.read_register(A64_PC_REGISTER),
            expected.pc(),
            "{name} PC"
        );
        let observed_nzcv = u32::from_le_bytes(
            oracle
                .read_raw_register(33)
                .try_into()
                .expect("32-bit CPSR"),
        ) & 0xf000_0000;
        assert_eq!(observed_nzcv, expected.nzcv().bits(), "{name} NZCV");
    }
}

#[test]
#[ignore = "requires the optional QEMU user-mode and AArch64 cross-toolchain dependencies"]
fn qemu_a64_matches_every_register_semantic_family() {
    let _serial = qemu_gdb_test_guard();
    let filter = DifferentialFilter::from_environment();
    let configuration = configurations()[0];
    let compiler = configured_tool(
        configuration.compiler_environment,
        configuration.compiler_default,
    );
    let emulator = configured_tool(
        configuration.emulator_environment,
        configuration.emulator_default,
    );
    let temporary = TestDirectory::new(ExecutionState::A64);
    let runner = temporary.path().join("a64-register-oracle-runner");
    compile_runner(&compiler, ExecutionState::A64, &runner);
    let slot = symbol_address(&runner, "nixe_oracle_slot");
    let mut oracle = QemuA64Oracle::start(&emulator, &runner);
    let objdump = configured_tool("NIXE_AARCH64_OBJDUMP", "aarch64-linux-gnu-objdump");

    let profile = GuestCpuProfile::switch_1();
    let mut cases = Vec::new();
    let mut covered = BTreeMap::new();
    for pattern in nixe_cpu::decode::a64::patterns() {
        if !register_semantic_id(pattern.coverage_id.get())
            || !filter.matches(pattern.coverage_id.get(), pattern.name)
        {
            continue;
        }
        let Some(encoding) = allocated_register_encoding(
            &profile,
            pattern.coverage_id.get(),
            pattern.mask,
            pattern.value,
            &objdump,
            temporary.path(),
        ) else {
            continue;
        };
        cases.push((encoding, pattern.name));
        covered.insert(pattern.coverage_id.get(), pattern.name);
    }

    // The two historical fallback rows contain several independently decoded
    // Advanced SIMD and floating-point operations. Keep real captured aliases
    // in the oracle corpus so broad fallback coverage cannot hide a gap.
    for (encoding, name, family, coverage_id) in [
        (
            0x4e22_1c20,
            "ORR V0.16B,V1.16B,V2.16B",
            "advanced-simd-fallback",
            0x38,
        ),
        (
            0x4e22_8420,
            "ADD V0.16B,V1.16B,V2.16B",
            "advanced-simd-fallback",
            0x38,
        ),
        (
            0x4e32_be31,
            "ADDP V17.4S,V17.4S,V18.4S",
            "advanced-simd-fallback",
            0x38,
        ),
        (
            0x6e31_a631,
            "UMAXP V17.4S,V17.4S,V17.4S",
            "advanced-simd-fallback",
            0x38,
        ),
        (
            0x6e21_3ca3,
            "CMHS V3.16B,V5.16B,V1.16B",
            "advanced-simd-fallback",
            0x38,
        ),
        (
            0x4e20_9823,
            "CMEQ V3.16B,V1.16B,#0",
            "advanced-simd-fallback",
            0x38,
        ),
        (
            0x4e21_dbfc,
            "SCVTF V28.4S,V31.4S",
            "advanced-simd-fallback",
            0x38,
        ),
        (
            0x6e3e_ff9c,
            "FDIV V28.4S,V28.4S,V30.4S",
            "advanced-simd-fallback",
            0x38,
        ),
        (
            0x1e22_1820,
            "FDIV S0,S1,S2",
            "floating-point-fallback",
            0x39,
        ),
        (
            0x1e22_2820,
            "FADD S0,S1,S2",
            "floating-point-fallback",
            0x39,
        ),
        (
            0x1e22_3820,
            "FSUB S0,S1,S2",
            "floating-point-fallback",
            0x39,
        ),
        (
            0x1e22_0820,
            "FMUL S0,S1,S2",
            "floating-point-fallback",
            0x39,
        ),
    ] {
        if filter.matches(coverage_id, family) {
            cases.push((encoding, name));
        }
    }
    if filter.matches(0x91, "simd-scalar-shift-right-immediate") {
        cases.push((0x7f60_07fe, "USHR D30,D31,#32 (captured)"));
    }
    if filter.matches(0x93, "simd-count-bits") {
        cases.push((0x0e20_5bde, "CNT V30.8B,V30.8B (captured)"));
    }
    if filter.matches(0x94, "simd-add-across-vector") {
        cases.push((0x0e31_bbde, "ADDV B30,V30.8B (captured)"));
    }
    if filter.matches(0x95, "simd-signed-shift-left-register") {
        cases.extend([
            (0x0e22_4420, "SSHL V0.8B,V1.8B,V2.8B"),
            (0x0e62_4420, "SSHL V0.4H,V1.4H,V2.4H"),
            (0x0ebd_47fd, "SSHL V29.2S,V31.2S,V29.2S (captured)"),
            (0x4ee2_4420, "SSHL V0.2D,V1.2D,V2.2D"),
        ]);
    }
    if filter.matches(0x96, "simd-unsigned-shift-left-register") {
        cases.extend([
            (0x2e22_4420, "USHL V0.8B,V1.8B,V2.8B"),
            (0x2e62_4420, "USHL V0.4H,V1.4H,V2.4H"),
            (0x2ea2_4420, "USHL V0.2S,V1.2S,V2.2S"),
            (0x6ee2_4420, "USHL V0.2D,V1.2D,V2.2D"),
        ]);
    }
    if filter.matches(0x97, "fp-scalar-fused-multiply-add") {
        cases.extend([
            (0x1f02_0c20, "FMADD S0,S1,S2,S3"),
            (0x1f02_8c20, "FMSUB S0,S1,S2,S3"),
            (0x1f22_0c20, "FNMADD S0,S1,S2,S3"),
            (0x1f22_8c20, "FNMSUB S0,S1,S2,S3"),
            (0x1f40_7bbe, "FMADD D30,D29,D0,D30 (captured)"),
            (0x1f42_8c20, "FMSUB D0,D1,D2,D3"),
            (0x1f62_0c20, "FNMADD D0,D1,D2,D3"),
            (0x1f62_8c20, "FNMSUB D0,D1,D2,D3"),
        ]);
    }
    if filter.matches(0x98, "fp-scalar-square-root") {
        cases.extend([
            (0x1e21_c3de, "FSQRT S30,S30 (captured)"),
            (0x1e61_c020, "FSQRT D0,D1"),
        ]);
    }

    if filter.is_active() {
        assert!(
            !cases.is_empty(),
            "differential filter {} matched no implemented A64 register family",
            filter.description()
        );
        eprintln!(
            "A64 differential filter {} selected {} case(s)",
            filter.description(),
            cases.len()
        );
    } else {
        assert!(
            covered.len() >= 70,
            "unexpectedly small A64 oracle coverage"
        );
    }
    for (index, (encoding, _)) in cases.iter().enumerate() {
        oracle.write_instruction(slot + index as u64 * 4, *encoding);
    }
    for (index, (encoding, name)) in cases.into_iter().enumerate() {
        compare_a64_register_case(
            &profile,
            &mut oracle,
            slot + index as u64 * 4,
            encoding,
            name,
        );
    }
}

#[derive(Debug, Default)]
struct DifferentialFilter {
    family: Option<String>,
    coverage_id: Option<u32>,
}

impl DifferentialFilter {
    fn from_environment() -> Self {
        Self {
            family: env::var("NIXE_DIFF_FAMILY").ok(),
            coverage_id: env::var("NIXE_DIFF_COVERAGE_ID")
                .ok()
                .map(|value| parse_coverage_id(&value)),
        }
    }

    fn matches(&self, coverage_id: u32, family: &str) -> bool {
        self.coverage_id
            .is_none_or(|expected| expected == coverage_id)
            && self
                .family
                .as_deref()
                .is_none_or(|expected| expected.eq_ignore_ascii_case(family))
    }

    fn is_active(&self) -> bool {
        self.family.is_some() || self.coverage_id.is_some()
    }

    fn description(&self) -> String {
        match (&self.family, self.coverage_id) {
            (Some(family), Some(id)) => format!("family={family}, coverage={id:#010x}"),
            (Some(family), None) => format!("family={family}"),
            (None, Some(id)) => format!("coverage={id:#010x}"),
            (None, None) => "all families".into(),
        }
    }
}

fn parse_coverage_id(value: &str) -> u32 {
    let (digits, radix) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .map_or((value, 10), |digits| (digits, 16));
    u32::from_str_radix(digits, radix)
        .unwrap_or_else(|error| panic!("invalid NIXE_DIFF_COVERAGE_ID {value:?}: {error}"))
}

#[test]
fn differential_filter_matches_names_and_decimal_or_hex_coverage_ids() {
    let filter = DifferentialFilter {
        family: Some("SIMD-DUPLICATE-ELEMENT".into()),
        coverage_id: Some(parse_coverage_id("0x8c")),
    };
    assert!(filter.matches(0x8c, "simd-duplicate-element"));
    assert!(!filter.matches(0x8d, "simd-duplicate-element"));
    assert!(!filter.matches(0x8c, "simd-duplicate-general"));
    assert_eq!(parse_coverage_id("140"), 0x8c);
}

// Keep this manifest explicit: adding a register-only semantic requires adding
// it here, which makes the optional QEMU job review its differential coverage.
fn register_semantic_id(id: u32) -> bool {
    matches!(
        id,
        0x0000_0003
            | 0x0000_0010..=0x0000_001d
            | 0x0000_0020..=0x0000_0021
            | 0x0000_0030..=0x0000_0032
            | 0x0000_0035..=0x0000_003f
            | 0x0000_0098
            | 0x0000_0048
            | 0x0000_004a..=0x0000_004b
            | 0x0000_004e..=0x0000_0061
            | 0x0000_0064..=0x0000_0097
    )
}

fn allocated_register_encoding(
    profile: &GuestCpuProfile,
    coverage_id: u32,
    pattern_mask: u32,
    pattern_value: u32,
    objdump: &str,
    temporary: &Path,
) -> Option<u32> {
    let mut random = 0x6a09_e667_f3bc_c909_u64 ^ u64::from(coverage_id);
    let mut candidates = Vec::new();
    for attempt in 0..100_001_u32 {
        let variable = if attempt == 0 {
            0
        } else {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            (random as u32).wrapping_add((attempt - 1).wrapping_mul(0x9e37_79b9))
        };
        let encoding = pattern_value | (variable & !pattern_mask);
        let location = LocationDescriptor::new(
            nixe_cpu::address::GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            profile.id(),
        );
        let DecodeResult::Decoded(decoded) =
            nixe_cpu::decode::decode(profile, location, encoding.into())
        else {
            continue;
        };
        if decoded.instruction.coverage_id().get() != coverage_id || !has_semantics(&decoded) {
            continue;
        }
        let (A64Instruction::Integer(_) | A64Instruction::FpSimd(_)) =
            nixe_cpu::decode::a64::normalize(&decoded.instruction, encoding.into())
        else {
            continue;
        };
        if !candidates.contains(&encoding) {
            candidates.push(encoding);
        }
        if candidates.len() == 256 {
            break;
        }
    }
    for encoding in allocated_a64_encodings(objdump, temporary, coverage_id, &candidates) {
        let mut state = initial_oracle_a64_state(0x1000);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_one(profile, &mut state, encoding.into())
        }))
        .unwrap_or_else(|_| {
            panic!(
                "A64 semantic {coverage_id:#010x} panicked for allocated encoding {encoding:#010x}"
            )
        });
        if matches!(outcome, Ok(InterpreterOutcome::Resume(_))) {
            return Some(encoding);
        }
    }
    None
}

fn allocated_a64_encodings(
    objdump: &str,
    temporary: &Path,
    coverage_id: u32,
    candidates: &[u32],
) -> Vec<u32> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let binary = temporary.join(format!("a64-{coverage_id:08x}.bin"));
    let bytes: Vec<_> = candidates
        .iter()
        .flat_map(|encoding| encoding.to_le_bytes())
        .collect();
    fs::write(&binary, bytes).expect("write A64 allocation corpus");
    let output = Command::new(objdump)
        .args(["-D", "-b", "binary", "-m", "aarch64"])
        .arg(&binary)
        .output()
        .unwrap_or_else(|error| panic!("failed to launch {objdump}: {error}"));
    assert!(
        output.status.success(),
        "{objdump} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let disassembly = String::from_utf8(output.stdout).expect("objdump output is UTF-8");
    let mut allocated = Vec::new();
    for line in disassembly.lines() {
        let Some((address, instruction)) = line.trim().split_once(':') else {
            continue;
        };
        let Ok(address) = usize::from_str_radix(address, 16) else {
            continue;
        };
        if address % 4 == 0
            && !instruction.contains("undefined")
            && let Some(encoding) = candidates.get(address / 4)
        {
            allocated.push(*encoding);
        }
    }
    allocated
}

fn initial_oracle_a64_state(pc: u64) -> ThreadCpuState {
    let mut state = A64State::default();
    for index in 0..31 {
        state.write_x(
            x(index),
            0x1020_3040_5060_7080_u64.rotate_left(u32::from(index)),
        );
    }
    state.write_x(A64Register::StackPointer, 0x1000);
    state.set_pc(pc);
    state.set_nzcv(Nzcv::from_bits(0xa000_0000));
    for index in 0..32 {
        let lane = 0x3f80_0000_u128 | (u128::from(index) << 32);
        assert!(state.set_vector(index, lane | (lane << 64)));
    }
    ThreadCpuState::A64(Box::new(state))
}

fn compare_a64_register_case(
    profile: &GuestCpuProfile,
    oracle: &mut QemuA64Oracle,
    slot: u64,
    encoding: u32,
    name: &str,
) {
    let mut expected = initial_oracle_a64_state(slot);
    let ThreadCpuState::A64(initial) = &expected else {
        unreachable!()
    };
    let location = LocationDescriptor::new(
        nixe_cpu::address::GuestVirtualAddress::new(slot),
        ExecutionState::A64,
        profile.id(),
    );
    let decoded = match nixe_cpu::decode::decode(profile, location, encoding.into()) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => decoded,
        _ => panic!("{name} ({encoding:#010x}) no longer decodes"),
    };
    let (registers, vectors, destination) =
        match nixe_cpu::decode::a64::normalize(&decoded.instruction, encoding.into()) {
            A64Instruction::Integer(instruction) => {
                let fields = instruction.operands();
                (
                    vec![fields.rd, fields.rn, fields.rm, fields.ra],
                    Vec::new(),
                    fields.rd,
                )
            }
            A64Instruction::FpSimd(instruction) => {
                let fields = instruction.operands();
                let mut vectors = vec![fields.rd, fields.rn, fields.rm];
                if matches!(
                    instruction,
                    nixe_cpu::decode::a64::fp_simd::Instruction::ScalarFloatFusedMultiplyAdd(_)
                ) {
                    vectors.push(fields.ra);
                }
                (vec![fields.rd, fields.rn, fields.rm], vectors, fields.rd)
            }
            A64Instruction::RecognizedFallback { .. } => {
                let rd = (encoding & 0x1f) as u8;
                let rn = ((encoding >> 5) & 0x1f) as u8;
                let rm = ((encoding >> 16) & 0x1f) as u8;
                (vec![rd, rn, rm], vec![rd, rn, rm], rd)
            }
            _ => panic!("{name} ({encoding:#010x}) is not a register semantic"),
        };
    for index in registers.iter().copied().filter(|index| *index < 31) {
        oracle.write_register(u32::from(index), initial.read_x(x(index)));
    }
    oracle.write_register(31, initial.read_x(A64Register::StackPointer));
    oracle.write_raw_register(33, &initial.nzcv().bits().to_le_bytes());
    oracle.write_raw_register(A64_FPSR_REGISTER, &initial.fpsr().to_le_bytes());
    oracle.write_raw_register(A64_FPCR_REGISTER, &initial.fpcr().to_le_bytes());
    for index in vectors.iter().copied() {
        let mut value = vec![0_u8; 256];
        value[..16].copy_from_slice(&initial.vector(index).unwrap().to_le_bytes());
        oracle.write_raw_register(34 + u32::from(index), &value);
    }
    oracle.write_register(A64_PC_REGISTER, slot);

    let outcome = execute_one(profile, &mut expected, encoding.into())
        .unwrap_or_else(|error| panic!("{name} ({encoding:#010x}) failed in Nixe: {error}"));
    assert!(
        matches!(outcome, InterpreterOutcome::Resume(_)),
        "{name} ({encoding:#010x}) did not resume in Nixe: {outcome:?}"
    );
    oracle.step(name, encoding);

    let ThreadCpuState::A64(expected) = expected else {
        unreachable!()
    };
    for index in registers.iter().copied().filter(|index| *index < 31) {
        assert_eq!(
            oracle.read_register(u32::from(index)),
            expected.read_x(x(index)),
            "{name} ({encoding:#010x}) X{index}"
        );
    }
    assert_eq!(
        oracle.read_register(A64_PC_REGISTER),
        expected.pc(),
        "{name} ({encoding:#010x}) PC"
    );
    let qemu_nzcv = u32::from_le_bytes(
        oracle
            .read_raw_register(33)
            .try_into()
            .expect("32-bit CPSR"),
    ) & 0xf000_0000;
    assert_eq!(
        qemu_nzcv,
        expected.nzcv().bits(),
        "{name} ({encoding:#010x}) NZCV"
    );
    assert_eq!(
        u32::from_le_bytes(
            oracle
                .read_raw_register(A64_FPSR_REGISTER)
                .try_into()
                .expect("32-bit FPSR"),
        ),
        expected.fpsr(),
        "{name} ({encoding:#010x}) FPSR"
    );
    assert_eq!(
        u32::from_le_bytes(
            oracle
                .read_raw_register(A64_FPCR_REGISTER)
                .try_into()
                .expect("32-bit FPCR"),
        ),
        expected.fpcr(),
        "{name} ({encoding:#010x}) FPCR"
    );
    for index in vectors
        .iter()
        .copied()
        .filter(|index| *index == destination)
    {
        let observed = oracle.read_raw_register(34 + u32::from(index));
        assert_eq!(
            u128::from_le_bytes(observed[..16].try_into().unwrap()),
            expected.vector(index).unwrap(),
            "{name} ({encoding:#010x}) V{index}"
        );
    }
}

const A64_PC_REGISTER: u32 = 32;
// QEMU's AArch64 SVE target description keeps FPSR/FPCR immediately after Z31.
const A64_FPSR_REGISTER: u32 = 66;
const A64_FPCR_REGISTER: u32 = 67;

fn qemu_gdb_test_guard() -> MutexGuard<'static, ()> {
    QEMU_GDB_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct QemuA64Oracle {
    child: Child,
    remote: GdbRemote,
}

impl QemuA64Oracle {
    fn start(emulator: &str, runner: &Path) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("reserve QEMU GDB port");
        let port = listener.local_addr().expect("reserved address").port();
        drop(listener);
        let mut child = Command::new(emulator)
            .arg("-g")
            .arg(port.to_string())
            .arg(runner)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|error| panic!("failed to launch {emulator}: {error}"));
        let mut connection = None;
        for _ in 0..200 {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(stream) => {
                    connection = Some(stream);
                    break;
                }
                Err(error) => {
                    if let Some(status) = child.try_wait().expect("query QEMU process status") {
                        let mut stderr = String::new();
                        if let Some(mut stream) = child.stderr.take() {
                            let _ = stream.read_to_string(&mut stderr);
                        }
                        panic!(
                            "{emulator} exited before its GDB stub accepted connections: \
                             status={status}, connect={error}, stderr={stderr}"
                        );
                    }
                    thread::sleep(std::time::Duration::from_millis(5));
                }
            }
        }
        let mut remote = GdbRemote::new(connection.expect("connect to QEMU GDB stub"));
        let supported = remote.command("qSupported:multiprocess+");
        assert!(
            !supported.starts_with('E'),
            "QEMU rejected qSupported: {supported}"
        );
        let stop = remote.command("?");
        assert!(
            stop.starts_with(['S', 'T']),
            "unexpected initial stop: {stop}"
        );
        Self { child, remote }
    }

    fn write_instruction(&mut self, address: u64, encoding: u32) {
        let response = self.remote.command(&format!(
            "M{address:x},4:{}",
            encode_hex(&encoding.to_le_bytes())
        ));
        assert_eq!(
            response, "OK",
            "QEMU rejected instruction patch at {address:#x} for {encoding:#010x}"
        );
    }

    fn write_register(&mut self, register: u32, value: u64) {
        self.write_raw_register(register, &value.to_le_bytes());
    }

    fn write_raw_register(&mut self, register: u32, value: &[u8]) {
        let response = self
            .remote
            .command(&format!("P{register:x}={}", encode_hex(value)));
        assert_eq!(response, "OK", "QEMU rejected register {register}");
    }

    fn read_register(&mut self, register: u32) -> u64 {
        let bytes = self.read_raw_register(register);
        assert_eq!(bytes.len(), 8, "unexpected A64 register width");
        u64::from_le_bytes(bytes.try_into().unwrap())
    }

    fn read_raw_register(&mut self, register: u32) -> Vec<u8> {
        decode_hex(&self.remote.command(&format!("p{register:x}")))
    }

    fn step(&mut self, name: &str, encoding: u32) {
        let stop = self.remote.command("s");
        assert!(
            stop.starts_with("S05") || stop.starts_with("T05"),
            "{name} ({encoding:#010x}) produced unexpected single-step stop: {stop}"
        );
    }
}

impl Drop for QemuA64Oracle {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct GdbRemote {
    stream: TcpStream,
}

impl GdbRemote {
    fn new(stream: TcpStream) -> Self {
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .expect("set GDB read timeout");
        Self { stream }
    }

    fn command(&mut self, command: &str) -> String {
        let checksum = command
            .as_bytes()
            .iter()
            .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
        write!(self.stream, "${command}#{checksum:02x}").expect("write GDB packet");
        self.stream.flush().expect("flush GDB packet");
        let mut byte = [0_u8; 1];
        loop {
            self.stream
                .read_exact(&mut byte)
                .expect("read GDB response");
            if byte[0] == b'$' {
                break;
            }
        }
        let mut payload = Vec::new();
        loop {
            self.stream.read_exact(&mut byte).expect("read GDB payload");
            if byte[0] == b'#' {
                break;
            }
            payload.push(byte[0]);
        }
        let mut received_checksum = [0_u8; 2];
        self.stream
            .read_exact(&mut received_checksum)
            .expect("read GDB checksum");
        self.stream.write_all(b"+").expect("acknowledge GDB packet");
        String::from_utf8(payload).expect("GDB response is ASCII")
    }
}

fn symbol_address(executable: &Path, symbol: &str) -> u64 {
    let output = Command::new("aarch64-linux-gnu-nm")
        .arg(executable)
        .output()
        .expect("launch AArch64 nm");
    assert!(output.status.success(), "AArch64 nm failed");
    let symbols = String::from_utf8(output.stdout).expect("nm output is UTF-8");
    let line = symbols
        .lines()
        .find(|line| line.ends_with(&format!(" {symbol}")))
        .unwrap_or_else(|| panic!("missing oracle symbol {symbol}"));
    u64::from_str_radix(line.split_whitespace().next().unwrap(), 16).expect("hex symbol address")
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn decode_hex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0, "hex response has complete bytes");
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&text[index..index + 2], 16).expect("hex response byte"))
        .collect()
}

fn run_configuration(configuration: OracleConfiguration) {
    let compiler = configured_tool(
        configuration.compiler_environment,
        configuration.compiler_default,
    );
    let emulator = configured_tool(
        configuration.emulator_environment,
        configuration.emulator_default,
    );
    let temporary = TestDirectory::new(configuration.state);
    let runner = temporary.path().join(match configuration.state {
        ExecutionState::A64 => "a64-oracle-runner",
        ExecutionState::A32 => "a32-oracle-runner",
        ExecutionState::T32 => "t32-oracle-runner",
    });
    compile_runner(&compiler, configuration.state, &runner);
    let version = tool_version(&emulator);
    let metadata = OracleMetadata {
        emulator: emulator.clone(),
        version,
        profile: "armv8-a",
        execution_state: configuration.state,
    };
    eprintln!(
        "oracle={} version={} profile={} state={}",
        metadata.emulator, metadata.version, metadata.profile, metadata.execution_state
    );

    for (lhs, rhs) in operands(configuration.state) {
        let expected = nixe_adds(configuration.state, lhs, rhs);
        let observed = qemu_adds(&emulator, &runner, configuration.state, lhs, rhs);
        assert_eq!(
            observed, expected,
            "QEMU mismatch for {} lhs={lhs:#x} rhs={rhs:#x}; metadata={metadata:?}",
            configuration.state
        );
    }
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(state: ExecutionState) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let path = env::temp_dir().join(format!(
            "nixe-qemu-differential-{}-{state}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create QEMU differential build directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn configurations() -> [OracleConfiguration; 3] {
    [
        OracleConfiguration {
            state: ExecutionState::A64,
            compiler_default: "aarch64-linux-gnu-gcc",
            compiler_environment: "NIXE_AARCH64_CC",
            emulator_default: "qemu-aarch64",
            emulator_environment: "NIXE_QEMU_AARCH64",
        },
        OracleConfiguration {
            state: ExecutionState::A32,
            compiler_default: "arm-linux-gnueabihf-gcc",
            compiler_environment: "NIXE_ARM_CC",
            emulator_default: "qemu-arm",
            emulator_environment: "NIXE_QEMU_ARM",
        },
        OracleConfiguration {
            state: ExecutionState::T32,
            compiler_default: "arm-linux-gnueabihf-gcc",
            compiler_environment: "NIXE_ARM_CC",
            emulator_default: "qemu-arm",
            emulator_environment: "NIXE_QEMU_ARM",
        },
    ]
}

fn configured_tool(environment: &str, default: &str) -> String {
    env::var(environment).unwrap_or_else(|_| default.to_string())
}

fn compile_runner(compiler: &str, state: ExecutionState, output: &Path) {
    let mut command = Command::new(compiler);
    command.args([
        "-std=c11",
        "-O2",
        "-static",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-march=armv8-a",
    ]);
    if state != ExecutionState::A64 {
        command.arg("-mfpu=neon-fp-armv8");
    }
    match state {
        ExecutionState::A32 => {
            command.arg("-marm");
        }
        ExecutionState::T32 => {
            command.arg("-mthumb");
        }
        ExecutionState::A64 => {}
    }
    let result = command
        .arg(RUNNER_SOURCE)
        .arg("-o")
        .arg(output)
        .output()
        .unwrap_or_else(|error| panic!("failed to launch {compiler}: {error}"));
    assert!(
        result.status.success(),
        "{compiler} failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn tool_version(tool: &str) -> String {
    let output = Command::new(tool)
        .arg("--version")
        .output()
        .unwrap_or_else(|error| panic!("failed to query {tool}: {error}"));
    assert!(output.status.success(), "{tool} --version failed");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or("unknown")
        .to_string()
}

fn qemu_adds(
    emulator: &str,
    runner: &Path,
    state: ExecutionState,
    lhs: u64,
    rhs: u64,
) -> (u64, u32) {
    let output = Command::new(emulator)
        .arg(runner)
        .arg(format!("{lhs:x}"))
        .arg(format!("{rhs:x}"))
        .output()
        .unwrap_or_else(|error| panic!("failed to launch {emulator}: {error}"));
    assert!(
        output.status.success(),
        "{emulator} oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("oracle output is UTF-8");
    let fields: BTreeMap<_, _> = stdout
        .split_whitespace()
        .map(|field| field.split_once('=').expect("oracle field uses key=value"))
        .collect();
    let expected_arch = match state {
        ExecutionState::A64 => "a64",
        ExecutionState::A32 => "a32",
        ExecutionState::T32 => "t32",
    };
    assert_eq!(fields.get("arch"), Some(&expected_arch));
    assert_eq!(fields.get("profile"), Some(&"armv8-a"));
    (
        u64::from_str_radix(fields["result"], 16).expect("hexadecimal result"),
        u32::from_str_radix(fields["flags"], 16).expect("hexadecimal flags"),
    )
}

fn nixe_adds(state: ExecutionState, lhs: u64, rhs: u64) -> (u64, u32) {
    let profile = GuestCpuProfile::switch_1();
    let (mut state, encoding) = match state {
        ExecutionState::A64 => {
            let mut cpu = A64State::default();
            cpu.write_x(x(1), lhs);
            cpu.write_x(x(2), rhs);
            (
                ThreadCpuState::A64(Box::new(cpu)),
                InstructionEncoding::from_u32(0xab02_0020), // ADDS X0,X1,X2
            )
        }
        ExecutionState::A32 => {
            let mut cpu = A32State::a32();
            cpu.write_r(r(1), lhs as u32);
            cpu.write_r(r(2), rhs as u32);
            (
                ThreadCpuState::A32(Box::new(cpu)),
                InstructionEncoding::from_u32(0xe091_0002), // ADDS R0,R1,R2
            )
        }
        ExecutionState::T32 => {
            let mut cpu = A32State::t32();
            cpu.write_r(r(1), lhs as u32);
            cpu.write_r(r(2), rhs as u32);
            (
                ThreadCpuState::A32(Box::new(cpu)),
                InstructionEncoding::from_u16(0x1888), // ADDS R0,R1,R2
            )
        }
    };
    let outcome = execute_one(&profile, &mut state, encoding).expect("Nixe implements ADDS");
    assert!(matches!(outcome, InterpreterOutcome::Resume(_)));
    match state {
        ThreadCpuState::A64(cpu) => (cpu.read_x(x(0)), cpu.nzcv().bits()),
        ThreadCpuState::A32(cpu) => (
            u64::from(cpu.read_r(r(0))),
            cpu.cpsr().bits() & (Cpsr::N | Cpsr::Z | Cpsr::C | Cpsr::V),
        ),
    }
}

fn operands(state: ExecutionState) -> Vec<(u64, u64)> {
    let mask = if state == ExecutionState::A64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let sign = if state == ExecutionState::A64 {
        1_u64 << 63
    } else {
        1_u64 << 31
    };
    let mut values = vec![
        (0, 0),
        (mask, 1),
        (sign - 1, 1),
        (sign, sign),
        (0x1234_5678 & mask, 0x7654_3210 & mask),
    ];
    let mut random = 0x7175_656d_755f_6469_u64;
    for _ in 0..24 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let lhs = random & mask;
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        values.push((lhs, random & mask));
    }
    values
}

fn x(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}

fn r(index: u8) -> A32GeneralRegister {
    A32GeneralRegister::new(index).unwrap()
}
