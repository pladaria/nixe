use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use nixe_cpu::{
    interpreter::{InterpreterOutcome, execute_one},
    location::{ExecutionState, InstructionEncoding},
    profile::GuestCpuProfile,
    state::{
        A32State, A64State, ThreadCpuState,
        a32::{A32GeneralRegister, Cpsr},
        a64::{A64GeneralRegister, A64Register},
    },
};

const RUNNER_SOURCE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/oracle/arm_oracle_runner.c"
);

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
