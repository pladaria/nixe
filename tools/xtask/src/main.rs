//! Repository development tasks exposed through Cargo aliases.

use std::{
    env,
    ffi::{OsStr, OsString},
    path::PathBuf,
    process::{Command, ExitCode},
};

const DEFAULT_FUZZ_RUNS: &str = "10000";
const FUZZ_TARGETS: [&str; 4] = ["decoder", "translation", "ir_verifier", "diagnostics"];

fn main() -> ExitCode {
    let mut arguments = env::args_os().skip(1);
    match arguments.next().as_deref() {
        Some(command) if command == "fuzz" => run_fuzz(arguments.collect()),
        Some(command) => fail(format!(
            "unknown xtask command: {}",
            command.to_string_lossy()
        )),
        None => fail("missing xtask command"),
    }
}

fn run_fuzz(arguments: Vec<OsString>) -> ExitCode {
    if !command_succeeds("rustup", ["run", "nightly", "rustc", "--version"]) {
        return fail("the nightly toolchain is missing; run: rustup toolchain install nightly");
    }
    if !command_succeeds("cargo-fuzz", ["--help"]) {
        return fail("cargo-fuzz is missing; run: cargo install cargo-fuzz");
    }

    if !arguments.is_empty() {
        return run_cargo_fuzz(arguments);
    }

    let runs = env::var("SWIITX_FUZZ_RUNS").unwrap_or_else(|_| DEFAULT_FUZZ_RUNS.to_owned());
    if runs.is_empty() || !runs.bytes().all(|byte| byte.is_ascii_digit()) || runs == "0" {
        return fail("SWIITX_FUZZ_RUNS must be a positive integer");
    }

    for target in FUZZ_TARGETS {
        println!("==> fuzzing {target} for {runs} runs");
        let arguments = vec![
            OsString::from("run"),
            OsString::from(target),
            OsString::from("--"),
            OsString::from(format!("-runs={runs}")),
        ];
        let result = run_cargo_fuzz(arguments);
        if result != ExitCode::SUCCESS {
            return result;
        }
    }
    ExitCode::SUCCESS
}

fn run_cargo_fuzz(arguments: Vec<OsString>) -> ExitCode {
    let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let Some(nightly_cargo) = nightly_tool("cargo") else {
        return fail("could not locate Cargo in the nightly toolchain");
    };
    let Some(nightly_rustc) = nightly_tool("rustc") else {
        return fail("could not locate rustc in the nightly toolchain");
    };
    let Some(nightly_bin) = nightly_cargo.parent() else {
        return fail("nightly Cargo has no parent directory");
    };
    let mut search_paths = vec![nightly_bin.to_path_buf()];
    if let Some(current_path) = env::var_os("PATH") {
        search_paths.extend(env::split_paths(&current_path));
    }
    let Ok(search_path) = env::join_paths(search_paths) else {
        return fail("could not construct the nightly tool search path");
    };
    match Command::new("cargo-fuzz")
        .args(arguments)
        .env("CARGO", nightly_cargo)
        .env("RUSTC", nightly_rustc)
        .env("PATH", search_path)
        .env("RUSTUP_TOOLCHAIN", "nightly")
        .current_dir(repository_root)
        .status()
    {
        Ok(status) => ExitCode::from(status.code().unwrap_or(1) as u8),
        Err(error) => fail(format!("failed to start cargo-fuzz: {error}")),
    }
}

fn nightly_tool(tool: &str) -> Option<PathBuf> {
    let output = Command::new("rustup")
        .args(["which", "--toolchain", "nightly", tool])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8(output.stdout).ok()?;
    Some(PathBuf::from(path.trim()))
}

fn command_succeeds<I, S>(program: &str, arguments: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    Command::new(program)
        .args(arguments)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn fail(message: impl std::fmt::Display) -> ExitCode {
    eprintln!("error: {message}");
    ExitCode::FAILURE
}
