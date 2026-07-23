use std::fs;
use std::path::{Path, PathBuf};

use nixe_cpu::state::ThreadCpuState;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_horizon::{HorizonSvcDispatcher, HorizonSvcFault, HorizonSvcSupport};
use nixe_runtime::{
    ExceptionHandlingResult, ExecutionStop, Launcher, LauncherInput, ProcessBuilder,
    ProcessExecutionStatus, ProcessExitCause,
};

fn asset(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("roms/homebrew")
        .join(relative)
}

fn parse_number(value: &str) -> usize {
    usize::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16).unwrap()
}

fn materialize_fixture(path: &Path) -> Vec<u8> {
    let source = fs::read_to_string(path).expect("acceptance fixture must be readable");
    let mut image = None;
    for (line_number, raw) in source.lines().enumerate() {
        let line = raw.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let fields = line.split_ascii_whitespace().collect::<Vec<_>>();
        match fields.as_slice() {
            ["size", size] if image.is_none() => image = Some(vec![0; parse_number(size)]),
            ["u32", offset, value] => {
                let image = image.as_mut().expect("size must precede writes");
                let offset = parse_number(offset);
                let value = u32::try_from(parse_number(value)).unwrap().to_le_bytes();
                image[offset..offset + value.len()].copy_from_slice(&value);
            }
            ["bytes", offset, value] => {
                let image = image.as_mut().expect("size must precede writes");
                let offset = parse_number(offset);
                assert!(value.len().is_multiple_of(2));
                let bytes = (0..value.len())
                    .step_by(2)
                    .map(|index| u8::from_str_radix(&value[index..index + 2], 16).unwrap())
                    .collect::<Vec<_>>();
                image[offset..offset + bytes.len()].copy_from_slice(&bytes);
            }
            ["fill", offset, size, value] => {
                let image = image.as_mut().expect("size must precede writes");
                let offset = parse_number(offset);
                let size = parse_number(size);
                let value = u8::try_from(parse_number(value)).unwrap();
                image[offset..offset + size].fill(value);
            }
            _ => panic!("invalid fixture directive at line {}", line_number + 1),
        }
    }
    image.expect("fixture must declare its size")
}

#[test]
fn minimal_nro_enters_real_abi_resumes_from_svc_and_returns_to_loader() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("minimal-a64.nro");
    fs::write(
        &path,
        materialize_fixture(&asset("acceptance/minimal-a64.nro.fixture")),
    )
    .unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    let mut process = ProcessBuilder::new().build(&plan).unwrap();
    let ThreadCpuState::A64(state) = &process.main_thread().state else {
        panic!("NRO must enter in A64 state")
    };
    assert_ne!(
        state.read_x(A64Register::General(A64GeneralRegister::new(0).unwrap())),
        0
    );
    assert_eq!(
        state.read_x(A64Register::General(A64GeneralRegister::new(1).unwrap())),
        u64::MAX
    );
    assert_eq!(
        state.read_x(A64Register::General(A64GeneralRegister::new(30).unwrap())),
        process.main_thread().loader_return.unwrap().get()
    );

    let mut dispatcher = HorizonSvcDispatcher::default();
    let first = process.run_reference(16).unwrap();
    assert!(matches!(
        first.stop,
        ExecutionStop::SupervisorCall {
            immediate: 0x10,
            ..
        }
    ));
    assert_eq!(
        process
            .route_supervisor_call(&first.stop, &mut dispatcher)
            .unwrap(),
        ExceptionHandlingResult::<HorizonSvcFault>::Resumed
    );

    let second = process.run_reference(16).unwrap();
    assert!(matches!(
        second.stop,
        ExecutionStop::LoaderReturn { result_code: 0, .. }
    ));
    assert_eq!(process.execution_status(), ProcessExecutionStatus::Exited);
    assert_eq!(
        process.exit().unwrap().cause,
        ProcessExitCause::LoaderReturned
    );
    assert_eq!(dispatcher.coverage().len(), 1);
    assert_eq!(dispatcher.coverage()[0].immediate, 0x10);
    assert_eq!(
        dispatcher.coverage()[0].support,
        HorizonSvcSupport::Complete
    );

    let teardown = process.teardown();
    assert_eq!(teardown.previous_status, ProcessExecutionStatus::Exited);
    assert!(teardown.threads_released > 0);
    assert!(teardown.physical_pages_released > 0);
}

#[test]
fn contemporary_libnx_nro_initializes_hid_and_time_then_reaches_libc_time_setup() {
    let path = asset("templates/application/application.nro");
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    let mut process = ProcessBuilder::new().build(&plan).unwrap();
    let mut dispatcher = HorizonSvcDispatcher::default();
    let mut executed = 0_u64;

    let reached_libc_time_setup = loop {
        let report = process.run_reference(512).unwrap();
        executed += report.instructions_executed;
        assert!(
            executed <= 20_000,
            "libnx startup exceeded its acceptance bound"
        );
        match &report.stop {
            ExecutionStop::BudgetExhausted => {}
            ExecutionStop::SupervisorCall { .. } => {
                let outcome = process
                    .route_supervisor_call(&report.stop, &mut dispatcher)
                    .unwrap();
                match outcome {
                    ExceptionHandlingResult::Resumed => {}
                    ExceptionHandlingResult::Terminated { .. }
                        if matches!(
                            &report.stop,
                            ExecutionStop::SupervisorCall {
                                immediate: 0x26,
                                ..
                            }
                        ) =>
                    {
                        break true;
                    }
                    _ => panic!(
                        "libnx SVC failed at {stop}: {outcome:?}",
                        stop = report.stop
                    ),
                }
            }
            ExecutionStop::UnallocatedEncoding { error }
                if error.instruction.encoding.bits() == 0x4cdf_a041 =>
            {
                break true;
            }
            stop => panic!("libnx startup stopped before the libc time-setup frontier: {stop}"),
        }
    };

    assert!(
        reached_libc_time_setup && executed > 9_000,
        "libnx did not initialize HID/time and reach libc time setup: executed={executed}"
    );
    let coverage = dispatcher.coverage();
    for immediate in [0x01, 0x02, 0x03, 0x06, 0x13, 0x29] {
        assert!(
            coverage.iter().any(|entry| {
                entry.immediate == immediate && entry.support != HorizonSvcSupport::Unsupported
            }),
            "libnx did not exercise required SVC {immediate:#x}"
        );
    }
    for immediate in [0x1f, 0x21] {
        assert!(
            coverage.iter().any(|entry| {
                entry.immediate == immediate
                    && entry.support == HorizonSvcSupport::Complete
                    && entry.resumed > 0
            }),
            "missing completed SVC {immediate:#x}; coverage={coverage:?}"
        );
    }
}
