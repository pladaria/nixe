use std::fs;
use std::path::{Path, PathBuf};

use nixe_gpu::GraphicsGapKind;
use nixe_horizon::{HorizonSvcDispatcher, HorizonSvcFault, UnsupportedNvDrvOperation};
use nixe_runtime::{
    ExceptionHandlingResult, ExecutionStop, Launcher, LauncherInput, ProcessBuilder,
    ProcessExecutionStatus,
};
use sha2::{Digest, Sha256};

fn baseline_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../roms/homebrew/graphics/opengl/simple_triangle")
}

#[test]
fn triangle_baseline_manifest_identifies_the_retained_artifact_and_public_sources() {
    let directory = baseline_directory();
    let manifest =
        fs::read(directory.join("baseline.toml")).expect("baseline manifest must be readable");
    let manifest: toml::Value = toml::from_str(
        std::str::from_utf8(&manifest).expect("baseline manifest must contain UTF-8"),
    )
    .expect("baseline manifest must be valid TOML");
    let artifact = fs::read(directory.join("simple_triangle.nro"))
        .expect("retained simple_triangle artifact must be readable");

    assert_eq!(manifest["schema_version"].as_integer(), Some(1));
    assert_eq!(
        manifest["source"]["revision"].as_str(),
        Some("669786898205b7beb25ff1731e72982e6d0397d3")
    );
    assert_eq!(
        manifest["libnx"]["revision"].as_str(),
        Some("dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb")
    );
    assert_eq!(
        manifest["service_table"]["reference"].as_str(),
        Some("https://switchbrew.org/w/index.php?title=NV_services&oldid=14790")
    );
    assert_eq!(
        manifest["artifact"]["size"].as_integer(),
        i64::try_from(artifact.len()).ok()
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&artifact)),
        manifest["artifact"]["sha256"].as_str().unwrap()
    );
}

#[test]
fn triangle_baseline_classifies_the_first_gap_and_pins_its_wire_layout() {
    let manifest = fs::read_to_string(baseline_directory().join("baseline.toml"))
        .expect("baseline manifest must be readable");
    let manifest: toml::Value =
        toml::from_str(&manifest).expect("baseline manifest must be valid TOML");
    let ioctls = manifest["ioctl"].as_array().unwrap();
    let tpc_masks = ioctls
        .iter()
        .find(|ioctl| ioctl["request"].as_str() == Some("0xc0184706"))
        .expect("TPC-mask ioctl must be recorded");

    assert_eq!(tpc_masks["device"].as_str(), Some("/dev/nvhost-ctrl-gpu"));
    assert_eq!(tpc_masks["wire_size"].as_integer(), Some(24));
    assert_eq!(tpc_masks["direction"].as_str(), Some("inout"));
    assert_eq!(manifest["baseline"]["graphics_gap"].as_str(), Some("ioctl"));
    assert_eq!(
        manifest["baseline"]["host_pointers_allowed"].as_bool(),
        Some(false)
    );
}

#[test]
fn failing_triangle_has_a_bounded_typed_diagnostic_and_deterministic_teardown() {
    let path = baseline_directory().join("simple_triangle.nro");
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    let mut process = ProcessBuilder::new().build(&plan).unwrap();
    let mut dispatcher = HorizonSvcDispatcher::default();
    let video = dispatcher.video_system();
    let mut executed = 0_u64;

    let operation = loop {
        let report = process.run_reference(4_096).unwrap();
        executed = executed.saturating_add(report.instructions_executed);
        assert!(
            executed <= 5_000_000,
            "triangle exceeded its baseline instruction bound: {report}"
        );
        match &report.stop {
            ExecutionStop::BudgetExhausted
            | ExecutionStop::Safepoint
            | ExecutionStop::PendingEvent { .. } => {}
            ExecutionStop::Scheduled { .. } => {
                assert!(process.resume(), "scheduled triangle did not resume");
            }
            ExecutionStop::SupervisorCall { .. } => {
                match process
                    .route_supervisor_call(&report.stop, &mut dispatcher)
                    .unwrap()
                {
                    ExceptionHandlingResult::Resumed => {}
                    ExceptionHandlingResult::Suspended => {
                        assert!(process.resume(), "suspended triangle did not resume");
                    }
                    ExceptionHandlingResult::Fault(HorizonSvcFault::UnsupportedNvDrv {
                        operation,
                        ..
                    }) => break operation,
                    outcome => {
                        panic!("triangle reached the wrong baseline outcome: {outcome:?}; {report}")
                    }
                }
            }
            stop => panic!("triangle reached the wrong baseline stop: {stop}; {report}"),
        }
    };

    assert_eq!(operation.gap_kind(), GraphicsGapKind::Ioctl);
    assert_eq!(
        operation,
        UnsupportedNvDrvOperation::Ioctl {
            device: "/dev/nvhost-ctrl-gpu",
            request: 0xc018_4706,
        }
    );
    let diagnostic = operation.to_string();
    assert_eq!(
        diagnostic,
        "graphics-gap=ioctl nvdrv ioctl is not implemented: \
         device=/dev/nvhost-ctrl-gpu request=0xc0184706"
    );
    assert!(diagnostic.len() < 256);

    let handles = process.handles().len();
    assert!(handles > 0);
    drop(dispatcher);
    let process_teardown = process.teardown();
    assert_eq!(
        process_teardown.previous_status,
        ProcessExecutionStatus::Faulted
    );
    assert_eq!(process_teardown.handles_released, handles);
    assert!(process_teardown.physical_pages_released > 0);

    let graphics_teardown = video.teardown();
    assert!(graphics_teardown.layers_released > 0);
    assert_eq!(
        graphics_teardown.queues_released,
        graphics_teardown.layers_released
    );
    assert!(graphics_teardown.device_fds_released > 0);
    assert_eq!(
        graphics_teardown.allocations_released, 0,
        "the pinned TPC-mask stop occurs before this workload creates nvmap allocations"
    );
    assert_eq!(video.teardown(), Default::default());
}
