use std::collections::BTreeSet;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use nixe_cli::library::{Library, LibraryTitleSource};
use nixe_config::{
    CpuEngineSelection, DiagnosticReportDetail, DiagnosticsConfig, InitialOperationMode, TimeMode,
};
use nixe_cpu_engine::{EngineCapabilities, EnginePreference, EngineProvider, EngineRegistry};
use nixe_cpu_engine_interpreter::{INTERPRETER_ENGINE_ID, InterpreterProvider};
use nixe_gpu::BackendInstanceId;
use nixe_gpu_wgpu::{WgpuBackendConfiguration, initialize_backend};
use nixe_horizon::{
    HorizonSvcDispatcher, HorizonSvcFault, OperationMode, TimeEnvironment,
    UnsupportedNvDrvOperation, VideoSystem, switch_1_machine_profile,
};
use nixe_input::{
    ControllerId, EmulatedButtonState, GamepadProfiles, InputManager, ProfiledControllerState,
    sdl::SdlInputBackend,
};
use nixe_memory::NonCpuDeviceId;
use nixe_runtime::{
    DiagnosticsPolicy, ExceptionHandlingResult, ExecutionStop, Launcher, LauncherInput,
    ProcessBuilder, ProcessExit, ProcessExitCause, ProcessRegistration, ProcessTeardownReport,
    ReportDetail, RunnableProcess, RuntimeCoordinator, VcpuExecutionMode, VirtualClock,
    VirtualClockMode,
};
use nixe_scheduler::ProcessId;
use nixe_video::FrameMailbox;
use nixe_video_winit::{FrontendControl, WindowFrontend};

use crate::logging::LogLevel;

use super::load_config;

const EXECUTION_PROGRESS_INTERVAL: u64 = 10_000_000;
const INPUT_POLL_INTERVAL: Duration = Duration::from_millis(5);
const MAXWELL_PUSHBUFFER_DUMP_DIRECTORY: &str = "dump";
const MAXWELL_PUSHBUFFER_DUMP_FILENAME: &str = "pushbuffer.bin";

pub struct Arguments {
    pub config_path: Option<PathBuf>,
    pub log_level_override: Option<LogLevel>,
    pub identifier: String,
    pub headless: bool,
}

pub fn run(arguments: Arguments) -> Result<(), String> {
    let Arguments {
        config_path,
        log_level_override,
        identifier,
        headless,
    } = arguments;
    let frontend_stop_requested = Arc::new(AtomicBool::new(false));
    let (frontend, frontend_control, mailbox) = if headless {
        log::info!("headless presentation enabled; no host window will be created");
        (None, None, FrameMailbox::default())
    } else {
        let frontend = WindowFrontend::new(Arc::clone(&frontend_stop_requested))
            .map_err(|error| error.to_string())?;
        let control = frontend.control();
        let mailbox = frontend.mailbox();
        (Some(frontend), Some(control), mailbox)
    };
    let machine_profile = switch_1_machine_profile();
    let scheduler_profile = machine_profile.scheduler().clone();
    let config = load_config(config_path, log_level_override)?;
    log::info!("scanning configured title library");
    std::fs::create_dir_all(&config.filesystem.sd_card).map_err(|error| {
        format!(
            "cannot create configured SD-card directory {}: {error}",
            config.filesystem.sd_card.display()
        )
    })?;
    let sd_card_root = std::fs::canonicalize(&config.filesystem.sd_card).map_err(|error| {
        format!(
            "cannot resolve configured SD-card directory {}: {error}",
            config.filesystem.sd_card.display()
        )
    })?;
    log::debug!("SD card host directory: {}", sd_card_root.display());
    let scan_started = Instant::now();
    let library = Library::scan(&config)?;
    log::debug!(
        "configured title library scanned in {:?}",
        scan_started.elapsed()
    );
    let title = library
        .find(&identifier)
        .ok_or_else(|| format!("unknown title ID or name: {identifier}"))?;
    log::info!("selected {}: {}", title.identifier, title.name);

    let plan_started = Instant::now();
    let plan = match &title.source {
        LibraryTitleSource::Installed(title) => {
            log::info!(
                "source is an installed title; building from the resolved base, update and DLC set"
            );
            Launcher::build_resolved_title((**title).clone(), &library.keys)
        }
        LibraryTitleSource::Homebrew(path) => {
            log::info!("source is a homebrew NRO: {}", path.display());
            Launcher::build(LauncherInput::new(path))
        }
    }
    .map_err(|error| error.to_string())?;
    log::debug!("launch plan built in {:?}", plan_started.elapsed());
    log::info!(
        "launch plan ready: {} module(s), entry={}, primary RomFS={}, DLC={}",
        plan.modules().len(),
        plan.entry_module().name(),
        if plan.primary_file_system().is_some() {
            "yes"
        } else {
            "no"
        },
        plan.add_ons().len()
    );
    for module in plan.modules() {
        log::info!(
            "module {} ({:?}) loaded into the plan",
            module.name(),
            module.role()
        );
    }

    log::info!("preparing process memory and initial thread state");
    let mut diagnostics = diagnostics_policy(config.diagnostics);
    let instruction_trace = log::log_enabled!(log::Level::Trace);
    if instruction_trace {
        diagnostics.instruction_trace = true;
        log::info!("instruction trace enabled; execution will be substantially slower");
    }
    let clock_mode = match config.system.time.mode {
        TimeMode::Realtime => VirtualClockMode::Realtime,
        TimeMode::Fixed => VirtualClockMode::Fixed {
            unix_seconds: config
                .system
                .time
                .fixed_unix_timestamp
                .expect("fixed time configuration was validated"),
        },
    };
    let virtual_clock = VirtualClock::new(clock_mode);
    let execution_mode = if config.cpu.parallel_vcpus {
        VcpuExecutionMode::Parallel
    } else {
        VcpuExecutionMode::Deterministic
    };
    let coordinator = RuntimeCoordinator::try_with_execution_mode(
        scheduler_profile,
        virtual_clock.clone(),
        execution_mode,
    )
    .map_err(|error| error.to_string())?;
    let external_events = coordinator.event_sender();
    install_interrupt_handler(frontend_control.clone(), external_events.clone())?;
    let process_started = Instant::now();
    let engine_provider = select_cpu_engine(
        config.cpu.engine,
        diagnostics,
        config.cpu.parallel_vcpus,
        machine_profile.cpu(),
        machine_profile.scheduler().vcpus().len(),
    )?;
    let process = ProcessBuilder::new()
        .with_diagnostics(diagnostics)
        .with_virtual_clock(virtual_clock.clone())
        .with_sd_card_root(sd_card_root)
        .with_config(machine_profile.process_build_config())
        .with_engine_provider(engine_provider)
        .build(&plan)
        .map_err(|error| error.to_string())?;
    log::debug!("process prepared in {:?}", process_started.elapsed());
    log::info!(
        "process ready: entry={:#018x}, modules={}",
        process.entry_module().entry_address(),
        process.modules().len()
    );
    log::info!("starting the reference CPU interpreter");

    let initial_operation_mode = match config.system.initial_operation_mode {
        InitialOperationMode::Handheld => OperationMode::Handheld,
        InitialOperationMode::Docked => OperationMode::Console,
    };
    log::debug!("initial operation mode: {initial_operation_mode:?}");
    let time_environment = TimeEnvironment::new(virtual_clock, &config.system.time.timezone)
        .map_err(|error| format!("cannot create Horizon time environment: {error}"))?;
    log::debug!(
        "virtual time: mode={clock_mode:?}, timezone={}",
        config.system.time.timezone
    );
    let gpu_backend = initialize_backend(
        BackendInstanceId::new(1),
        NonCpuDeviceId::new(1),
        WgpuBackendConfiguration::default(),
    )
    .map_err(|error| format!("cannot initialize accelerated GPU backend: {error}"))?;
    log::info!(
        "GPU backend initialized: api={:?} adapter={} driver={}",
        gpu_backend.adapter.backend,
        gpu_backend.adapter.name,
        gpu_backend.adapter.driver
    );
    let video_system = VideoSystem::with_gpu_backend(mailbox, gpu_backend.into_runtime());
    let gamepad_profiles = GamepadProfiles::new(config.input.profiles.clone());

    let Some(frontend) = frontend else {
        return finish_execution(execute_worker(
            coordinator,
            process,
            instruction_trace,
            initial_operation_mode,
            time_environment,
            video_system,
            gamepad_profiles,
        ));
    };

    let worker_control =
        frontend_control.expect("window frontend construction provides its control channel");
    let worker = thread::Builder::new()
        .name("nixe-guest".to_owned())
        .spawn(move || {
            let _completion = WorkerCompletion(worker_control);
            execute_worker(
                coordinator,
                process,
                instruction_trace,
                initial_operation_mode,
                time_environment,
                video_system,
                gamepad_profiles,
            )
        })
        .map_err(|error| format!("cannot start guest execution worker: {error}"))?;

    let frontend_result = frontend.run().map_err(|error| error.to_string());
    frontend_stop_requested.store(true, Ordering::Release);
    let _ = external_events.submit(nixe_runtime::ExternalEvent::HostStop);
    let worker_result = worker
        .join()
        .map_err(|_| "guest execution worker panicked".to_owned())?;
    let execution_result = finish_execution(worker_result);
    frontend_result.and(execution_result)
}

fn diagnostics_policy(config: DiagnosticsConfig) -> DiagnosticsPolicy {
    DiagnosticsPolicy {
        report_detail: match config.report_detail {
            DiagnosticReportDetail::Detailed => ReportDetail::Detailed,
            DiagnosticReportDetail::Sanitized => ReportDetail::Sanitized,
        },
        instruction_trace: config.instruction_trace,
        ..DiagnosticsPolicy::default()
    }
}

fn select_cpu_engine(
    selection: CpuEngineSelection,
    diagnostics: DiagnosticsPolicy,
    parallel_vcpus: bool,
    profile: nixe_cpu::profile::GuestCpuProfile,
    vcpu_count: usize,
) -> Result<Arc<dyn EngineProvider>, String> {
    let interpreter: Arc<dyn EngineProvider> = Arc::new(InterpreterProvider);
    let registry = EngineRegistry::new([interpreter]);
    let preference = match selection {
        CpuEngineSelection::Auto => EnginePreference::Auto,
        CpuEngineSelection::Interpreter => EnginePreference::Explicit(INTERPRETER_ENGINE_ID),
    };
    registry
        .select(
            profile,
            EngineCapabilities {
                a64: true,
                a32: false,
                t32: false,
                precise_instruction_budget: true,
                instruction_trace: diagnostics.instruction_trace,
                interpret_one_fallback: false,
                native_execution: false,
                concurrent_executors: parallel_vcpus,
                max_safepoint_instructions: parallel_vcpus
                    .then(|| std::num::NonZeroU64::new(u64::MAX).unwrap()),
                acknowledged_invalidation: parallel_vcpus,
                canonical_state_version: 1,
                deterministic_execution: !parallel_vcpus,
                precise_exceptions: true,
                engine_handoff: true,
                canonical_memory_binding: false,
                max_concurrent_executors: parallel_vcpus.then(|| {
                    std::num::NonZeroUsize::new(vcpu_count)
                        .expect("a machine profile contains at least one vCPU")
                }),
            },
            preference,
        )
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod engine_selection_tests {
    use super::*;

    #[test]
    fn application_composition_resolves_auto_and_explicit_interpreter() {
        let profile = switch_1_machine_profile();
        for selection in [CpuEngineSelection::Auto, CpuEngineSelection::Interpreter] {
            let provider = select_cpu_engine(
                selection,
                DiagnosticsPolicy::default(),
                false,
                profile.cpu(),
                profile.scheduler().vcpus().len(),
            )
            .unwrap();
            assert_eq!(provider.descriptor().id, INTERPRETER_ENGINE_ID);
        }
        assert!(
            select_cpu_engine(
                CpuEngineSelection::Interpreter,
                DiagnosticsPolicy::default(),
                true,
                profile.cpu(),
                profile.scheduler().vcpus().len(),
            )
            .is_ok()
        );
    }
}

fn install_interrupt_handler(
    control: Option<FrontendControl>,
    external_events: nixe_runtime::ExternalEventSender,
) -> Result<(), String> {
    ctrlc::set_handler(move || {
        let _ = external_events.submit(nixe_runtime::ExternalEvent::HostStop);
        if let Some(control) = &control {
            control.stop_requested();
        }
    })
    .map_err(|error| format!("cannot install Ctrl+C handler: {error}"))?;
    Ok(())
}

struct WorkerCompletion(FrontendControl);

impl Drop for WorkerCompletion {
    fn drop(&mut self) {
        self.0.worker_finished();
    }
}

struct WorkerResult {
    execution: Result<ExecutionSummary, String>,
    teardown: ProcessTeardownReport,
    graphics_teardown: nixe_horizon::GraphicsTeardownReport,
}

fn execute_worker(
    mut coordinator: RuntimeCoordinator,
    process: RunnableProcess,
    instruction_trace: bool,
    initial_operation_mode: OperationMode,
    time_environment: TimeEnvironment,
    video_system: VideoSystem,
    gamepad_profiles: GamepadProfiles,
) -> WorkerResult {
    let registration = ProcessRegistration {
        priority: process.initial_thread_priority(),
        ideal_vcpu: Some(process.initial_ideal_vcpu()),
        affinity: coordinator.scheduler().profile().all_cores(),
    };
    let process_id = coordinator
        .register_process(process, registration)
        .expect("CLI process and verified Switch 1 scheduler profile are compatible");
    let execution_started = Instant::now();
    let execution_video = video_system.clone();
    let mut execution = SdlInputBackend::new()
        .map_err(|error| error.to_string())
        .and_then(|backend| {
            let mut input = InputManager::with_profiles(backend, gamepad_profiles);
            let mut scheduled = ScheduledProcess {
                coordinator: &mut coordinator,
                process_id,
            };
            execute(
                &mut scheduled,
                instruction_trace,
                initial_operation_mode,
                time_environment,
                execution_video,
                &mut input,
            )
        });
    log::debug!(
        "guest execution stopped after {:?}",
        execution_started.elapsed()
    );
    let process = coordinator
        .remove_process(process_id)
        .expect("serialized execution returns every scheduler lease");
    let teardown = match process.try_teardown() {
        Ok(report) => report,
        Err(failure) => {
            let diagnostic = failure.to_string();
            let report = *failure.report;
            execution = Err(match execution {
                Ok(_) => diagnostic,
                Err(error) => format!("{error}; {diagnostic}"),
            });
            report
        }
    };
    WorkerResult {
        execution,
        teardown,
        graphics_teardown: video_system.teardown(),
    }
}

fn finish_execution(result: WorkerResult) -> Result<(), String> {
    log::debug!(
        "resources released: handles={}, address_waiters={}, layers={}, queues={}, \
         pending_frames={}, nvdrv_fds={}, nvmap_allocations={}",
        result.teardown.handles_released,
        result.teardown.address_waiters_released,
        result.graphics_teardown.layers_released,
        result.graphics_teardown.queues_released,
        result.graphics_teardown.pending_frames_released,
        result.graphics_teardown.device_fds_released,
        result.graphics_teardown.allocations_released,
    );
    let summary = match result.execution {
        Ok(summary) => summary,
        Err(error) => {
            log::info!("process resources released after failure: {error}");
            return Err(error);
        }
    };
    let exit_code = result.teardown.exit.map_or(0, |exit| exit.exit_code);
    let exit_cause = result.teardown.exit.map_or_else(
        || "without an exit record".to_owned(),
        |exit| format!("{:?}", exit.cause),
    );
    log::info!(
        "execution finished: instructions={}, SVC calls={}, rejected SVC kinds={}, cause={}, code={:#x}",
        summary.instructions,
        summary.svc_calls,
        summary.rejected_svc_kinds,
        exit_cause,
        exit_code
    );
    classify_exit(result.teardown.exit)
}

struct ExecutionSummary {
    instructions: u64,
    svc_calls: u64,
    rejected_svc_kinds: usize,
}

struct ScheduledProcess<'a> {
    coordinator: &'a mut RuntimeCoordinator,
    process_id: ProcessId,
}

fn execute(
    scheduled: &mut ScheduledProcess<'_>,
    print_trace: bool,
    initial_operation_mode: OperationMode,
    time_environment: TimeEnvironment,
    video_system: VideoSystem,
    input: &mut InputManager<SdlInputBackend>,
) -> Result<ExecutionSummary, String> {
    let coordinator = &mut *scheduled.coordinator;
    let process_id = scheduled.process_id;
    let mut dispatcher = HorizonSvcDispatcher::new_with_video(
        initial_operation_mode,
        time_environment,
        video_system,
    );
    let mut instructions = 0_u64;
    let execution_started = Instant::now();
    let mut next_progress = EXECUTION_PROGRESS_INTERVAL;
    let mut last_progress_instructions = 0_u64;
    let mut last_progress_elapsed = Duration::ZERO;
    let mut rejected = BTreeSet::new();
    let mut last_trace_sequence = None;
    let mut next_input_poll = Duration::ZERO;
    let mut last_input_poll = Duration::ZERO;
    let mut active_input = None;
    let mut input_observed = false;
    let mut active_buttons = EmulatedButtonState::default();
    loop {
        dispatcher.synchronize_virtual_time(coordinator.virtual_time_ns());
        coordinator
            .drain_external_events()
            .map_err(|error| error.to_string())?;
        if coordinator.host_stop_requested() {
            log::info!("host stop received; stopping the guest process cleanly");
            if !coordinator
                .terminate_process(process_id)
                .map_err(|error| error.to_string())?
            {
                return Err("host stop could not terminate the guest process cleanly".to_owned());
            }
            return Ok(execution_summary(instructions, &dispatcher, rejected.len()));
        }
        let elapsed = execution_started.elapsed();
        dispatcher
            .advance_video(elapsed)
            .map_err(|error| error.to_string())?;
        if elapsed >= next_input_poll {
            let profiled = input
                .read_profiled_input()
                .map_err(|error| error.to_string())?;
            report_input_change(&mut active_input, &profiled, input_observed);
            let current_buttons = profiled
                .as_ref()
                .map_or_else(EmulatedButtonState::default, |controller| {
                    controller.state.buttons
                });
            if log::log_enabled!(log::Level::Debug) {
                for (button, pressed) in button_transitions(active_buttons, current_buttons) {
                    log::debug!(
                        "emulated button {button} {}: instructions={instructions} elapsed={elapsed:?}",
                        if pressed { "pressed" } else { "released" }
                    );
                }
            }
            active_buttons = current_buttons;
            input_observed = true;
            dispatcher
                .advance_input(
                    coordinator
                        .process_mut(process_id)
                        .expect("registered process remains available"),
                    profiled.as_ref().map(|controller| &controller.state),
                    elapsed.saturating_sub(last_input_poll),
                )
                .map_err(|error| format!("cannot publish Horizon HID state: {error}"))?;
            last_input_poll = elapsed;
            next_input_poll = elapsed.saturating_add(INPUT_POLL_INTERVAL);
        }
        let instruction_budget = if print_trace {
            1
        } else {
            coordinator
                .scheduler()
                .profile()
                .default_timeslice_instructions()
        };
        let executions = match coordinator.execution_mode() {
            VcpuExecutionMode::Deterministic => coordinator
                .run_next(instruction_budget)
                .map(|execution| execution.into_iter().collect()),
            VcpuExecutionMode::Parallel => coordinator.run_parallel_wave(instruction_budget),
        }
        .map_err(|error| error.to_string())?;
        if executions.is_empty() {
            let host_wait =
                host_service_wait_duration(execution_started.elapsed(), next_input_poll);
            coordinator
                .wait_for_external_event_for(host_wait)
                .map_err(|error| error.to_string())?;
            continue;
        }
        for execution in executions {
            let thread_id = execution.lease.thread;
            let report = execution.report;
            instructions = instructions.saturating_add(report.instructions_executed);
            if log::log_enabled!(log::Level::Debug) && instructions >= next_progress {
                let elapsed = execution_started.elapsed();
                let interval_instructions = instructions.saturating_sub(last_progress_instructions);
                let interval_elapsed = elapsed.saturating_sub(last_progress_elapsed);
                let interval_ips = instructions_per_second(interval_instructions, interval_elapsed);
                log::debug!(
                    "guest execution progress: instructions={instructions}, elapsed={elapsed:?}, interval_ips={interval_ips:.0}"
                );
                last_progress_instructions = instructions;
                last_progress_elapsed = elapsed;
                next_progress = next_progress.saturating_add(EXECUTION_PROGRESS_INTERVAL);
            }
            if print_trace {
                for entry in report.trace.entries() {
                    if last_trace_sequence.is_none_or(|sequence| entry.sequence > sequence) {
                        log::trace!("{entry}");
                        last_trace_sequence = Some(entry.sequence);
                    }
                }
            }
            match &report.stop {
                ExecutionStop::BudgetExhausted
                | ExecutionStop::Safepoint
                | ExecutionStop::PendingEvent { .. } => {}
                ExecutionStop::Scheduled { .. } => {
                    coordinator
                        .make_thread_ready(thread_id)
                        .map_err(|error| error.to_string())?;
                }
                ExecutionStop::SupervisorCall { .. } => {
                    let handling = dispatcher
                        .route_scheduled_supervisor_call(coordinator, execution.lease, &report.stop)
                        .map_err(|error| error.to_string())?;
                    match handling {
                        ExceptionHandlingResult::Resumed => {}
                        ExceptionHandlingResult::Rejected(error) => {
                            let diagnostic = error.to_string();
                            if rejected.insert(diagnostic.clone()) {
                                log::debug!(
                                    "guest operation returned a Horizon error: {diagnostic}"
                                );
                            }
                        }
                        ExceptionHandlingResult::Terminated { .. } => {
                            return Ok(execution_summary(
                                instructions,
                                &dispatcher,
                                rejected.len(),
                            ));
                        }
                        ExceptionHandlingResult::Suspended => {}
                        ExceptionHandlingResult::Fault(error) => {
                            dump_maxwell_pushbuffer_on_fault(&error);
                            return Err(format!(
                                "Horizon SVC dispatch failed after {instructions} instructions: {error}; {report}"
                            ));
                        }
                    }
                }
                ExecutionStop::LoaderReturn { .. } => {
                    return Ok(execution_summary(instructions, &dispatcher, rejected.len()));
                }
                stop => return Err(execution_stop_error(stop, instructions, &report)),
            }
        }
    }
}

fn dump_maxwell_pushbuffer_on_fault(fault: &HorizonSvcFault) {
    let HorizonSvcFault::UnsupportedNvDrv {
        operation: UnsupportedNvDrvOperation::ScheduledGpfifoSubmission { boundary, .. },
        ..
    } = fault
    else {
        return;
    };
    let Some(capture) = boundary.frontend_capture() else {
        return;
    };
    let directory = PathBuf::from(MAXWELL_PUSHBUFFER_DUMP_DIRECTORY);
    let path = directory.join(MAXWELL_PUSHBUFFER_DUMP_FILENAME);

    let result = (|| -> io::Result<()> {
        std::fs::create_dir_all(&directory)?;
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        write_nv_push_dump_words(&mut writer, capture.words().iter().map(|word| word.value()))?;
        writer.flush()
    })();

    match result {
        Ok(()) => log::info!(
            "Maxwell pushbuffer dumped for nv_push_dump: path={} words={} complete={}",
            path.display(),
            capture.words().len(),
            capture.is_complete(),
        ),
        Err(error) => log::warn!(
            "cannot dump Maxwell pushbuffer to {}: {error}",
            path.display()
        ),
    }
}

fn write_nv_push_dump_words(
    writer: &mut impl Write,
    words: impl IntoIterator<Item = u32>,
) -> io::Result<()> {
    // Mesa's nv_push_dump consumes a headerless array of native uint32_t
    // command words. Nixe emits the Switch/Maxwell little-endian form.
    // https://android.googlesource.com/platform/external/mesa3d/+/refs/tags/upstream-mesa-26.0.6/src/nouveau/headers/nv_push_dump.c
    for word in words {
        writer.write_all(&word.to_le_bytes())?;
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ActiveInput {
    controller_id: ControllerId,
    device: String,
    profile_name: String,
}

fn report_input_change(
    active: &mut Option<ActiveInput>,
    current: &Option<ProfiledControllerState>,
    previously_observed: bool,
) {
    let next = current.as_ref().map(|controller| ActiveInput {
        controller_id: controller.controller_id,
        device: controller.device.clone(),
        profile_name: controller.profile_name.clone(),
    });
    if *active == next && previously_observed {
        return;
    }
    match &next {
        Some(controller) => log::info!(
            "using input profile `{}` for {}",
            controller.profile_name,
            controller.device
        ),
        None if previously_observed && active.is_some() => {
            log::info!("mapped gamepad disconnected; player one is now disconnected");
        }
        None => log::debug!("no matching first-gamepad input profile; player one is disconnected"),
    }
    *active = next;
}

fn instructions_per_second(instructions: u64, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        0.0
    } else {
        instructions as f64 / elapsed.as_secs_f64()
    }
}

fn host_service_wait_duration(now: Duration, next_input_poll: Duration) -> Duration {
    next_input_poll.saturating_sub(now)
}

fn button_transitions(
    previous: EmulatedButtonState,
    current: EmulatedButtonState,
) -> impl Iterator<Item = (&'static str, bool)> {
    [
        ("A", previous.a, current.a),
        ("B", previous.b, current.b),
        ("X", previous.x, current.x),
        ("Y", previous.y, current.y),
        ("Plus", previous.plus, current.plus),
        ("Minus", previous.minus, current.minus),
        ("Home", previous.home, current.home),
        ("Capture", previous.capture, current.capture),
        ("L", previous.l, current.l),
        ("R", previous.r, current.r),
        ("ZL", previous.zl, current.zl),
        ("ZR", previous.zr, current.zr),
        ("LeftStick", previous.left_stick, current.left_stick),
        ("RightStick", previous.right_stick, current.right_stick),
        ("DPadUp", previous.dpad_up, current.dpad_up),
        ("DPadDown", previous.dpad_down, current.dpad_down),
        ("DPadLeft", previous.dpad_left, current.dpad_left),
        ("DPadRight", previous.dpad_right, current.dpad_right),
    ]
    .into_iter()
    .filter_map(|(name, previous, current)| (previous != current).then_some((name, current)))
}

fn classify_exit(exit: Option<ProcessExit>) -> Result<(), String> {
    let Some(exit) = exit else {
        return Err("guest execution ended without an exit record".to_owned());
    };
    match exit.cause {
        ProcessExitCause::ProcessRequested
        | ProcessExitCause::LastThreadExited
        | ProcessExitCause::LoaderReturned => {
            if exit.exit_code == 0 {
                Ok(())
            } else {
                Err(format!(
                    "title exited normally with non-zero code {:#x} ({:?})",
                    exit.exit_code, exit.cause
                ))
            }
        }
        ProcessExitCause::HostRequested => {
            if exit.exit_code == 0 {
                Ok(())
            } else {
                Err(format!(
                    "guest execution was interrupted by the host with non-zero code {:#x}",
                    exit.exit_code
                ))
            }
        }
        ProcessExitCause::GuestBreak { reason, info, size } => Err(format!(
            "guest requested a fatal break: reason={reason:#x}, info={info:#x}, size={size:#x}, code={:#x}",
            exit.exit_code
        )),
    }
}

fn execution_summary(
    instructions: u64,
    dispatcher: &HorizonSvcDispatcher,
    rejected_svc_kinds: usize,
) -> ExecutionSummary {
    ExecutionSummary {
        instructions,
        svc_calls: dispatcher.coverage().iter().map(|entry| entry.calls).sum(),
        rejected_svc_kinds,
    }
}

fn execution_stop_error(
    stop: &ExecutionStop,
    instructions: u64,
    report: &nixe_runtime::ExecutionReport,
) -> String {
    let reason = match stop {
        ExecutionStop::UnsupportedSemantics {
            source,
            encoding,
            disassembly,
            coverage_id,
        } => format!(
            "CPU instruction semantics are not implemented: source=[{source}] encoding={encoding} instruction={disassembly} coverage={coverage_id}"
        ),
        ExecutionStop::ProfileDisabled { error } => {
            format!("CPU instruction is disabled by the selected CPU profile: {error}")
        }
        ExecutionStop::UnallocatedEncoding { error } => {
            format!("guest executed an unallocated instruction encoding: {error}")
        }
        ExecutionStop::FetchFault { fault } => {
            format!("instruction fetch failed: {fault}")
        }
        ExecutionStop::ArchitecturalException {
            source,
            kind,
            syndrome,
        } => format!(
            "unhandled architectural exception: source=[{source}] kind={kind:?} syndrome={syndrome:?}"
        ),
        ExecutionStop::DataFault { source, fault } => {
            format!("guest memory access failed: source=[{source}] fault={fault:?}")
        }
        _ => format!("unexpected execution stop: {stop}"),
    };
    format!("{reason} after {instructions} instructions; diagnostic: {report}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runnable_process_can_be_owned_by_the_guest_worker() {
        fn assert_send<T: Send>() {}

        assert_send::<RunnableProcess>();
    }

    #[test]
    fn computes_instruction_rate_for_the_progress_interval() {
        assert_eq!(
            instructions_per_second(20_000_000, Duration::from_millis(2_500)),
            8_000_000.0
        );
        assert_eq!(instructions_per_second(1, Duration::ZERO), 0.0);
    }

    #[test]
    fn host_service_wait_is_bounded_by_the_next_input_deadline() {
        assert_eq!(
            host_service_wait_duration(Duration::from_millis(7), Duration::from_millis(12)),
            Duration::from_millis(5)
        );
        assert_eq!(
            host_service_wait_duration(Duration::from_millis(12), Duration::from_millis(12)),
            Duration::ZERO
        );
        assert_eq!(
            host_service_wait_duration(Duration::from_millis(13), Duration::from_millis(12)),
            Duration::ZERO
        );
    }

    #[test]
    fn button_transitions_report_every_changed_emulated_button_once() {
        let all_pressed = EmulatedButtonState {
            a: true,
            b: true,
            x: true,
            y: true,
            plus: true,
            minus: true,
            home: true,
            capture: true,
            l: true,
            r: true,
            zl: true,
            zr: true,
            left_stick: true,
            right_stick: true,
            dpad_up: true,
            dpad_down: true,
            dpad_left: true,
            dpad_right: true,
        };

        let pressed =
            button_transitions(EmulatedButtonState::default(), all_pressed).collect::<Vec<_>>();
        assert_eq!(pressed.len(), 18);
        assert!(pressed.iter().all(|(_, state)| *state));
        assert!(pressed.contains(&("Plus", true)));
        assert!(pressed.contains(&("DPadRight", true)));
        assert_eq!(button_transitions(all_pressed, all_pressed).count(), 0);
        let released =
            button_transitions(all_pressed, EmulatedButtonState::default()).collect::<Vec<_>>();
        assert_eq!(released.len(), 18);
        assert!(released.iter().all(|(_, state)| !state));
    }

    #[test]
    fn nv_push_dump_words_are_headerless_little_endian_u32_values() {
        let mut bytes = Vec::new();
        write_nv_push_dump_words(&mut bytes, [0x2001_4000, 0x0002_0002]).unwrap();

        assert_eq!(bytes, [0x00, 0x40, 0x01, 0x20, 0x02, 0x00, 0x02, 0x00]);
    }

    fn process_exit(cause: ProcessExitCause, exit_code: u64) -> ProcessExit {
        ProcessExit {
            cause,
            exit_code,
            source: None,
            thread_id: 1,
        }
    }

    #[test]
    fn accepts_only_zero_code_normal_guest_terminations() {
        for cause in [
            ProcessExitCause::ProcessRequested,
            ProcessExitCause::LastThreadExited,
            ProcessExitCause::LoaderReturned,
        ] {
            assert_eq!(classify_exit(Some(process_exit(cause, 0))), Ok(()));
            assert!(classify_exit(Some(process_exit(cause, 7))).is_err());
        }
    }

    #[test]
    fn accepts_clean_host_termination_and_rejects_other_host_exit_codes() {
        assert_eq!(
            classify_exit(Some(process_exit(ProcessExitCause::HostRequested, 0))),
            Ok(())
        );
        assert!(classify_exit(Some(process_exit(ProcessExitCause::HostRequested, 7))).is_err());
        assert!(classify_exit(None).is_err());
    }

    #[test]
    fn rejects_fatal_guest_breaks_even_when_the_code_is_zero() {
        let error = classify_exit(Some(process_exit(
            ProcessExitCause::GuestBreak {
                reason: 0,
                info: 0x1234,
                size: 4,
            },
            0,
        )))
        .unwrap_err();
        assert!(error.contains("fatal break"));
        assert!(error.contains("info=0x1234"));
    }
}
