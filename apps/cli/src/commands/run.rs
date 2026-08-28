use std::collections::BTreeSet;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use nixe_cli::library::{Library, LibraryTitleSource};
use nixe_config::{CpuBackendSelection, CpuConfig, InitialOperationMode, TimeMode};
use nixe_cpu_jit::JitConfiguration;
use nixe_gpu::BackendInstanceId;
use nixe_gpu_wgpu::{WgpuBackendConfiguration, initialize_backend};
use nixe_horizon::{
    HorizonSvcDispatcher, HorizonSvcFault, OperationMode, SettingsEnvironment, SystemLanguage,
    TimeEnvironment, UnsupportedNvDrvOperation, VideoSystem, switch_1_machine_profile,
};
use nixe_input::{
    ControllerId, EmulatedButtonState, GamepadProfiles, InputManager, ProfiledControllerState,
    sdl::SdlInputBackend,
};
use nixe_loader_title::NacpLanguage;
use nixe_memory::NonCpuDeviceId;
use nixe_runtime::{
    CpuBackendConfig, ExceptionHandlingResult, ExecutionStop, Launcher, LauncherInput,
    ProcessBuilder, ProcessExit, ProcessExitCause, ProcessRegistration, ProcessTeardownReport,
    RunnableProcess, RuntimeCoordinator, VcpuExecutionMode, VirtualClock, VirtualClockMode,
};
use nixe_scheduler::ProcessId;
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
    pub cpu_backend_override: Option<CpuBackendSelection>,
}

pub fn run(arguments: Arguments) -> Result<(), String> {
    let Arguments {
        config_path,
        log_level_override,
        identifier,
        headless,
        cpu_backend_override,
    } = arguments;
    let frontend_stop_requested = Arc::new(AtomicBool::new(false));
    let (frontend, frontend_control, presenter) = if headless {
        log::info!("headless presentation enabled; no host window will be created");
        (None, None, None)
    } else {
        let frontend = WindowFrontend::new(Arc::clone(&frontend_stop_requested))
            .map_err(|error| error.to_string())?;
        let control = frontend.control();
        let mailbox = frontend.mailbox();
        (Some(frontend), Some(control), Some(mailbox))
    };
    let machine_profile = switch_1_machine_profile();
    let scheduler_profile = machine_profile.scheduler().clone();
    let config = load_config(config_path, log_level_override)?;
    let cpu_configuration = effective_cpu_configuration(config.cpu.clone(), cpu_backend_override);
    if let Some(backend) = cpu_backend_override {
        log::info!("CPU backend selection overridden by CLI: {backend:?}");
    }
    log::info!(
        "GPU cache policy: shaders={} pipelines={} variants-per-pipeline={} bind-groups-per-table={} persistent-pipeline-cache={} MiB",
        config.gpu.shader_entries(),
        config.gpu.pipeline_entries(),
        config.gpu.pipeline_variants_per_resource(),
        config.gpu.bind_groups_per_descriptor_table(),
        config.gpu.persistent_pipeline_cache_bytes() / (1024 * 1024)
    );
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
    let execution_mode = if cpu_configuration.parallel_vcpus {
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
    let cpu_backend = select_cpu_backend(cpu_configuration, &title.name)?;
    let trace_interpreter = matches!(cpu_backend, CpuBackendConfig::Interpreter)
        && log::log_enabled!(log::Level::Trace);
    let process_builder = ProcessBuilder::new()
        .with_virtual_clock(virtual_clock.clone())
        .with_sd_card_root(sd_card_root)
        .with_config(machine_profile.process_build_config())
        .with_cpu_backend(cpu_backend);
    let process = process_builder
        .build(&plan)
        .map_err(|error| error.to_string())?;
    log::debug!("process prepared in {:?}", process_started.elapsed());
    log::info!(
        "process ready: entry={:#018x}, modules={}",
        process.entry_module().entry_address(),
        process.modules().len()
    );
    log::info!("starting CPU backend {}", process.cpu_backend_name());

    let initial_operation_mode = match config.system.initial_operation_mode {
        InitialOperationMode::Handheld => OperationMode::Handheld,
        InitialOperationMode::Docked => OperationMode::Console,
    };
    log::debug!("initial operation mode: {initial_operation_mode:?}");
    let time_environment = TimeEnvironment::new(virtual_clock, &config.system.time.timezone)
        .map_err(|error| format!("cannot create Horizon time environment: {error}"))?;
    let settings_environment = SettingsEnvironment::for_language(
        config
            .system
            .preferred_languages
            .first()
            .copied()
            .map(system_language)
            .unwrap_or(SystemLanguage::AmericanEnglish),
    );
    let horizon_environment = HorizonEnvironment {
        operation_mode: initial_operation_mode,
        time: time_environment,
        settings: settings_environment,
    };
    log::debug!(
        "virtual time: mode={clock_mode:?}, timezone={}",
        config.system.time.timezone
    );
    let gpu_backend = initialize_backend(
        BackendInstanceId::new(1),
        NonCpuDeviceId::new(1),
        WgpuBackendConfiguration {
            cache: config.gpu,
            ..WgpuBackendConfiguration::default()
        },
    )
    .map_err(|error| format!("cannot initialize accelerated GPU backend: {error}"))?;
    log::info!(
        "GPU backend initialized: api={:?} adapter={} driver={}",
        gpu_backend.adapter.backend,
        gpu_backend.adapter.name,
        gpu_backend.adapter.driver
    );
    let presentation_context = gpu_backend.presentation_context();
    let video_system =
        VideoSystem::with_gpu_backend(presenter, gpu_backend.into_runtime(), config.gpu);
    let gamepad_profiles = GamepadProfiles::new(config.input.profiles.clone());

    let Some(frontend) = frontend else {
        return finish_execution(execute_worker(
            coordinator,
            process,
            horizon_environment,
            video_system,
            gamepad_profiles,
            trace_interpreter,
        ));
    };
    let frontend = frontend.with_gpu_context(presentation_context);

    let worker_control =
        frontend_control.expect("window frontend construction provides its control channel");
    let worker = thread::Builder::new()
        .name("nixe-guest".to_owned())
        .spawn(move || {
            let _completion = WorkerCompletion(worker_control);
            execute_worker(
                coordinator,
                process,
                horizon_environment,
                video_system,
                gamepad_profiles,
                trace_interpreter,
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

const fn system_language(language: NacpLanguage) -> SystemLanguage {
    match language {
        NacpLanguage::AmericanEnglish => SystemLanguage::AmericanEnglish,
        NacpLanguage::BritishEnglish => SystemLanguage::BritishEnglish,
        NacpLanguage::Japanese => SystemLanguage::Japanese,
        NacpLanguage::French => SystemLanguage::French,
        NacpLanguage::German => SystemLanguage::German,
        NacpLanguage::LatinAmericanSpanish => SystemLanguage::LatinAmericanSpanish,
        NacpLanguage::Spanish => SystemLanguage::Spanish,
        NacpLanguage::Italian => SystemLanguage::Italian,
        NacpLanguage::Dutch => SystemLanguage::Dutch,
        NacpLanguage::CanadianFrench => SystemLanguage::CanadianFrench,
        NacpLanguage::Portuguese => SystemLanguage::Portuguese,
        NacpLanguage::Russian => SystemLanguage::Russian,
        NacpLanguage::Korean => SystemLanguage::Korean,
        NacpLanguage::TraditionalChinese => SystemLanguage::TraditionalChinese,
        NacpLanguage::SimplifiedChinese => SystemLanguage::SimplifiedChinese,
        NacpLanguage::BrazilianPortuguese => SystemLanguage::BrazilianPortuguese,
    }
}

fn effective_cpu_configuration(
    configuration: CpuConfig,
    backend_override: Option<CpuBackendSelection>,
) -> CpuConfig {
    CpuConfig {
        backend: match backend_override {
            Some(backend) => backend,
            None => configuration.backend,
        },
        ..configuration
    }
}

fn select_cpu_backend(
    configuration: CpuConfig,
    title_name: &str,
) -> Result<CpuBackendConfig, String> {
    Ok(match configuration.backend {
        CpuBackendSelection::Jit => CpuBackendConfig::Jit(
            JitConfiguration::default()
                .with_dump_directory(configuration.jit.dump_directory)
                .with_performance_report_directory(configuration.jit.performance_report_directory)
                .with_performance_report_title(title_name),
        ),
        CpuBackendSelection::Interpreter => CpuBackendConfig::Interpreter,
    })
}

#[cfg(test)]
mod backend_selection_tests {
    use super::*;

    #[test]
    fn cli_backend_override_has_priority_without_replacing_other_cpu_policy() {
        let configured = CpuConfig {
            backend: CpuBackendSelection::Interpreter,
            parallel_vcpus: true,
            jit: nixe_config::CpuJitConfig {
                dump_directory: Some("jit-diagnostics".into()),
                performance_report_directory: Some("jit-performance".into()),
            },
        };

        assert_eq!(
            effective_cpu_configuration(configured.clone(), None),
            configured
        );
        assert_eq!(
            effective_cpu_configuration(configured.clone(), Some(CpuBackendSelection::Jit)),
            CpuConfig {
                backend: CpuBackendSelection::Jit,
                ..configured.clone()
            }
        );
    }

    #[test]
    fn application_composition_selects_one_concrete_backend() {
        let interpreter = select_cpu_backend(
            CpuConfig {
                backend: CpuBackendSelection::Interpreter,
                parallel_vcpus: true,
                ..CpuConfig::default()
            },
            "test-title",
        )
        .unwrap();
        assert!(matches!(interpreter, CpuBackendConfig::Interpreter));

        let jit = select_cpu_backend(CpuConfig::default(), "test-title").unwrap();
        assert!(matches!(jit, CpuBackendConfig::Jit(_)));
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

struct HorizonEnvironment {
    operation_mode: OperationMode,
    time: TimeEnvironment,
    settings: SettingsEnvironment,
}

fn execute_worker(
    mut coordinator: RuntimeCoordinator,
    process: RunnableProcess,
    horizon_environment: HorizonEnvironment,
    video_system: VideoSystem,
    gamepad_profiles: GamepadProfiles,
    trace_interpreter: bool,
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
                horizon_environment,
                execution_video,
                &mut input,
                trace_interpreter,
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
    let exit_code = result
        .teardown
        .exit
        .as_ref()
        .map_or(0, |exit| exit.exit_code);
    let exit_cause = result.teardown.exit.as_ref().map_or_else(
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
    horizon_environment: HorizonEnvironment,
    video_system: VideoSystem,
    input: &mut InputManager<SdlInputBackend>,
    trace_interpreter: bool,
) -> Result<ExecutionSummary, String> {
    let coordinator = &mut *scheduled.coordinator;
    let process_id = scheduled.process_id;
    let mut dispatcher = HorizonSvcDispatcher::new_with_video_and_settings(
        horizon_environment.operation_mode,
        horizon_environment.time,
        horizon_environment.settings,
        video_system,
    );
    let mut instructions = 0_u64;
    let execution_started = Instant::now();
    let mut next_progress = EXECUTION_PROGRESS_INTERVAL;
    let mut last_progress_instructions = 0_u64;
    let mut last_progress_elapsed = Duration::ZERO;
    let mut rejected = BTreeSet::new();
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
        let executions = match coordinator.execution_mode() {
            VcpuExecutionMode::Deterministic => coordinator
                .run_next_adaptive()
                .map(|execution| execution.into_iter().collect()),
            VcpuExecutionMode::Parallel => coordinator.run_parallel_wave_adaptive(),
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
            let report = execution.report;
            instructions = instructions.saturating_add(report.instructions_executed);
            if trace_interpreter {
                log::trace!(
                    "interpreter slice completed: process={:?} thread={:?} vcpu={:?} generation={:?} {report}",
                    execution.lease.process,
                    execution.lease.thread,
                    execution.lease.vcpu,
                    execution.lease.generation,
                );
            }
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
            match &report.stop {
                ExecutionStop::BudgetExhausted
                | ExecutionStop::Safepoint
                | ExecutionStop::PendingEvent { .. } => {}
                ExecutionStop::Scheduled { .. } => {}
                ExecutionStop::SupervisorCall { .. } => {
                    if trace_interpreter {
                        log::trace!(
                            "interpreter SVC dispatch started: process={:?} thread={:?} vcpu={:?} stop=[{}]",
                            execution.lease.process,
                            execution.lease.thread,
                            execution.lease.vcpu,
                            report.stop,
                        );
                    }
                    let handling = dispatcher
                        .route_scheduled_supervisor_call(coordinator, execution.lease, &report.stop)
                        .map_err(|error| error.to_string())?;
                    if trace_interpreter {
                        let outcome = match &handling {
                            ExceptionHandlingResult::Resumed => "resumed",
                            ExceptionHandlingResult::Suspended => "suspended",
                            ExceptionHandlingResult::Rejected(_) => "rejected",
                            ExceptionHandlingResult::Terminated { .. } => "terminated",
                            ExceptionHandlingResult::Fault(_) => "fault",
                        };
                        log::trace!(
                            "interpreter SVC dispatch completed: process={:?} thread={:?} vcpu={:?} outcome={outcome}",
                            execution.lease.process,
                            execution.lease.thread,
                            execution.lease.vcpu,
                        );
                    }
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
    let HorizonSvcFault::Ipc { fault, .. } = fault else {
        return;
    };
    let Some(UnsupportedNvDrvOperation::ScheduledGpfifoSubmission { boundary, .. }) =
        fault.unsupported_nvdrv()
    else {
        return;
    };
    if boundary.frontend_failure().is_none() {
        return;
    }
    let diagnostic = match boundary.frontend_diagnostic() {
        Ok(Some(diagnostic)) => diagnostic,
        Ok(None) => return,
        Err(error) => {
            log::warn!("cannot reconstruct failed Maxwell pushbuffer: {error}");
            return;
        }
    };
    let directory = PathBuf::from(MAXWELL_PUSHBUFFER_DUMP_DIRECTORY);
    let path = directory.join(MAXWELL_PUSHBUFFER_DUMP_FILENAME);

    let result = (|| -> io::Result<()> {
        std::fs::create_dir_all(&directory)?;
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        write_nv_push_dump_words(&mut writer, diagnostic.words().iter().copied())?;
        writer.flush()
    })();

    match result {
        Ok(()) => log::info!(
            "Maxwell pushbuffer dumped for nv_push_dump: path={} words={} total={} complete={}",
            path.display(),
            diagnostic.words().len(),
            diagnostic.total_words(),
            diagnostic.is_complete(),
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
        ProcessExitCause::GuestBreak {
            reason,
            info,
            size,
            payload,
        } => {
            let payload = payload.map_or_else(String::new, |payload| {
                let bytes = payload.as_bytes();
                let encoded = bytes
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>();
                if let Ok(bytes) = <[u8; 4]>::try_from(bytes) {
                    let result =
                        nixe_horizon::HorizonIpcResult::from_raw(u32::from_le_bytes(bytes));
                    format!(
                        ", payload=0x{encoded}, result={:#x} (module={}, description={})",
                        result.raw(),
                        result.module(),
                        result.description()
                    )
                } else {
                    format!(", payload=0x{encoded}")
                }
            });
            let source = exit
                .source
                .map_or_else(|| "unknown".to_owned(), |source| source.to_string());
            let registers = exit
                .context
                .as_ref()
                .map_or_else(|| "unavailable".to_owned(), |context| context.to_string());
            let frames = exit
                .frames
                .iter()
                .map(|frame| {
                    format!(
                        "fp=0x{:016x} lr=0x{:016x}",
                        frame.frame_pointer, frame.return_address
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            Err(format!(
                "guest requested a fatal break: reason={reason:#x}, info={info:#x}, size={size:#x}{payload}, source=[{source}], thread={}, registers=[{registers}], frames=[{frames}], code={:#x}",
                exit.thread_id, exit.exit_code
            ))
        }
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
            context: None,
            frames: Box::new([]),
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
        let mut exit = process_exit(
            ProcessExitCause::GuestBreak {
                reason: 0,
                info: 0x1234,
                size: 4,
                payload: None,
            },
            0,
        );
        let source = nixe_cpu::location::LocationDescriptor::new(
            nixe_memory::GuestVirtualAddress::new(0x7525_1264),
            nixe_cpu::profile::CpuProfileId::new(1),
        );
        let mut x = [0; nixe_cpu::state::a64::GENERAL_REGISTER_COUNT];
        x[30] = 0x7522_7af8;
        exit.source = Some(source);
        exit.thread_id = 7;
        exit.context = Some(Box::new(nixe_cpu::state::RegisterContext {
            x,
            sp: 0x1076_0ffec0,
            pc: source.pc,
            nzcv: nixe_cpu::state::Nzcv::from_bits(nixe_cpu::state::Nzcv::C),
        }));
        exit.frames = Box::new([nixe_runtime::GuestStackFrame {
            frame_pointer: 0x1076_0ffcf0,
            return_address: 0x7518_7c14,
        }]);

        let error = classify_exit(Some(exit)).unwrap_err();
        assert!(error.contains("fatal break"));
        assert!(error.contains("info=0x1234"));
        assert!(
            error.contains("source=[pc=0x0000000075251264 profile=0x0000000000000001], thread=7")
        );
        assert!(error.contains("x30=0x0000000075227af8"));
        assert!(error.contains("sp=0x00000010760ffec0"));
        assert!(error.contains("pc=0x0000000075251264"));
        assert!(error.contains("flags=N0Z0C1V0"));
        assert!(error.contains("fp=0x00000010760ffcf0 lr=0x0000000075187c14"));
    }

    #[test]
    fn fatal_break_decodes_a_four_byte_horizon_result_payload() {
        let payload = nixe_runtime::GuestBreakPayload::new(&0x60a_u32.to_le_bytes()).unwrap();
        let error = classify_exit(Some(process_exit(
            ProcessExitCause::GuestBreak {
                reason: 0,
                info: 0x1234,
                size: 4,
                payload: Some(payload),
            },
            0,
        )))
        .unwrap_err();

        assert!(error.contains("payload=0x0a060000"));
        assert!(error.contains("result=0x60a (module=10, description=3)"));
    }
}
