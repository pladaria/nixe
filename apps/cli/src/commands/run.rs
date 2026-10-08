mod display;
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
    CpuBackendSelection, CpuConfig, DiagnosticsConfig, GuestLogsLevel, InitialOperationMode,
    TimeMode, WindowState,
};
use nixe_gpu::BackendInstanceId;
use nixe_gpu_wgpu::{WgpuBackendConfiguration, initialize_backend};
use nixe_horizon::{
    GuestLogLevel, HorizonDiagnostics, HorizonSvcDispatcher, HorizonSvcFault, OperationMode,
    SaveDataSystem, SettingsEnvironment, SystemLanguage, TimeEnvironment,
    UnsupportedNvDrvOperation, VideoSystem, switch_1_machine_profile,
};
use nixe_input::{
    ControllerId, EmulatedButtonState, GamepadProfiles, InputReader, InputWorker,
    ProfiledControllerState, TouchScreenReader,
};
use nixe_loader_title::{NacpLanguage, SupportedLanguages, UserAccountSwitchLock};
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

const EXECUTION_RATE_COMPLETIONS: u64 = 1024;
const EXECUTION_RATE_LOG_INTERVAL: Duration = Duration::from_secs(5);
const MAXWELL_PUSHBUFFER_DUMP_DIRECTORY: &str = "dump";
const MAXWELL_PUSHBUFFER_DUMP_FILENAME: &str = "pushbuffer.bin";

pub struct Arguments {
    pub config_path: Option<PathBuf>,
    pub log_level_override: Option<LogLevel>,
    pub identifier: String,
    pub headless: bool,
    pub cpu_backend_override: Option<CpuBackendSelection>,
    pub guest_logs_level_override: Option<GuestLogsLevel>,
    pub file_system_access_log_override: Option<bool>,
}

pub fn run(arguments: Arguments) -> Result<(), String> {
    let Arguments {
        config_path,
        log_level_override,
        identifier,
        headless,
        cpu_backend_override,
        guest_logs_level_override,
        file_system_access_log_override,
    } = arguments;
    let frontend_stop_requested = Arc::new(AtomicBool::new(false));
    let (frontend, frontend_control, presenter, touch_screen) = if headless {
        log::info!("headless presentation enabled; no host window will be created");
        (None, None, None, None)
    } else {
        let mut frontend = WindowFrontend::new(Arc::clone(&frontend_stop_requested))
            .map_err(|error| error.to_string())?;
        let control = frontend.control();
        let mailbox = frontend.mailbox();
        let touch_screen = frontend.take_touch_screen();
        (
            Some(frontend),
            Some(control),
            Some(mailbox),
            Some(touch_screen),
        )
    };
    let machine_profile = switch_1_machine_profile();
    let scheduler_profile = machine_profile.scheduler().clone();
    let config = load_config(config_path, log_level_override)?;
    let cpu_configuration = effective_cpu_configuration(config.cpu.clone(), cpu_backend_override);
    let diagnostics_configuration = effective_diagnostics_configuration(
        config.diagnostics,
        guest_logs_level_override,
        file_system_access_log_override,
    );
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
    let warmup_profile = cpu_configuration.warmup_profile;
    let cpu_backend = select_cpu_backend(cpu_configuration)?;
    let trace_interpreter = matches!(cpu_backend, CpuBackendConfig::Interpreter)
        && log::log_enabled!(log::Level::Trace);
    let mut process_builder = ProcessBuilder::new()
        .with_virtual_clock(virtual_clock.clone())
        .with_sd_card_root(sd_card_root)
        .with_config(machine_profile.process_build_config())
        .with_cpu_backend(cpu_backend);
    if warmup_profile
        && let Some(directory) = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
    {
        process_builder =
            process_builder.with_jit_warmup_directory(directory.join("nixe").join("cpu"));
    }
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
        application_language: plan.control_metadata().and_then(|control| {
            desired_application_language(
                control.supported_languages(),
                &config.system.preferred_languages,
            )
        }),
        user_account_switch_locked: plan.control_metadata().and_then(|control| {
            match control.nacp.user_account_switch_lock {
                UserAccountSwitchLock::Disable => Some(false),
                UserAccountSwitchLock::Enable => Some(true),
                UserAccountSwitchLock::Unknown(_) => None,
            }
        }),
        save_data: plan.packaged_identity().zip(plan.control_metadata()).map(
            |(identity, control)| {
                SaveDataSystem::new(
                    config.filesystem.save_data.clone(),
                    identity.application_id().get(),
                    &control.nacp,
                )
            },
        ),
        diagnostics: horizon_diagnostics(diagnostics_configuration),
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
    // SDL initialization and final subsystem shutdown remain on the main thread.
    // The owners outlive guest execution in both windowed and headless modes.
    let sdl = sdl3::init().map_err(|error| format!("cannot initialize SDL: {error}"))?;
    let audio = nixe_audio::HostAudioRuntime::new(&sdl, config.audio.output)
        .map_err(|error| error.to_string())?;
    let input_events = coordinator.event_sender();
    let input_owner = InputWorker::with_profiles(&sdl, gamepad_profiles, move || {
        input_events.notify_host_service(nixe_runtime::ExternalEventSource::Input);
    })
    .map_err(|error| format!("cannot start input worker: {error}"))?;
    let input = input_owner.reader();
    let host_input = HostInputReaders {
        controller: input,
        touch_screen,
        vibration: input_owner
            .vibration_output()
            .expect("SDL input owns actuator output"),
    };
    let audio_backend = audio.backend();

    let Some(frontend) = frontend else {
        return finish_execution(execute_worker(
            coordinator,
            process,
            horizon_environment,
            video_system,
            host_input,
            audio_backend,
            trace_interpreter,
        ));
    };
    let window_state_path = config.window_state_path();
    let saved_window_state = match WindowState::load(&window_state_path) {
        Ok(Some(state)) => {
            log::info!("read window configuration {}", window_state_path.display());
            Some(state)
        }
        Ok(None) => {
            log::info!(
                "window configuration {} does not exist; using defaults",
                window_state_path.display()
            );
            None
        }
        Err(error) => {
            log::warn!(
                "cannot use window configuration {}: {error}; discarding its contents and using defaults",
                window_state_path.display()
            );
            None
        }
    };
    let frontend = frontend
        .with_gpu_context(presentation_context)
        .with_window_state(saved_window_state)
        .with_screenshots(title.name.clone(), PathBuf::from("dump/screenshots/nixe"));

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
                host_input,
                audio_backend,
                trace_interpreter,
            )
        })
        .map_err(|error| format!("cannot start guest execution worker: {error}"))?;

    let frontend_result = frontend.run().map_err(|error| error.to_string());
    if let Ok(Some(state)) = &frontend_result {
        match state.save(&window_state_path) {
            Ok(()) => log::info!("wrote window configuration {}", window_state_path.display()),
            Err(error) => log::warn!(
                "cannot write window configuration {}: {error}; discarding unsaved window state",
                window_state_path.display()
            ),
        }
    }
    frontend_stop_requested.store(true, Ordering::Release);
    let _ = external_events.submit(nixe_runtime::ExternalEvent::HostStop);
    let worker_result = worker
        .join()
        .map_err(|_| "guest execution worker panicked".to_owned())?;
    let execution_result = finish_execution(worker_result);
    frontend_result.map(|_| ()).and(execution_result)
}

fn desired_application_language(
    supported: SupportedLanguages,
    preferences: &[NacpLanguage],
) -> Option<SystemLanguage> {
    // The configured preference order is frontend policy. AM receives a
    // language the title actually declares; settings retain the system locale.
    // https://switchbrew.org/wiki/NACP#Structure
    preferences
        .iter()
        .copied()
        .chain([NacpLanguage::AmericanEnglish])
        .chain(NacpLanguage::ALL)
        .find(|language| supported.contains(*language))
        .map(system_language)
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

const fn effective_diagnostics_configuration(
    configuration: DiagnosticsConfig,
    guest_logs_level_override: Option<GuestLogsLevel>,
    file_system_access_log_override: Option<bool>,
) -> DiagnosticsConfig {
    DiagnosticsConfig {
        log_level: configuration.log_level,
        guest_logs_level: match guest_logs_level_override {
            Some(level) => level,
            None => configuration.guest_logs_level,
        },
        file_system_access_log: match file_system_access_log_override {
            Some(enabled) => enabled,
            None => configuration.file_system_access_log,
        },
    }
}

const fn horizon_diagnostics(configuration: DiagnosticsConfig) -> HorizonDiagnostics {
    let guest_logs_level = match configuration.guest_logs_level {
        GuestLogsLevel::Inherit => GuestLogLevel::Inherit,
        GuestLogsLevel::Trace => GuestLogLevel::Trace,
        GuestLogsLevel::Debug => GuestLogLevel::Debug,
        GuestLogsLevel::Info => GuestLogLevel::Info,
        GuestLogsLevel::Warn => GuestLogLevel::Warn,
        GuestLogsLevel::Error => GuestLogLevel::Error,
        GuestLogsLevel::Off => GuestLogLevel::Off,
    };
    HorizonDiagnostics::new(guest_logs_level, configuration.file_system_access_log)
}

fn select_cpu_backend(configuration: CpuConfig) -> Result<CpuBackendConfig, String> {
    Ok(match configuration.backend {
        CpuBackendSelection::Jit => CpuBackendConfig::Jit,
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
            warmup_profile: false,
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
        let interpreter = select_cpu_backend(CpuConfig {
            backend: CpuBackendSelection::Interpreter,
            parallel_vcpus: true,
            warmup_profile: false,
        })
        .unwrap();
        assert!(matches!(interpreter, CpuBackendConfig::Interpreter));

        let jit = select_cpu_backend(CpuConfig::default()).unwrap();
        assert!(matches!(jit, CpuBackendConfig::Jit));
    }

    #[test]
    fn cli_guest_diagnostics_override_only_the_selected_policy() {
        let configured = DiagnosticsConfig {
            log_level: nixe_config::DiagnosticLogLevel::Info,
            guest_logs_level: GuestLogsLevel::Warn,
            file_system_access_log: true,
        };
        assert_eq!(
            effective_diagnostics_configuration(configured, Some(GuestLogsLevel::Off), None),
            DiagnosticsConfig {
                guest_logs_level: GuestLogsLevel::Off,
                ..configured
            }
        );
        assert_eq!(
            effective_diagnostics_configuration(configured, None, Some(false)),
            DiagnosticsConfig {
                file_system_access_log: false,
                ..configured
            }
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
    teardown: Result<ProcessTeardownReport, String>,
    graphics_teardown: nixe_horizon::GraphicsTeardownReport,
}

struct HorizonEnvironment {
    operation_mode: OperationMode,
    time: TimeEnvironment,
    settings: SettingsEnvironment,
    application_language: Option<SystemLanguage>,
    user_account_switch_locked: Option<bool>,
    save_data: Option<SaveDataSystem>,
    diagnostics: HorizonDiagnostics,
}

struct HostInputReaders {
    controller: InputReader<Option<ProfiledControllerState>>,
    touch_screen: Option<TouchScreenReader>,
    vibration: nixe_input::VibrationOutput,
}

fn execute_worker(
    mut coordinator: RuntimeCoordinator,
    process: RunnableProcess,
    horizon_environment: HorizonEnvironment,
    video_system: VideoSystem,
    mut host_input: HostInputReaders,
    audio_backend: Arc<dyn nixe_audio::AudioBackend>,
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
    let mut scheduled = ScheduledProcess {
        coordinator: &mut coordinator,
        process_id,
    };
    let mut dispatcher = HorizonSvcDispatcher::new_with_video_and_settings(
        horizon_environment.operation_mode,
        horizon_environment.time,
        horizon_environment.settings,
        execution_video,
    )
    .with_diagnostics(horizon_environment.diagnostics)
    .with_audio_backend(audio_backend)
    .with_vibration_output(host_input.vibration.clone());
    if let Some(save_data) = horizon_environment.save_data {
        dispatcher = dispatcher.with_save_data(save_data);
    }
    if let Some(locked) = horizon_environment.user_account_switch_locked {
        dispatcher = dispatcher.with_user_account_switch_lock(locked);
    }
    if let Some(language) = horizon_environment.application_language {
        dispatcher = dispatcher.with_application_language(language);
    }
    let mut execution = execute(
        &mut scheduled,
        &mut dispatcher,
        &mut host_input,
        trace_interpreter,
    );
    log::debug!(
        "guest execution stopped after {:?}",
        execution_started.elapsed()
    );
    if let Err(error) = coordinator.quiesce() {
        execution = Err(match execution {
            Ok(_) => error.to_string(),
            Err(original) => format!("{original}; cannot quiesce CPU workers: {error}"),
        });
    }
    // Join/cancel frontend and storage jobs while CPU admission and canonical
    // mappings still exist, then drain the backend before removing the process.
    drop(dispatcher);
    // Stop and join GPU work while its
    // canonical memory transitions can still use the JIT coordinator. Removing
    // the process closes native admission and would reject those transitions.
    let graphics_teardown = video_system.teardown();
    let process = match coordinator.remove_process(process_id) {
        Ok(process) => process,
        Err(error) => {
            // Stopping the backend can report an asynchronous compiler failure
            // even after every scheduler lease has been returned. Preserve it
            // and let the coordinator's Drop stop/join its remaining workers.
            return WorkerResult {
                execution,
                teardown: Err(format!("cannot remove process during teardown: {error}")),
                graphics_teardown,
            };
        }
    };
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
        teardown: Ok(teardown),
        graphics_teardown,
    }
}

fn finish_execution(result: WorkerResult) -> Result<(), String> {
    let teardown = result
        .teardown
        .map_err(|diagnostic| match &result.execution {
            Ok(_) => diagnostic,
            Err(error) => format!("{error}; {diagnostic}"),
        })?;
    log::debug!(
        "resources released: handles={}, address_waiters={}, layers={}, queues={}, \
         pending_frames={}, nvdrv_fds={}, nvmap_allocations={}",
        teardown.handles_released,
        teardown.address_waiters_released,
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
    let exit_code = teardown.exit.as_ref().map_or(0, |exit| exit.exit_code);
    let exit_cause = teardown.exit.as_ref().map_or_else(
        || "without an exit record".to_owned(),
        |exit| format!("{:?}", exit.cause),
    );
    log::info!(
        "execution finished: SVC calls={}, rejected SVC kinds={}, cause={}, code={:#x}",
        summary.svc_calls,
        summary.rejected_svc_kinds,
        exit_cause,
        exit_code
    );
    if let Some(exit) = teardown.exit.as_ref() {
        log::debug!("guest exit context: {}", exit_context(exit));
    }
    classify_exit(teardown.exit)
}

struct ExecutionSummary {
    svc_calls: u64,
    rejected_svc_kinds: usize,
}

struct ScheduledProcess<'a> {
    coordinator: &'a mut RuntimeCoordinator,
    process_id: ProcessId,
}

fn execute(
    scheduled: &mut ScheduledProcess<'_>,
    dispatcher: &mut HorizonSvcDispatcher,
    host_input: &mut HostInputReaders,
    trace_interpreter: bool,
) -> Result<ExecutionSummary, String> {
    let coordinator = &mut *scheduled.coordinator;
    let process_id = scheduled.process_id;
    let execution_started = Instant::now();
    let display_clock = display::DisplayClock::start(dispatcher.video_system(), execution_started)?;
    let execution_rate_enabled = log::log_enabled!(log::Level::Info);
    let mut execution_completions = 0_u64;
    let mut last_rate_completions = 0_u64;
    let mut last_rate_elapsed = Duration::ZERO;
    let mut rejected = BTreeSet::new();
    let mut last_input_sample: Option<Instant> = None;
    let mut active_input = None;
    let mut input_observed = false;
    let mut active_buttons = EmulatedButtonState::default();
    loop {
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
            return Ok(execution_summary(dispatcher, rejected.len()));
        }
        let elapsed = execution_started.elapsed();
        display_clock.require_healthy()?;
        dispatcher.require_graphics_healthy().map_err(|error| {
            dump_maxwell_pushbuffer_on_ipc_fault(&error);
            error.to_string()
        })?;
        let input_trace = nixe_trace::Span::new("host.input", 0, 0);
        if let Some(sample) = host_input
            .controller
            .take_latest()
            .map_err(|error| error.to_string())?
        {
            // The input worker owns the only sampling timer. Publish each
            // consumed sample once, using capture time rather than host delay.
            report_input_change(&mut active_input, &sample.state, input_observed);
            input_observed = true;
            let profiled = sample.state.as_ref();
            let current_buttons = profiled
                .map_or_else(EmulatedButtonState::default, |controller| {
                    controller.state.buttons
                });
            if log::log_enabled!(log::Level::Debug) {
                for (button, pressed) in button_transitions(active_buttons, current_buttons) {
                    log::debug!(
                        "emulated button {button} {}: elapsed={elapsed:?}",
                        if pressed { "pressed" } else { "released" }
                    );
                }
            }
            active_buttons = current_buttons;
            let delta = last_input_sample.map_or(Duration::ZERO, |previous| {
                sample.captured_at.saturating_duration_since(previous)
            });
            host_input
                .vibration
                .select_controller(profiled.map(|controller| controller.controller_id))
                .map_err(|error| format!("cannot route controller vibration: {error}"))?;
            dispatcher
                .advance_input(profiled.map(|controller| &controller.state), delta)
                .map_err(|error| format!("cannot publish Horizon HID state: {error}"))?;
            let touch_state = host_input
                .touch_screen
                .as_mut()
                .map_or_else(Default::default, TouchScreenReader::sample);
            dispatcher
                .advance_touch_screen(&touch_state, delta)
                .map_err(|error| format!("cannot publish Horizon touch-screen state: {error}"))?;
            last_input_sample = Some(sample.captured_at);
        }
        drop(input_trace);
        let scheduler_trace = nixe_trace::Span::new("coordinator.reconcile", 0, 0);
        let execution = match coordinator.execution_mode() {
            VcpuExecutionMode::Deterministic => coordinator.run_next_adaptive(),
            VcpuExecutionMode::Parallel => coordinator.run_parallel_adaptive(),
        }
        .map_err(|error| error.to_string())?;
        drop(scheduler_trace);
        if execution.is_none() {
            let host_wait = Duration::from_millis(100);
            let _trace = nixe_trace::Span::new("coordinator.idle", 0, host_wait.as_nanos() as u64);
            coordinator
                .wait_for_external_event_for(host_wait)
                .map_err(|error| error.to_string())?;
            nixe_trace::event("coordinator.wake", 0, 0);
            continue;
        }
        if let Some(execution) = execution {
            let report = execution.report;
            if execution_rate_enabled {
                execution_completions = execution_completions.wrapping_add(1);
                if execution_completions.is_multiple_of(EXECUTION_RATE_COMPLETIONS) {
                    let elapsed = execution_started.elapsed();
                    let interval = elapsed.saturating_sub(last_rate_elapsed);
                    if interval >= EXECUTION_RATE_LOG_INTERVAL {
                        let interval_completions =
                            execution_completions.wrapping_sub(last_rate_completions);
                        let completions_per_second =
                            interval_completions as f64 / interval.as_secs_f64();
                        log::info!(
                            "guest CPU completion rate: completions_per_second={completions_per_second:.0}, completions={execution_completions}, elapsed={elapsed:?}"
                        );
                        last_rate_completions = execution_completions;
                        last_rate_elapsed = elapsed;
                    }
                }
            }
            if trace_interpreter {
                log::trace!(
                    "interpreter slice completed: process={:?} thread={:?} vcpu={:?} generation={:?} {report}",
                    execution.lease.process,
                    execution.lease.thread,
                    execution.lease.vcpu,
                    execution.lease.generation,
                );
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
                            // ExitThread can leave the rest of the process runnable.
                            // Only the runtime's process lifecycle ends this launch.
                            if coordinator.process(process_id).is_some_and(|process| {
                                process.lifecycle() == nixe_scheduler::ProcessLifecycle::Exited
                            }) {
                                return Ok(execution_summary(dispatcher, rejected.len()));
                            }
                        }
                        ExceptionHandlingResult::Suspended => {}
                        ExceptionHandlingResult::Fault(error) => {
                            dump_maxwell_pushbuffer_on_fault(&error);
                            return Err(format!("Horizon SVC dispatch failed: {error}; {report}"));
                        }
                    }
                }
                ExecutionStop::LoaderReturn { .. } => {
                    return Ok(execution_summary(dispatcher, rejected.len()));
                }
                stop => return Err(execution_stop_error(stop, &report)),
            }
        }
    }
}

fn dump_maxwell_pushbuffer_on_fault(fault: &HorizonSvcFault) {
    let HorizonSvcFault::Ipc { fault, .. } = fault else {
        return;
    };
    dump_maxwell_pushbuffer_on_ipc_fault(fault);
}

fn dump_maxwell_pushbuffer_on_ipc_fault(fault: &nixe_horizon::HorizonIpcFault) {
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
    device: std::sync::Arc<str>,
    profile_name: std::sync::Arc<str>,
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
        None => log::warn!("no matching first-gamepad input profile; player one is disconnected"),
    }
    *active = next;
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

fn exit_context(exit: &ProcessExit) -> String {
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
    format!(
        "source=[{source}], thread={}, registers=[{registers}], frames=[{frames}]",
        exit.thread_id
    )
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
            let context = exit_context(&exit);
            Err(format!(
                "guest requested a fatal break: reason={reason:#x}, info={info:#x}, size={size:#x}{payload}, {context}, code={:#x}",
                exit.exit_code
            ))
        }
    }
}

fn execution_summary(
    dispatcher: &HorizonSvcDispatcher,
    rejected_svc_kinds: usize,
) -> ExecutionSummary {
    ExecutionSummary {
        svc_calls: dispatcher.coverage().iter().map(|entry| entry.calls).sum(),
        rejected_svc_kinds,
    }
}

fn execution_stop_error(stop: &ExecutionStop, report: &nixe_runtime::ExecutionReport) -> String {
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
    format!("{reason}; diagnostic: {report}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn application_language_respects_preferences_and_declared_support() {
        let supported = SupportedLanguages::from_raw(
            NacpLanguage::Spanish.bit() | NacpLanguage::AmericanEnglish.bit(),
        );
        assert_eq!(
            desired_application_language(
                supported,
                &[NacpLanguage::Japanese, NacpLanguage::Spanish]
            ),
            Some(SystemLanguage::Spanish)
        );
        assert_eq!(
            desired_application_language(
                supported,
                &[NacpLanguage::Spanish, NacpLanguage::AmericanEnglish]
            ),
            Some(SystemLanguage::Spanish)
        );
        assert_eq!(
            desired_application_language(supported, &[NacpLanguage::Japanese]),
            Some(SystemLanguage::AmericanEnglish)
        );
        assert_eq!(
            desired_application_language(
                SupportedLanguages::from_raw(NacpLanguage::Japanese.bit()),
                &[]
            ),
            Some(SystemLanguage::Japanese)
        );
        assert_eq!(
            desired_application_language(SupportedLanguages::default(), &[]),
            None
        );
        assert_eq!(
            desired_application_language(SupportedLanguages::from_raw(1 << 31), &[]),
            None
        );
    }

    #[test]
    fn teardown_failure_preserves_execution_error_without_panicking() {
        for execution in [
            Ok(ExecutionSummary {
                svc_calls: 0,
                rejected_svc_kinds: 0,
            }),
            Err("HCQ Cranelift: invalid checkpoint".to_owned()),
        ] {
            let failed = execution.is_err();
            let error = finish_execution(WorkerResult {
                execution,
                teardown: Err("cannot remove process during teardown: backend failure".into()),
                graphics_teardown: Default::default(),
            })
            .unwrap_err();
            assert!(error.contains("cannot remove process during teardown: backend failure"));
            assert_eq!(error.contains("HCQ Cranelift: invalid checkpoint"), failed);
        }
    }

    #[test]
    fn runnable_process_can_be_owned_by_the_guest_worker() {
        fn assert_send<T: Send>() {}

        assert_send::<RunnableProcess>();
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
