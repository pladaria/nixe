use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use nixe_cli::library::{Library, LibraryTitleSource};
use nixe_config::InitialOperationMode;
use nixe_horizon::{HorizonSvcDispatcher, OperationMode};
use nixe_runtime::{
    DiagnosticsPolicy, ExceptionHandlingResult, ExecutionStop, Launcher, LauncherInput,
    ProcessBuilder, ProcessExit, ProcessExitCause, RunnableProcess,
};

use super::load_config;

const EXECUTION_SLICE_INSTRUCTIONS: u64 = 100_000;
const EXECUTION_PROGRESS_INTERVAL: u64 = 10_000_000;

pub struct Arguments {
    pub config_path: Option<PathBuf>,
    pub identifier: String,
}

pub fn run(arguments: Arguments) -> Result<(), String> {
    let interrupted = install_interrupt_handler()?;
    log::info!("scanning configured title library");
    let config = load_config(arguments.config_path)?;
    let scan_started = Instant::now();
    let library = Library::scan(&config)?;
    log::debug!(
        "configured title library scanned in {:?}",
        scan_started.elapsed()
    );
    let title = library
        .find(&arguments.identifier)
        .ok_or_else(|| format!("unknown title ID: {}", arguments.identifier))?;
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
    let mut diagnostics = DiagnosticsPolicy::from(config.diagnostics);
    let instruction_trace = log::log_enabled!(log::Level::Trace);
    if instruction_trace {
        diagnostics.instruction_trace = true;
        log::info!("instruction trace enabled; execution will be substantially slower");
    }
    let process_started = Instant::now();
    let mut process = ProcessBuilder::new()
        .with_diagnostics(diagnostics)
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
    let execution_started = Instant::now();
    let execution = execute(
        &mut process,
        instruction_trace,
        &interrupted,
        initial_operation_mode,
    );
    log::debug!(
        "guest execution stopped after {:?}",
        execution_started.elapsed()
    );
    let teardown = process.teardown();
    let summary = match execution {
        Ok(summary) => summary,
        Err(error) => {
            log::info!("process resources released after failure: {error}");
            return Err(error);
        }
    };
    let exit_code = teardown.exit.map_or(0, |exit| exit.exit_code);
    let exit_cause = teardown.exit.map_or_else(
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
    classify_exit(teardown.exit)
}

fn install_interrupt_handler() -> Result<Arc<AtomicBool>, String> {
    let interrupted = Arc::new(AtomicBool::new(false));
    let signal_flag = Arc::clone(&interrupted);
    ctrlc::set_handler(move || {
        signal_flag.store(true, Ordering::SeqCst);
    })
    .map_err(|error| format!("cannot install Ctrl+C handler: {error}"))?;
    Ok(interrupted)
}

struct ExecutionSummary {
    instructions: u64,
    svc_calls: u64,
    rejected_svc_kinds: usize,
}

fn execute(
    process: &mut RunnableProcess,
    print_trace: bool,
    interrupted: &AtomicBool,
    initial_operation_mode: OperationMode,
) -> Result<ExecutionSummary, String> {
    let mut dispatcher = HorizonSvcDispatcher::new(initial_operation_mode);
    let mut instructions = 0_u64;
    let execution_started = Instant::now();
    let mut next_progress = EXECUTION_PROGRESS_INTERVAL;
    let mut rejected = BTreeSet::new();
    let mut last_trace_sequence = None;
    loop {
        if interrupted.load(Ordering::SeqCst) {
            log::info!("Ctrl+C received; stopping the guest process cleanly");
            if !process.terminate() {
                return Err(
                    "Ctrl+C received, but the guest process could not be terminated cleanly"
                        .to_owned(),
                );
            }
            return Ok(execution_summary(instructions, &dispatcher, rejected.len()));
        }
        let report = process
            .run_reference(if print_trace {
                1
            } else {
                EXECUTION_SLICE_INSTRUCTIONS
            })
            .map_err(|error| error.to_string())?;
        instructions = instructions.saturating_add(report.instructions_executed);
        if log::log_enabled!(log::Level::Debug) && instructions >= next_progress {
            log::debug!(
                "guest execution progress: instructions={instructions}, elapsed={:?}",
                execution_started.elapsed()
            );
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
                if !process.resume() {
                    return Err(format!("cannot resume scheduled process: {report}"));
                }
            }
            ExecutionStop::SupervisorCall { .. } => {
                match process
                    .route_supervisor_call(&report.stop, &mut dispatcher)
                    .map_err(|error| error.to_string())?
                {
                    ExceptionHandlingResult::Resumed => {}
                    ExceptionHandlingResult::Rejected(error) => {
                        let diagnostic = error.to_string();
                        if rejected.insert(diagnostic.clone()) {
                            log::warn!(
                                "guest requested an unavailable or incomplete Horizon service: {diagnostic}"
                            );
                        }
                    }
                    ExceptionHandlingResult::Terminated { .. } => {
                        return Ok(execution_summary(instructions, &dispatcher, rejected.len()));
                    }
                    ExceptionHandlingResult::Suspended => {
                        return Err(format!(
                            "title suspended without a scheduler after {instructions} instructions: {report}"
                        ));
                    }
                    ExceptionHandlingResult::Fault(error) => {
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
        ProcessExitCause::HostRequested => Err(format!(
            "guest execution was interrupted by the host after a clean shutdown (code {:#x})",
            exit.exit_code
        )),
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
    fn rejects_host_interrupts_and_missing_exit_records() {
        let interrupted =
            classify_exit(Some(process_exit(ProcessExitCause::HostRequested, 0))).unwrap_err();
        assert!(interrupted.contains("interrupted by the host"));
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
