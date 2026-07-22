use std::collections::BTreeSet;
use std::path::PathBuf;

use swiitx_cli::library::{Library, LibraryTitleSource};
use swiitx_horizon::HorizonSvcDispatcher;
use swiitx_runtime::{
    ExceptionHandlingResult, ExecutionStop, Launcher, LauncherInput, ProcessBuilder,
    RunnableProcess,
};

use super::load_config;

const EXECUTION_SLICE_INSTRUCTIONS: u64 = 100_000;

pub struct Arguments {
    pub config_path: Option<PathBuf>,
    pub identifier: String,
}

pub fn run(arguments: Arguments) -> Result<(), String> {
    log("scanning configured title library");
    let config = load_config(arguments.config_path)?;
    let library = Library::scan(&config)?;
    let title = library
        .find(&arguments.identifier)
        .ok_or_else(|| format!("unknown title ID: {}", arguments.identifier))?;
    log(&format!("selected {}: {}", title.identifier, title.name));

    let plan = match &title.source {
        LibraryTitleSource::Installed(title) => {
            log(
                "source is an installed title; building from the resolved base, update and DLC set",
            );
            Launcher::build_resolved_title((**title).clone(), &library.keys)
        }
        LibraryTitleSource::Homebrew(path) => {
            log(&format!("source is a homebrew NRO: {}", path.display()));
            Launcher::build(LauncherInput::new(path))
        }
    }
    .map_err(|error| error.to_string())?;
    log(&format!(
        "launch plan ready: {} module(s), entry={}, primary RomFS={}, DLC={}",
        plan.modules().len(),
        plan.entry_module().name(),
        if plan.primary_file_system().is_some() {
            "yes"
        } else {
            "no"
        },
        plan.add_ons().len()
    ));
    for module in plan.modules() {
        log(&format!(
            "module {} ({:?}) loaded into the plan",
            module.name(),
            module.role()
        ));
    }

    log("preparing process memory and initial thread state");
    let mut process = ProcessBuilder::new()
        .with_diagnostics(config.diagnostics.into())
        .build(&plan)
        .map_err(|error| error.to_string())?;
    log(&format!(
        "process ready: entry={:#018x}, modules={}",
        process.entry_module().entry_address(),
        process.modules().len()
    ));
    log("starting the reference CPU interpreter");

    let execution = execute(&mut process);
    let teardown = process.teardown();
    let summary = match execution {
        Ok(summary) => summary,
        Err(error) => {
            log(&format!(
                "process resources released after failure: {error}"
            ));
            return Err(error);
        }
    };
    let exit_code = teardown.exit.map_or(0, |exit| exit.exit_code);
    let exit_cause = teardown.exit.map_or_else(
        || "without an exit record".to_owned(),
        |exit| format!("{:?}", exit.cause),
    );
    log(&format!(
        "execution finished: instructions={}, SVC calls={}, rejected SVC kinds={}, cause={}, code={:#x}",
        summary.instructions, summary.svc_calls, summary.rejected_svc_kinds, exit_cause, exit_code
    ));
    if exit_code == 0 {
        Ok(())
    } else {
        Err(format!("title exited with code {exit_code:#x}"))
    }
}

struct ExecutionSummary {
    instructions: u64,
    svc_calls: u64,
    rejected_svc_kinds: usize,
}

fn execute(process: &mut RunnableProcess) -> Result<ExecutionSummary, String> {
    let mut dispatcher = HorizonSvcDispatcher::default();
    let mut instructions = 0_u64;
    let mut rejected = BTreeSet::new();
    loop {
        let report = process
            .run_reference(EXECUTION_SLICE_INSTRUCTIONS)
            .map_err(|error| error.to_string())?;
        instructions = instructions.saturating_add(report.instructions_executed);
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
                            warning(&format!(
                                "guest requested an unavailable or incomplete Horizon service: {diagnostic}"
                            ));
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
    report: &swiitx_runtime::ExecutionReport,
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

fn log(message: &str) {
    eprintln!("[swiitx] {message}");
}

fn warning(message: &str) {
    eprintln!("[swiitx] warning: {message}");
}
