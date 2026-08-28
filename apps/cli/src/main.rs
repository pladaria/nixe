mod commands;
mod logging;

use std::env;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::ExitCode;

use nixe_config::{CpuBackendSelection, GuestLogsLevel};

enum Command {
    Input,
    List(commands::list::Arguments),
    Run(commands::run::Arguments),
}

struct Invocation {
    command: Command,
    log_level: logging::LogLevel,
}

fn main() -> ExitCode {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .unwrap_or_else(|| OsString::from("nixe-cli"));

    let invocation = match parse_arguments(arguments) {
        Ok(Some(invocation)) => invocation,
        Ok(None) => {
            print_usage(&program);
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("error: {error}");
            print_usage(&program);
            return ExitCode::from(2);
        }
    };

    if let Err(error) = logging::init(invocation.log_level) {
        eprintln!("error: cannot initialize logging: {error}");
        return ExitCode::FAILURE;
    }

    let result = match invocation.command {
        Command::Input => commands::input::run(),
        Command::List(arguments) => commands::list::run(arguments),
        Command::Run(arguments) => commands::run::run(arguments),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            log::error!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn parse_arguments(
    arguments: impl Iterator<Item = OsString>,
) -> Result<Option<Invocation>, String> {
    let mut config_path = None;
    let mut log_level = None;
    let mut cpu_backend = None;
    let mut guest_logs_level = None;
    let mut file_system_access_log = None;
    let mut headless = false;
    let mut positionals = Vec::new();
    let mut arguments = arguments;

    while let Some(argument) = arguments.next() {
        if argument == "-h" || argument == "--help" {
            return Ok(None);
        }
        if argument == "--config" {
            if config_path.is_some() {
                return Err("--config may only be specified once".to_owned());
            }
            config_path = Some(PathBuf::from(
                arguments
                    .next()
                    .ok_or_else(|| "--config requires a file path".to_owned())?,
            ));
            continue;
        }
        if argument == "--log-level" {
            if log_level.is_some() {
                return Err("--log-level may only be specified once".to_owned());
            }
            let value = arguments
                .next()
                .ok_or_else(|| "--log-level requires a level".to_owned())?;
            let value = value
                .to_str()
                .ok_or_else(|| "log level must be valid UTF-8".to_owned())?;
            log_level = Some(value.parse::<logging::LogLevel>()?);
            continue;
        }
        if argument == "--headless" {
            if headless {
                return Err("--headless may only be specified once".to_owned());
            }
            headless = true;
            continue;
        }
        if argument == "--guest-logs-level" {
            if guest_logs_level.is_some() {
                return Err("--guest-logs-level may only be specified once".to_owned());
            }
            let value = arguments
                .next()
                .ok_or_else(|| "--guest-logs-level requires a level".to_owned())?;
            let value = value
                .to_str()
                .ok_or_else(|| "guest log level must be valid UTF-8".to_owned())?;
            guest_logs_level = Some(parse_guest_logs_level(value)?);
            continue;
        }
        if argument == "--file-system-access-log" || argument == "--no-file-system-access-log" {
            if file_system_access_log.is_some() {
                return Err("filesystem access-log policy may only be specified once".to_owned());
            }
            file_system_access_log = Some(argument == "--file-system-access-log");
            continue;
        }
        if argument == "--cpu-backend" {
            if cpu_backend.is_some() {
                return Err("--cpu-backend may only be specified once".to_owned());
            }
            let value = arguments
                .next()
                .ok_or_else(|| "--cpu-backend requires jit or interpreter".to_owned())?;
            let value = value
                .to_str()
                .ok_or_else(|| "CPU backend must be valid UTF-8".to_owned())?;
            cpu_backend = Some(match value {
                "jit" => CpuBackendSelection::Jit,
                "interpreter" => CpuBackendSelection::Interpreter,
                _ => {
                    return Err(format!(
                        "invalid CPU backend {value:?}; expected jit or interpreter"
                    ));
                }
            });
            continue;
        }
        if argument.to_string_lossy().starts_with('-') {
            return Err(format!("unknown option: {}", argument.to_string_lossy()));
        }
        positionals.push(argument);
    }

    match positionals.as_slice() {
        [command] if command == "input" => {
            if headless {
                return Err("--headless is only valid with run".to_owned());
            }
            if cpu_backend.is_some() {
                return Err("--cpu-backend is only valid with run".to_owned());
            }
            if guest_logs_level.is_some() || file_system_access_log.is_some() {
                return Err("guest diagnostic options are only valid with run".to_owned());
            }
            Ok(Some(Invocation {
                command: Command::Input,
                log_level: log_level.unwrap_or_default(),
            }))
        }
        [command] if command == "list" => {
            if headless {
                return Err("--headless is only valid with run".to_owned());
            }
            if cpu_backend.is_some() {
                return Err("--cpu-backend is only valid with run".to_owned());
            }
            if guest_logs_level.is_some() || file_system_access_log.is_some() {
                return Err("guest diagnostic options are only valid with run".to_owned());
            }
            Ok(Some(Invocation {
                command: Command::List(commands::list::Arguments {
                    config_path,
                    log_level_override: log_level,
                }),
                log_level: log_level.unwrap_or_default(),
            }))
        }
        [command, identifier] if command == "run" => {
            let identifier = identifier
                .to_str()
                .ok_or_else(|| "title ID must be valid UTF-8".to_owned())?
                .to_owned();
            Ok(Some(Invocation {
                command: Command::Run(commands::run::Arguments {
                    config_path,
                    log_level_override: log_level,
                    identifier,
                    headless,
                    cpu_backend_override: cpu_backend,
                    guest_logs_level_override: guest_logs_level,
                    file_system_access_log_override: file_system_access_log,
                }),
                log_level: log_level.unwrap_or_default(),
            }))
        }
        [] => Err("a command is required".to_owned()),
        [command, ..] if command == "input" => Err("input does not accept arguments".to_owned()),
        [command, ..] if command == "list" => Err("list does not accept arguments".to_owned()),
        [command] if command == "run" => Err("run requires a title ID or name".to_owned()),
        [command, ..] if command == "run" => {
            Err("run accepts exactly one title ID or name".to_owned())
        }
        [command, ..] => Err(format!("unknown command: {}", command.to_string_lossy())),
    }
}

fn parse_guest_logs_level(value: &str) -> Result<GuestLogsLevel, String> {
    match value {
        "inherit" => Ok(GuestLogsLevel::Inherit),
        "trace" => Ok(GuestLogsLevel::Trace),
        "debug" => Ok(GuestLogsLevel::Debug),
        "info" => Ok(GuestLogsLevel::Info),
        "warn" => Ok(GuestLogsLevel::Warn),
        "error" => Ok(GuestLogsLevel::Error),
        "off" => Ok(GuestLogsLevel::Off),
        _ => Err(format!(
            "invalid guest log level {value:?}; expected inherit, trace, debug, info, warn, error, or off"
        )),
    }
}

fn print_usage(program: &OsStr) {
    eprintln!(
        "Usage: {} [--config <file>] [--log-level <level>] <command>\n\n\
         Commands:\n  \
           input           Display live state from the first connected gamepad\n  \
           list            List configured titles as title ID and localized name\n  \
           run <id|name>   Run a title\n\n\
         Run options:\n  \
           --headless              Run without creating a host window\n  \
           --cpu-backend <backend> Override CPU backend: jit or interpreter\n  \
           --guest-logs-level <level> Override diagnostics.guest_logs_level\n  \
           --file-system-access-log Enable guest filesystem access logs\n  \
           --no-file-system-access-log Disable guest filesystem access logs\n\n\
         Log levels:\n  \
           error, warn, info, debug, trace\n  \
           --log-level overrides diagnostics.log_level from nixe.toml\n  \
           debug reports phase timings; trace adds execution and service diagnostics\n\n\
         Guest log levels:\n  \
           inherit, trace, debug, info, warn, error, off\n  \
           inherit preserves the severity encoded by the Horizon guest\n\n\
         Configuration is discovered from NIXE_CONFIG, ./nixe.toml, or the\n\
         platform user configuration unless --config is supplied.",
        program.to_string_lossy()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> impl Iterator<Item = OsString> {
        values
            .iter()
            .map(OsString::from)
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn parses_list_with_discovered_configuration() {
        let invocation = parse_arguments(arguments(&["list"])).unwrap().unwrap();
        let Command::List(arguments) = invocation.command else {
            panic!("expected list command");
        };
        assert_eq!(arguments.config_path, None);
        assert_eq!(arguments.log_level_override, None);
        assert_eq!(invocation.log_level, logging::LogLevel::Info);
    }

    #[test]
    fn parses_input_without_configuration() {
        let invocation = parse_arguments(arguments(&["input"])).unwrap().unwrap();
        assert!(matches!(invocation.command, Command::Input));
        assert_eq!(invocation.log_level, logging::LogLevel::Info);
    }

    #[test]
    fn accepts_config_before_or_after_list() {
        for values in [
            &["--config", "custom.toml", "list"][..],
            &["list", "--config", "custom.toml"][..],
        ] {
            let invocation = parse_arguments(arguments(values)).unwrap().unwrap();
            let Command::List(arguments) = invocation.command else {
                panic!("expected list command");
            };
            assert_eq!(arguments.config_path, Some(PathBuf::from("custom.toml")));
        }
    }

    #[test]
    fn parses_run_with_installed_or_homebrew_identifier() {
        for identifier in ["01002CD00A51C000", "nro:48CAE2E7721D392D"] {
            let invocation = parse_arguments(arguments(&["run", identifier]))
                .unwrap()
                .unwrap();
            let Command::Run(arguments) = invocation.command else {
                panic!("expected run command");
            };
            assert_eq!(arguments.config_path, None);
            assert_eq!(arguments.log_level_override, None);
            assert_eq!(arguments.identifier, identifier);
            assert!(!arguments.headless);
            assert_eq!(arguments.cpu_backend_override, None);
            assert_eq!(arguments.guest_logs_level_override, None);
            assert_eq!(arguments.file_system_access_log_override, None);
            assert_eq!(invocation.log_level, logging::LogLevel::Info);
        }
    }

    #[test]
    fn parses_every_cpu_backend_before_or_after_run() {
        for (value, expected) in [
            ("jit", CpuBackendSelection::Jit),
            ("interpreter", CpuBackendSelection::Interpreter),
        ] {
            for values in [
                vec!["--cpu-backend", value, "run", "hello-world"],
                vec!["run", "--cpu-backend", value, "hello-world"],
                vec!["run", "hello-world", "--cpu-backend", value],
            ] {
                let invocation = parse_arguments(arguments(&values)).unwrap().unwrap();
                let Command::Run(arguments) = invocation.command else {
                    panic!("expected run command");
                };
                assert_eq!(arguments.cpu_backend_override, Some(expected));
            }
        }
    }

    #[test]
    fn parses_headless_before_or_after_run_identifier() {
        for values in [
            &["--headless", "run", "hello-world"][..],
            &["run", "--headless", "hello-world"][..],
            &["run", "hello-world", "--headless"][..],
        ] {
            let invocation = parse_arguments(arguments(values)).unwrap().unwrap();
            let Command::Run(arguments) = invocation.command else {
                panic!("expected run command");
            };
            assert_eq!(arguments.identifier, "hello-world");
            assert!(arguments.headless);
        }
    }

    #[test]
    fn parses_guest_diagnostic_overrides_before_or_after_run() {
        for values in [
            &[
                "--guest-logs-level",
                "inherit",
                "--file-system-access-log",
                "run",
                "hello-world",
            ][..],
            &[
                "run",
                "hello-world",
                "--guest-logs-level",
                "error",
                "--no-file-system-access-log",
            ][..],
        ] {
            let invocation = parse_arguments(arguments(values)).unwrap().unwrap();
            let Command::Run(arguments) = invocation.command else {
                panic!("expected run command");
            };
            if values.contains(&"inherit") {
                assert_eq!(
                    arguments.guest_logs_level_override,
                    Some(GuestLogsLevel::Inherit)
                );
                assert_eq!(arguments.file_system_access_log_override, Some(true));
            } else {
                assert_eq!(
                    arguments.guest_logs_level_override,
                    Some(GuestLogsLevel::Error)
                );
                assert_eq!(arguments.file_system_access_log_override, Some(false));
            }
        }
    }

    #[test]
    fn parses_log_level_before_or_after_command() {
        for values in [
            &["--log-level", "trace", "run", "01002CD00A51C000"][..],
            &["run", "--log-level", "trace", "01002CD00A51C000"][..],
            &["list", "--log-level", "trace"][..],
        ] {
            let invocation = parse_arguments(arguments(values)).unwrap().unwrap();
            assert_eq!(invocation.log_level, logging::LogLevel::Trace);
            match invocation.command {
                Command::Input => panic!("expected list or run command"),
                Command::List(arguments) => {
                    assert_eq!(arguments.log_level_override, Some(logging::LogLevel::Trace));
                }
                Command::Run(arguments) => {
                    assert_eq!(arguments.log_level_override, Some(logging::LogLevel::Trace));
                }
            }
        }
    }

    #[test]
    fn parses_every_log_level() {
        for (value, expected) in [
            ("error", logging::LogLevel::Error),
            ("warn", logging::LogLevel::Warn),
            ("info", logging::LogLevel::Info),
            ("debug", logging::LogLevel::Debug),
            ("trace", logging::LogLevel::Trace),
        ] {
            let invocation = parse_arguments(arguments(&["--log-level", value, "list"]))
                .unwrap()
                .unwrap();
            assert_eq!(invocation.log_level, expected);
        }
    }

    #[test]
    fn rejects_missing_unknown_and_extra_commands() {
        for values in [
            &[][..],
            &["run"][..],
            &["run", "one", "two"][..],
            &["list", "extra"][..],
            &["list", "--trace"][..],
            &["--log-level", "verbose", "list"][..],
            &["--log-level", "debug", "--log-level", "trace", "list"][..],
            &["list", "--headless"][..],
            &["input", "--headless"][..],
            &["run", "--headless", "--headless", "hello-world"][..],
            &["--cpu-backend", "jit", "list"][..],
            &["input", "--cpu-backend", "interpreter"][..],
            &["run", "hello-world", "--cpu-backend", "native"][..],
            &["run", "hello-world", "--cpu-backend"][..],
            &[
                "run",
                "hello-world",
                "--cpu-backend",
                "jit",
                "--cpu-backend",
                "interpreter",
            ][..],
            &["--unknown", "list"][..],
            &["--config", "list"][..],
        ] {
            assert!(parse_arguments(arguments(values)).is_err());
        }
    }

    #[test]
    fn accepts_help() {
        assert!(parse_arguments(arguments(&["--help"])).unwrap().is_none());
        assert!(
            parse_arguments(arguments(&["list", "--help"]))
                .unwrap()
                .is_none()
        );
    }
}
