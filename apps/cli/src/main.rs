mod commands;
mod logging;

use std::env;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::ExitCode;

enum Command {
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
        if argument.to_string_lossy().starts_with('-') {
            return Err(format!("unknown option: {}", argument.to_string_lossy()));
        }
        positionals.push(argument);
    }

    match positionals.as_slice() {
        [command] if command == "list" => Ok(Some(Invocation {
            command: Command::List(commands::list::Arguments {
                config_path,
                log_level_override: log_level,
            }),
            log_level: log_level.unwrap_or_default(),
        })),
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
                }),
                log_level: log_level.unwrap_or_default(),
            }))
        }
        [] => Err("a command is required".to_owned()),
        [command, ..] if command == "list" => Err("list does not accept arguments".to_owned()),
        [command] if command == "run" => Err("run requires a title ID".to_owned()),
        [command, ..] if command == "run" => Err("run accepts exactly one title ID".to_owned()),
        [command, ..] => Err(format!("unknown command: {}", command.to_string_lossy())),
    }
}

fn print_usage(program: &OsStr) {
    eprintln!(
        "Usage: {} [--config <file>] [--log-level <level>] <command>\n\n\
         Commands:\n  \
           list        List configured titles as title ID and localized name\n  \
           run <id>    Run a title\n\n\
         Log levels:\n  \
           error, warn, info, debug, trace\n  \
           --log-level overrides diagnostics.log_level from nixe.toml\n  \
           debug reports phase timings; trace also prints every instruction\n\n\
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
            assert_eq!(invocation.log_level, logging::LogLevel::Info);
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
