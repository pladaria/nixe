mod commands;

use std::env;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::ExitCode;

enum Command {
    List(commands::list::Arguments),
    Run(commands::run::Arguments),
}

fn main() -> ExitCode {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .unwrap_or_else(|| OsString::from("nixe-cli"));

    let command = match parse_arguments(arguments) {
        Ok(Some(command)) => command,
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

    let result = match command {
        Command::List(arguments) => commands::list::run(arguments),
        Command::Run(arguments) => commands::run::run(arguments),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn parse_arguments(arguments: impl Iterator<Item = OsString>) -> Result<Option<Command>, String> {
    let mut config_path = None;
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
        if argument.to_string_lossy().starts_with('-') {
            return Err(format!("unknown option: {}", argument.to_string_lossy()));
        }
        positionals.push(argument);
    }

    match positionals.as_slice() {
        [command] if command == "list" => Ok(Some(Command::List(commands::list::Arguments {
            config_path,
        }))),
        [command, identifier] if command == "run" => {
            let identifier = identifier
                .to_str()
                .ok_or_else(|| "title ID must be valid UTF-8".to_owned())?
                .to_owned();
            Ok(Some(Command::Run(commands::run::Arguments {
                config_path,
                identifier,
            })))
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
        "Usage: {} [--config <file>] <command>\n\n\
         Commands:\n  \
           list        List configured titles as title ID and localized name\n  \
           run <id>    Run the configured title selected by ID\n\n\
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
        let Some(Command::List(arguments)) = parse_arguments(arguments(&["list"])).unwrap() else {
            panic!("expected list command");
        };
        assert_eq!(arguments.config_path, None);
    }

    #[test]
    fn accepts_config_before_or_after_list() {
        for values in [
            &["--config", "custom.toml", "list"][..],
            &["list", "--config", "custom.toml"][..],
        ] {
            let Some(Command::List(arguments)) = parse_arguments(arguments(values)).unwrap() else {
                panic!("expected list command");
            };
            assert_eq!(arguments.config_path, Some(PathBuf::from("custom.toml")));
        }
    }

    #[test]
    fn parses_run_with_installed_or_homebrew_identifier() {
        for identifier in ["01002CD00A51C000", "nro:48CAE2E7721D392D"] {
            let Some(Command::Run(arguments)) =
                parse_arguments(arguments(&["run", identifier])).unwrap()
            else {
                panic!("expected run command");
            };
            assert_eq!(arguments.config_path, None);
            assert_eq!(arguments.identifier, identifier);
        }
    }

    #[test]
    fn rejects_missing_unknown_and_extra_commands() {
        for values in [
            &[][..],
            &["run"][..],
            &["run", "one", "two"][..],
            &["list", "extra"][..],
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
