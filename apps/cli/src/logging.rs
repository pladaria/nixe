use std::io::{self, IsTerminal};
use std::str::FromStr;

use log::{Level, LevelFilter, Log, Metadata, Record, SetLoggerError};
use nixe_config::DiagnosticLogLevel;

static LOGGER: NixeLogger = NixeLogger;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogLevel {
    Error,
    Warn,
    #[default]
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    const fn filter(self) -> LevelFilter {
        match self {
            Self::Error => LevelFilter::Error,
            Self::Warn => LevelFilter::Warn,
            Self::Info => LevelFilter::Info,
            Self::Debug => LevelFilter::Debug,
            Self::Trace => LevelFilter::Trace,
        }
    }
}

impl From<DiagnosticLogLevel> for LogLevel {
    fn from(level: DiagnosticLogLevel) -> Self {
        match level {
            DiagnosticLogLevel::Error => Self::Error,
            DiagnosticLogLevel::Warn => Self::Warn,
            DiagnosticLogLevel::Info => Self::Info,
            DiagnosticLogLevel::Debug => Self::Debug,
            DiagnosticLogLevel::Trace => Self::Trace,
        }
    }
}

impl FromStr for LogLevel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "error" => Ok(Self::Error),
            "warn" => Ok(Self::Warn),
            "info" => Ok(Self::Info),
            "debug" => Ok(Self::Debug),
            "trace" => Ok(Self::Trace),
            _ => Err(format!(
                "invalid log level {value:?}; expected error, warn, info, debug, or trace"
            )),
        }
    }
}

pub fn init(level: LogLevel) -> Result<(), SetLoggerError> {
    log::set_logger(&LOGGER)?;
    set_level(level);
    Ok(())
}

pub fn set_level(level: LogLevel) {
    log::set_max_level(level.filter());
}

struct NixeLogger;

impl Log for NixeLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        match (record.level(), color_enabled()) {
            (Level::Error, true) => {
                eprintln!("\x1b[31m[nixe] error: {}\x1b[0m", record.args());
            }
            (Level::Warn, true) => {
                eprintln!("\x1b[33m[nixe] warning: {}\x1b[0m", record.args());
            }
            (Level::Error, false) => eprintln!("[nixe] error: {}", record.args()),
            (Level::Warn, false) => eprintln!("[nixe] warning: {}", record.args()),
            (Level::Info, _) => eprintln!("[nixe] {}", record.args()),
            (Level::Debug, _) => eprintln!("[nixe] debug: {}", record.args()),
            (Level::Trace, _) => eprintln!("[nixe] trace: {}", record.args()),
        }
    }

    fn flush(&self) {}
}

fn color_enabled() -> bool {
    io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}
