use std::str::FromStr;

use log::{Level, LevelFilter, Log, Metadata, Record, SetLoggerError};

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
    log::set_max_level(level.filter());
    Ok(())
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
        match record.level() {
            Level::Error => eprintln!("[nixe] error: {}", record.args()),
            Level::Warn => eprintln!("[nixe] warning: {}", record.args()),
            Level::Info => eprintln!("[nixe] {}", record.args()),
            Level::Debug => eprintln!("[nixe] debug: {}", record.args()),
            Level::Trace => eprintln!("[nixe] trace: {}", record.args()),
        }
    }

    fn flush(&self) {}
}
