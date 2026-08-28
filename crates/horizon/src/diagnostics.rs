//! Guest-visible diagnostic policy and host-side guest log routing.

/// Host severity policy applied to messages received from Horizon's `lm` service.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum GuestLogLevel {
    /// Preserve the severity encoded in each guest log packet.
    #[default]
    Inherit,
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Off,
}

impl GuestLogLevel {
    pub(crate) const fn resolve(self, severity: GuestLogSeverity) -> Option<log::Level> {
        match self {
            Self::Inherit => Some(severity.host_level()),
            Self::Trace => Some(log::Level::Trace),
            Self::Debug => Some(log::Level::Debug),
            Self::Info => Some(log::Level::Info),
            Self::Warn => Some(log::Level::Warn),
            Self::Error => Some(log::Level::Error),
            Self::Off => None,
        }
    }
}

/// Severity encoded by the Horizon SDK in one `ILogger::Log` packet.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GuestLogSeverity {
    Trace,
    Info,
    Warn,
    Error,
    Fatal,
}

impl GuestLogSeverity {
    pub(crate) const fn decode(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Trace),
            1 => Some(Self::Info),
            2 => Some(Self::Warn),
            3 => Some(Self::Error),
            4 => Some(Self::Fatal),
            _ => None,
        }
    }

    const fn host_level(self) -> log::Level {
        match self {
            Self::Trace => log::Level::Trace,
            Self::Info => log::Level::Info,
            Self::Warn => log::Level::Warn,
            Self::Error | Self::Fatal => log::Level::Error,
        }
    }
}

/// Runtime diagnostic policy exposed to Horizon services.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct HorizonDiagnostics {
    pub guest_logs_level: GuestLogLevel,
    pub file_system_access_log: bool,
}

impl HorizonDiagnostics {
    #[must_use]
    pub const fn new(guest_logs_level: GuestLogLevel, file_system_access_log: bool) -> Self {
        Self {
            guest_logs_level,
            file_system_access_log,
        }
    }

    pub(crate) const fn file_system_access_log_mode(self) -> FileSystemAccessLogMode {
        if self.file_system_access_log {
            FileSystemAccessLogMode::Log
        } else {
            FileSystemAccessLogMode::None
        }
    }
}

/// Global access-log mode returned by `IFileSystemProxy`.
///
/// ABI values match Atmosphere's reconstruction of the Horizon filesystem SDK:
/// https://github.com/Atmosphere-NX/Atmosphere/blob/cb4b882e3b176480ac57a1161a85ff175c3f162c/libraries/libstratosphere/include/stratosphere/fs/fs_access_log.hpp#L23-L27
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum FileSystemAccessLogMode {
    #[default]
    None = 0,
    Log = 1,
    SdCard = 2,
}

impl FileSystemAccessLogMode {
    pub(crate) const fn raw(self) -> u32 {
        self as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inherited_guest_severity_maps_to_the_host_logger() {
        assert_eq!(
            GuestLogLevel::Inherit.resolve(GuestLogSeverity::Trace),
            Some(log::Level::Trace)
        );
        assert_eq!(
            GuestLogLevel::Inherit.resolve(GuestLogSeverity::Fatal),
            Some(log::Level::Error)
        );
        assert_eq!(GuestLogLevel::Off.resolve(GuestLogSeverity::Info), None);
        assert_eq!(
            GuestLogLevel::Debug.resolve(GuestLogSeverity::Error),
            Some(log::Level::Debug)
        );
    }

    #[test]
    fn filesystem_access_logging_is_opt_in() {
        assert_eq!(
            HorizonDiagnostics::default().file_system_access_log_mode(),
            FileSystemAccessLogMode::None
        );
        assert_eq!(
            HorizonDiagnostics::new(GuestLogLevel::Inherit, true).file_system_access_log_mode(),
            FileSystemAccessLogMode::Log
        );
    }
}
