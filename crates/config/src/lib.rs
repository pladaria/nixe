//! Shared configuration for Swiix applications.

use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use swiitx_loader_title::{DirectoryScanOptions, NacpLanguage};

/// Configuration file name used during automatic discovery.
pub const CONFIG_FILE_NAME: &str = "swiitx.toml";

/// Current configuration schema version.
pub const CONFIG_VERSION: u32 = 1;

/// Configuration shared by the CLI and desktop applications.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwiitxConfig {
    /// Version of the configuration schema.
    pub version: u32,
    /// Title-library locations and discovery behavior.
    pub library: LibraryConfig,
    /// System-wide preferences and caller-owned key location.
    pub system: SystemConfig,
    /// Cross-cutting diagnostic preferences consumed by application runtimes.
    pub diagnostics: DiagnosticsConfig,
    source_path: PathBuf,
}

impl SwiitxConfig {
    /// Loads and validates a configuration file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_owned(),
            source,
        })?;
        let raw: RawConfig = toml::from_str(&text).map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            source,
        })?;
        if raw.version != CONFIG_VERSION {
            return Err(ConfigError::UnsupportedVersion {
                path: path.to_owned(),
                version: raw.version,
            });
        }

        let source_path = absolute_path(path).map_err(|source| ConfigError::Io {
            path: path.to_owned(),
            source,
        })?;
        let base_directory = source_path
            .parent()
            .expect("an absolute file path must have a parent");

        Ok(Self {
            version: raw.version,
            library: LibraryConfig {
                paths: raw
                    .library
                    .paths
                    .into_iter()
                    .map(|path| resolve_path(base_directory, path))
                    .collect(),
                recursive_scan: raw.library.recursive_scan,
            },
            system: SystemConfig {
                preferred_languages: raw.system.preferred_languages,
                keys: resolve_path(base_directory, raw.system.keys),
            },
            diagnostics: DiagnosticsConfig {
                report_detail: raw.diagnostics.report_detail,
                instruction_trace: raw.diagnostics.instruction_trace,
            },
            source_path,
        })
    }

    /// Loads the first automatically discovered configuration file.
    pub fn load_discovered() -> Result<Option<Self>, ConfigError> {
        Self::discover_path().map(Self::load).transpose()
    }

    /// Finds the configuration selected by the environment or conventional paths.
    pub fn discover_path() -> Option<PathBuf> {
        if let Some(path) = env::var_os("SWIITX_CONFIG").filter(|path| !path.is_empty()) {
            return Some(PathBuf::from(path));
        }

        let local = PathBuf::from(CONFIG_FILE_NAME);
        if local.is_file() {
            return Some(local);
        }

        user_config_path().filter(|path| path.is_file())
    }

    /// Returns the absolute path of the file from which this value was loaded.
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }
}

/// Title-library locations and discovery behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryConfig {
    /// Directories containing title packages.
    pub paths: Vec<PathBuf>,
    /// Whether directory scans descend into subdirectories.
    pub recursive_scan: bool,
}

impl LibraryConfig {
    /// Converts the shared setting to the title loader's scan options.
    pub const fn scan_options(&self) -> DirectoryScanOptions {
        DirectoryScanOptions::new().with_recursive(self.recursive_scan)
    }
}

/// System-wide preferences shared by applications.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemConfig {
    /// Languages to try in descending preference order.
    pub preferred_languages: Vec<NacpLanguage>,
    /// Directory containing caller-owned `prod.keys` and optional `title.keys`.
    pub keys: PathBuf,
}

/// User preference for the amount of context retained in diagnostic reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DiagnosticReportDetail {
    /// Retain bounded local context useful during emulator development.
    #[default]
    Detailed,
    /// Retain only minimal context suitable for public sharing.
    Sanitized,
}

/// Cross-cutting diagnostics configuration shared by applications.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiagnosticsConfig {
    /// Detail level requested for CPU, backend, GPU, and runtime reports.
    pub report_detail: DiagnosticReportDetail,
    /// Whether runtimes retain a bounded recent guest-instruction trace.
    pub instruction_trace: bool,
}

/// Errors produced while locating or loading shared configuration.
#[derive(Debug)]
pub enum ConfigError {
    /// The configuration file could not be read or its path could not be resolved.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The TOML document does not match the configuration schema.
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    /// The document uses a schema version this build cannot interpret.
    UnsupportedVersion { path: PathBuf, version: u32 },
}

impl Display for ConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    formatter,
                    "cannot read configuration {}: {source}",
                    path.display()
                )
            }
            Self::Parse { path, source } => {
                write!(
                    formatter,
                    "invalid configuration {}: {source}",
                    path.display()
                )
            }
            Self::UnsupportedVersion { path, version } => write!(
                formatter,
                "configuration {} uses unsupported version {version}; expected {CONFIG_VERSION}",
                path.display()
            ),
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
            Self::UnsupportedVersion { .. } => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    version: u32,
    library: RawLibraryConfig,
    system: RawSystemConfig,
    #[serde(default)]
    diagnostics: RawDiagnosticsConfig,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLibraryConfig {
    paths: Vec<PathBuf>,
    #[serde(default = "default_recursive_scan")]
    recursive_scan: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSystemConfig {
    #[serde(deserialize_with = "deserialize_languages")]
    preferred_languages: Vec<NacpLanguage>,
    keys: PathBuf,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDiagnosticsConfig {
    #[serde(default)]
    report_detail: DiagnosticReportDetail,
    #[serde(default)]
    instruction_trace: bool,
}

const fn default_recursive_scan() -> bool {
    true
}

fn deserialize_languages<'de, D>(deserializer: D) -> Result<Vec<NacpLanguage>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let names = Vec::<String>::deserialize(deserializer)?;
    names
        .into_iter()
        .map(|name| {
            NacpLanguage::ALL
                .into_iter()
                .find(|language| language.icon_suffix() == name)
                .ok_or_else(|| serde::de::Error::custom(format!("unknown language `{name}`")))
        })
        .collect()
}

fn absolute_path(path: &Path) -> std::io::Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(env::current_dir()?.join(path))
    }
}

fn resolve_path(base_directory: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base_directory.join(path)
    }
}

#[cfg(target_os = "windows")]
fn user_config_path() -> Option<PathBuf> {
    env::var_os("APPDATA").map(|root| PathBuf::from(root).join("Swiitx").join(CONFIG_FILE_NAME))
}

#[cfg(target_os = "macos")]
fn user_config_path() -> Option<PathBuf> {
    env::var_os("HOME").map(|root| {
        PathBuf::from(root)
            .join("Library/Application Support/Swiitx")
            .join(CONFIG_FILE_NAME)
    })
}

#[cfg(all(unix, not(target_os = "macos")))]
fn user_config_path() -> Option<PathBuf> {
    env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|root| PathBuf::from(root).join(".config")))
        .map(|root| root.join("swiitx").join(CONFIG_FILE_NAME))
}

#[cfg(not(any(unix, target_os = "windows")))]
fn user_config_path() -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    struct TemporaryConfig {
        directory: PathBuf,
        path: PathBuf,
    }

    impl TemporaryConfig {
        fn new(contents: &str) -> Self {
            let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
            let directory =
                env::temp_dir().join(format!("swiitx-config-{}-{sequence}", std::process::id()));
            fs::create_dir(&directory).unwrap();
            let path = directory.join(CONFIG_FILE_NAME);
            fs::write(&path, contents).unwrap();
            Self { directory, path }
        }
    }

    impl Drop for TemporaryConfig {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.directory).unwrap();
        }
    }

    #[test]
    fn loads_typed_values_and_resolves_relative_paths() {
        let file = TemporaryConfig::new(
            r#"
                version = 1
                [library]
                paths = ["./roms", "other"]
                recursive_scan = false
                [system]
                preferred_languages = ["Spanish", "AmericanEnglish"]
                keys = "./keys"
            "#,
        );

        let config = SwiitxConfig::load(&file.path).unwrap();
        let base = file.path.parent().unwrap();

        assert_eq!(config.source_path(), file.path);
        assert_eq!(config.library.paths[0], base.join("./roms"));
        assert_eq!(config.library.paths[1], base.join("other"));
        assert!(!config.library.scan_options().recursive);
        assert_eq!(
            config.system.preferred_languages,
            vec![NacpLanguage::Spanish, NacpLanguage::AmericanEnglish]
        );
        assert_eq!(config.system.keys, base.join("./keys"));
        assert_eq!(
            config.diagnostics.report_detail,
            DiagnosticReportDetail::Detailed
        );
        assert!(!config.diagnostics.instruction_trace);
    }

    #[test]
    fn defaults_recursive_scanning_to_true() {
        let file = TemporaryConfig::new(
            r#"
                version = 1
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
            "#,
        );

        let config = SwiitxConfig::load(&file.path).unwrap();

        assert!(config.library.recursive_scan);
        assert_eq!(
            config.diagnostics.report_detail,
            DiagnosticReportDetail::Detailed
        );
        assert!(!config.diagnostics.instruction_trace);
    }

    #[test]
    fn loads_explicit_sanitized_diagnostic_policy() {
        let file = TemporaryConfig::new(
            r#"
                version = 1
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                [diagnostics]
                report_detail = "sanitized"
                instruction_trace = true
            "#,
        );

        let config = SwiitxConfig::load(&file.path).unwrap();
        assert_eq!(
            config.diagnostics.report_detail,
            DiagnosticReportDetail::Sanitized
        );
        assert!(config.diagnostics.instruction_trace);
    }

    #[test]
    fn rejects_unknown_fields_languages_and_versions() {
        for contents in [
            r#"
                version = 1
                typo = true
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
            "#,
            r#"
                version = 1
                [library]
                paths = []
                [system]
                preferred_languages = ["Klingon"]
                keys = "keys"
            "#,
        ] {
            let file = TemporaryConfig::new(contents);
            assert!(matches!(
                SwiitxConfig::load(&file.path),
                Err(ConfigError::Parse { .. })
            ));
        }

        let file = TemporaryConfig::new(
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
            "#,
        );
        assert!(matches!(
            SwiitxConfig::load(&file.path),
            Err(ConfigError::UnsupportedVersion { version: 2, .. })
        ));
    }
}
