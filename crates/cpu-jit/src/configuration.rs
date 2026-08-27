//! Product-selected diagnostics for the direct JIT.

use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct JitConfiguration {
    dump_directory: Option<PathBuf>,
    performance_report_directory: Option<PathBuf>,
    performance_report_title: Box<str>,
}

impl JitConfiguration {
    #[must_use]
    pub fn with_dump_directory(mut self, directory: Option<PathBuf>) -> Self {
        self.dump_directory = directory.filter(|path| !path.as_os_str().is_empty());
        self
    }

    #[must_use]
    pub fn with_performance_report_directory(mut self, directory: Option<PathBuf>) -> Self {
        self.performance_report_directory = directory.filter(|path| !path.as_os_str().is_empty());
        self
    }

    #[must_use]
    pub fn with_performance_report_title(mut self, title: impl Into<Box<str>>) -> Self {
        let title = title.into();
        if !title.trim().is_empty() {
            self.performance_report_title = title;
        }
        self
    }

    #[must_use]
    pub fn dump_directory(&self) -> Option<&Path> {
        self.dump_directory.as_deref()
    }

    #[must_use]
    pub fn performance_report_directory(&self) -> Option<&Path> {
        self.performance_report_directory.as_deref()
    }

    #[must_use]
    pub fn performance_report_title(&self) -> &str {
        if self.performance_report_title.is_empty() {
            "nixe"
        } else {
            &self.performance_report_title
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_retains_only_selected_diagnostics() {
        let configuration = JitConfiguration::default()
            .with_dump_directory(Some("dump".into()))
            .with_performance_report_directory(Some("performance".into()))
            .with_performance_report_title("es2gears");

        assert_eq!(configuration.dump_directory(), Some(Path::new("dump")));
        assert_eq!(
            configuration.performance_report_directory(),
            Some(Path::new("performance"))
        );
        assert_eq!(configuration.performance_report_title(), "es2gears");
        assert_eq!(
            JitConfiguration::default()
                .with_performance_report_title("   ")
                .performance_report_title(),
            "nixe"
        );
    }
}
