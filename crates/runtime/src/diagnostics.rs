//! Runtime-owned diagnostic policy and subsystem-specific views.

use swiitx_cpu::coverage::{CpuDiagnosticsConfig, MissingInstructionReportDetail};

/// Amount of diagnostic context retained across emulator subsystems.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum ReportDetail {
    /// Retain bounded local context useful during emulator development.
    #[default]
    Detailed,
    /// Retain only context intended for public sharing.
    Sanitized,
}

/// Immutable diagnostics policy selected for an emulation session.
///
/// The runtime owns this cross-cutting policy. Subsystems receive narrow,
/// dependency-safe views rather than the application configuration object.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DiagnosticsPolicy {
    pub report_detail: ReportDetail,
    pub missing_instruction_reports: bool,
    pub ir_dumps: bool,
    pub host_code_dumps: bool,
    pub gpu_command_dumps: bool,
}

impl DiagnosticsPolicy {
    /// Derives the only diagnostic settings needed by the CPU frontend.
    #[must_use]
    pub const fn cpu(self) -> CpuDiagnosticsConfig {
        CpuDiagnosticsConfig {
            missing_instruction_reports: self.missing_instruction_reports,
            report_detail: match self.report_detail {
                ReportDetail::Detailed => MissingInstructionReportDetail::Detailed,
                ReportDetail::Sanitized => MissingInstructionReportDetail::Sanitized,
            },
        }
    }
}

impl Default for DiagnosticsPolicy {
    fn default() -> Self {
        Self {
            report_detail: ReportDetail::Detailed,
            missing_instruction_reports: true,
            ir_dumps: false,
            host_code_dumps: false,
            gpu_command_dumps: false,
        }
    }
}

impl From<swiitx_config::DiagnosticsConfig> for DiagnosticsPolicy {
    fn from(config: swiitx_config::DiagnosticsConfig) -> Self {
        Self {
            report_detail: match config.report_detail {
                swiitx_config::DiagnosticReportDetail::Detailed => ReportDetail::Detailed,
                swiitx_config::DiagnosticReportDetail::Sanitized => ReportDetail::Sanitized,
            },
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detailed_reports_are_the_runtime_default() {
        let policy = DiagnosticsPolicy::default();
        assert_eq!(policy.report_detail, ReportDetail::Detailed);
        assert!(policy.missing_instruction_reports);
        assert_eq!(
            policy.cpu().report_detail,
            MissingInstructionReportDetail::Detailed
        );
    }

    #[test]
    fn cpu_receives_only_its_narrow_policy_view() {
        let policy = DiagnosticsPolicy {
            report_detail: ReportDetail::Sanitized,
            missing_instruction_reports: false,
            ir_dumps: true,
            host_code_dumps: true,
            gpu_command_dumps: true,
        };
        assert_eq!(
            policy.cpu(),
            CpuDiagnosticsConfig {
                missing_instruction_reports: false,
                report_detail: MissingInstructionReportDetail::Sanitized,
            }
        );
    }

    #[test]
    fn runtime_normalizes_the_application_configuration() {
        let policy = DiagnosticsPolicy::from(swiitx_config::DiagnosticsConfig {
            report_detail: swiitx_config::DiagnosticReportDetail::Sanitized,
        });
        assert_eq!(policy.report_detail, ReportDetail::Sanitized);
        assert!(policy.missing_instruction_reports);
    }
}
