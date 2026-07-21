/// Builds an emulated process from a prepared launch plan.
///
/// This component will create the guest address space, map executable
/// segments, install service and file-system handles, configure permissions,
/// and initialize the main thread at the executable entry point.
#[derive(Debug, Default)]
pub struct ProcessBuilder {
    diagnostics: crate::DiagnosticsPolicy,
}

impl ProcessBuilder {
    /// Creates a process builder using detailed diagnostic reports by default.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the immutable diagnostics policy inherited by the process.
    #[must_use]
    pub const fn with_diagnostics(mut self, diagnostics: crate::DiagnosticsPolicy) -> Self {
        self.diagnostics = diagnostics;
        self
    }

    /// Returns the session policy retained by this builder.
    #[must_use]
    pub const fn diagnostics(&self) -> crate::DiagnosticsPolicy {
        self.diagnostics
    }

    /// Returns the narrow diagnostic view passed to CPU frontend resources.
    #[must_use]
    pub const fn cpu_diagnostics(&self) -> swiitx_cpu::coverage::CpuDiagnosticsConfig {
        self.diagnostics.cpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_propagates_runtime_diagnostics_to_cpu_resources() {
        let builder = ProcessBuilder::new();
        assert_eq!(
            builder.cpu_diagnostics().report_detail,
            swiitx_cpu::coverage::MissingInstructionReportDetail::Detailed
        );

        let builder = builder.with_diagnostics(crate::DiagnosticsPolicy {
            report_detail: crate::ReportDetail::Sanitized,
            ..crate::DiagnosticsPolicy::default()
        });
        assert_eq!(
            builder.cpu_diagnostics().report_detail,
            swiitx_cpu::coverage::MissingInstructionReportDetail::Sanitized
        );
    }
}
