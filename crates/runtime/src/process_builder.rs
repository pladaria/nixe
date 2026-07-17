/// Builds an emulated process from a prepared launch plan.
///
/// This component will create the guest address space, map executable
/// segments, install service and file-system handles, configure permissions,
/// and initialize the main thread at the executable entry point.
#[derive(Debug)]
pub struct ProcessBuilder;
