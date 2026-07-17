/// Converts a user-supplied path into a format-independent launch plan.
///
/// This component will detect directories, title containers, and standalone
/// homebrew executables, then coordinate the title, content, and executable
/// loaders without exposing their format-specific details to the runtime.
#[derive(Debug)]
pub struct Launcher;
