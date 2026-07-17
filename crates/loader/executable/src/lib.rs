//! Loaders for executable images that can be mapped into emulated memory.

mod nro;
mod nso;

pub use nro::NroLoader;
pub use nso::NsoLoader;

/// Placeholder representation of a parsed executable image.
///
/// Entry points, memory segments, permissions, and relocation information will
/// be added when executable parsing is implemented.
#[derive(Debug)]
pub struct ExecutableImage;
