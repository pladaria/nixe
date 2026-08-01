//! Deterministic validation backend for the neutral GPU contract.
//!
//! This crate deliberately performs no rendering. It validates resource
//! lifetime, canonical aliases, access transitions, ordering, and completion
//! without knowing a guest ABI, a console command stream, or a host GPU API.

mod timeline;
mod validator;

pub use timeline::{HeadlessCompletionController, HeadlessControlError};
pub use validator::{HeadlessBackendDriver, HeadlessValidationError};

use nixe_gpu::{Backend, BackendCapabilities, BackendInstanceId};

/// Constructs a validated neutral backend and its independent manual
/// completion controller.
#[must_use]
pub fn backend(
    instance: BackendInstanceId,
    capabilities: BackendCapabilities,
) -> (Backend<HeadlessBackendDriver>, HeadlessCompletionController) {
    let (driver, completion) = HeadlessBackendDriver::new();
    (Backend::new(instance, capabilities, driver), completion)
}
