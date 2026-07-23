//! Host input abstractions and platform backends.

mod model;
pub mod sdl;

pub use model::{
    Button, ButtonSet, ControllerId, ControllerKind, ControllerState, DPadState, InputSnapshot,
    TriggerState,
};

/// Produces complete snapshots of the currently connected host controllers.
///
/// Consumers should replace their previous snapshot atomically after a
/// successful poll. A missing controller therefore means that it disconnected.
pub trait HostInputBackend {
    type Error: std::error::Error + Send + Sync + 'static;

    fn poll(&mut self) -> Result<InputSnapshot, Self::Error>;
}
