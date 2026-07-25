//! Host input abstractions and platform backends.

mod model;
pub mod sdl;

pub use model::{
    Axis, Button, ButtonLabel, ButtonSet, ControllerId, ControllerKind, ControllerState, DPadState,
    FaceButtonLabels, IdentifierError, InputSnapshot, MotionSensor, MotionState, MotionVector,
    StickState, TriggerState,
};

/// Produces complete snapshots of the currently connected host controllers.
///
/// Consumers should replace their previous snapshot atomically after a
/// successful poll. A missing controller therefore means that it disconnected.
pub trait HostInputBackend {
    type Error: std::error::Error + Send + Sync + 'static;

    fn poll(&mut self) -> Result<InputSnapshot, Self::Error>;
}

/// Selects the first attached controller from a host backend.
///
/// Backends keep controllers in attachment order, so the selected controller
/// remains stable until it disconnects.
pub struct InputManager<B> {
    backend: B,
}

impl<B> InputManager<B> {
    #[must_use]
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }

    #[must_use]
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    pub const fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    #[must_use]
    pub fn into_backend(self) -> B {
        self.backend
    }
}

impl<B: HostInputBackend> InputManager<B> {
    /// Reads the current state of the first attached controller.
    pub fn read_input(&mut self) -> Result<Option<ControllerState>, B::Error> {
        Ok(self.backend.poll()?.controllers.into_iter().next())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::convert::Infallible;

    use super::*;

    struct SnapshotBackend {
        snapshots: VecDeque<InputSnapshot>,
    }

    impl HostInputBackend for SnapshotBackend {
        type Error = Infallible;

        fn poll(&mut self) -> Result<InputSnapshot, Self::Error> {
            Ok(self.snapshots.pop_front().unwrap_or_default())
        }
    }

    fn controller(id: u64) -> ControllerState {
        ControllerState {
            id: ControllerId::new(id),
            name: format!("Controller {id}"),
            kind: ControllerKind::Standard,
            buttons: ButtonSet::default(),
            button_labels: FaceButtonLabels::default(),
            dpad: DPadState::default(),
            left_stick: StickState::default(),
            right_stick: StickState::default(),
            triggers: TriggerState::default(),
            motion: MotionState::default(),
        }
    }

    #[test]
    fn read_input_uses_the_first_controller_until_it_disconnects() {
        let mut input = InputManager::new(SnapshotBackend {
            snapshots: VecDeque::from([
                InputSnapshot {
                    controllers: vec![controller(1), controller(2)],
                },
                InputSnapshot {
                    controllers: vec![controller(2)],
                },
                InputSnapshot::default(),
            ]),
        });

        assert_eq!(
            input.read_input().unwrap().unwrap().id,
            ControllerId::new(1)
        );
        assert_eq!(
            input.read_input().unwrap().unwrap().id,
            ControllerId::new(2)
        );
        assert!(input.read_input().unwrap().is_none());
    }
}
