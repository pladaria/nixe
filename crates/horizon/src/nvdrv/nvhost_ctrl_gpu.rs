//! Kernel event resources owned by an open `/dev/nvhost-ctrl-gpu` descriptor.

use nixe_runtime::{EventObject, ExternalEventSource, ReadableEventObject, WritableEventObject};

use super::{NV_BAD_PARAMETER, diagnostics::NvDrvCallError};

#[derive(Debug)]
pub(super) struct NvHostControlGpuEvents {
    error: (WritableEventObject, ReadableEventObject),
    semaphore: (WritableEventObject, ReadableEventObject),
}

impl NvHostControlGpuEvents {
    pub(super) fn new() -> Self {
        Self {
            error: EventObject::create_pair_with_source(ExternalEventSource::GpuCompletion),
            semaphore: EventObject::create_pair_with_source(ExternalEventSource::GpuCompletion),
        }
    }

    pub(super) fn query(&self, event_id: u32) -> Result<ReadableEventObject, NvDrvCallError> {
        // QueryEvent copies an existing event; querying does not signal it.
        // https://switchbrew.org/w/index.php?title=NV_services#QueryEvent
        // GPU interrupt-producing methods remain unsupported at the Maxwell
        // execution boundary. Ordinary submission completion must not signal
        // these events: it is neither an error nor a semaphore interrupt.
        match event_id {
            1 => Ok(self.error.1.clone()),
            2 => Ok(self.semaphore.1.clone()),
            _ => Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvdrv::{NV_SUCCESS, NvDrvSession};

    #[test]
    fn control_gpu_events_preserve_identity_and_descriptor_lifetimes() {
        let session = NvDrvSession::new();
        session.initialize(0);
        let fd = session.open(b"/dev/nvhost-ctrl-gpu", 1).unwrap();
        let other_fd = session.open(b"/dev/nvhost-ctrl-gpu", 1).unwrap();
        let clone = session.clone_connection().unwrap();
        let (semaphore, result) = session.query_event(fd, 2, 1).unwrap();
        assert_eq!(result, NV_SUCCESS);
        let semaphore = semaphore.unwrap();
        let copy = clone.query_event(fd, 2, 1).unwrap().0.unwrap();
        let error = session.query_event(fd, 1, 1).unwrap().0.unwrap();
        let other = session.query_event(other_fd, 2, 1).unwrap().0.unwrap();
        for event in [&semaphore, &copy, &error, &other] {
            assert!(!event.is_signalled());
            assert_eq!(event.source(), Some(ExternalEventSource::GpuCompletion));
        }
        // Exercise the retained kernel pair to verify that QueryEvent copies
        // the same event, while different IDs and descriptors remain distinct.
        session.state.lock().unwrap().nvhost_control_gpu[&fd]
            .semaphore
            .0
            .signal();
        assert!(semaphore.is_signalled());
        assert!(copy.is_signalled());
        assert!(!error.is_signalled());
        assert!(!other.is_signalled());
        copy.clear();
        assert!(!semaphore.is_signalled());

        for id in [0, 3, u32::MAX] {
            let (event, result) = session.query_event(fd, id, 1).unwrap();
            assert!(event.is_none());
            assert_eq!(result, NV_BAD_PARAMETER);
        }
        let (event, result) = session.query_event(fd, 2, 2).unwrap();
        assert!(event.is_none());
        assert_eq!(result, NV_BAD_PARAMETER);

        assert_eq!(clone.close(fd), NV_SUCCESS);
        assert!(
            !session
                .state
                .lock()
                .unwrap()
                .nvhost_control_gpu
                .contains_key(&fd)
        );
        let (event, result) = session.query_event(fd, 2, 1).unwrap();
        assert!(event.is_none());
        assert_eq!(result, NV_BAD_PARAMETER);
        assert!(!semaphore.is_signalled());
        assert_eq!(session.teardown().device_fds_released, 1);
        assert!(session.state.lock().unwrap().nvhost_control_gpu.is_empty());
        assert!(!other.is_signalled());
    }
}
