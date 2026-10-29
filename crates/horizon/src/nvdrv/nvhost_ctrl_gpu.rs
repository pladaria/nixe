//! ZBC registrations and kernel events for `/dev/nvhost-ctrl-gpu`.

use nixe_runtime::{EventObject, ReadableEventObject, WritableEventObject};

use super::{
    NV_BAD_PARAMETER, NV_OVERFLOW, diagnostics::NvDrvCallError, input_u32, require_input_size,
    write_u32,
};

// Native clears carry their values directly and do not emit guest compression
// tags. The driver still retains registrations for subsequent table queries.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.cpp
#[derive(Debug, Default)]
pub(super) struct NvHostZbcTables {
    colors: Vec<ZbcColor>,
    depths: Vec<ZbcDepth>,
}

#[derive(Debug)]
struct ZbcColor {
    values: [u32; 8],
    format: u32,
    references: u32,
}

#[derive(Debug)]
struct ZbcDepth {
    value: u32,
    format: u32,
    references: u32,
}

impl NvHostZbcTables {
    // The ioctl carries two representations of the same clear color. Keep
    // both bit patterns; converting through a host float loses guest data.
    // https://github.com/switchbrew/libnx/blob/master/nx/source/nvidia/ioctl/nvhost-ctrl-gpu.c#L41-L78
    pub(super) fn set(&mut self, input: &[u8]) -> Result<Vec<u8>, NvDrvCallError> {
        require_input_size(input, 44)?;
        let format = input_u32(input, 36)?;
        match input_u32(input, 40)? {
            1 => {
                let mut values = [0; 8];
                for (index, value) in values.iter_mut().enumerate() {
                    *value = input_u32(input, index * 4)?;
                }
                if let Some(entry) = self
                    .colors
                    .iter_mut()
                    .find(|entry| entry.format == format && entry.values == values)
                {
                    entry.references = entry
                        .references
                        .checked_add(1)
                        .ok_or(NvDrvCallError::GuestResult(NV_OVERFLOW))?;
                } else {
                    self.colors.push(ZbcColor {
                        values,
                        format,
                        references: 1,
                    });
                }
            }
            2 => {
                let value = input_u32(input, 32)?;
                if let Some(entry) = self
                    .depths
                    .iter_mut()
                    .find(|entry| entry.format == format && entry.value == value)
                {
                    entry.references = entry
                        .references
                        .checked_add(1)
                        .ok_or(NvDrvCallError::GuestResult(NV_OVERFLOW))?;
                } else {
                    self.depths.push(ZbcDepth {
                        value,
                        format,
                        references: 1,
                    });
                }
            }
            _ => return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER)),
        }
        Ok(input.to_vec())
    }

    pub(super) fn query(&self, input: &[u8]) -> Result<Vec<u8>, NvDrvCallError> {
        require_input_size(input, 52)?;
        let kind = input_u32(input, 44)?;
        let index = input_u32(input, 48)? as usize;
        let mut output = vec![0; 52];
        let (references, format, count) = match kind {
            1 => {
                let entry = self
                    .colors
                    .get(index)
                    .ok_or(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))?;
                for (index, value) in entry.values.iter().enumerate() {
                    write_u32(&mut output, index * 4, *value)?;
                }
                (entry.references, entry.format, self.colors.len())
            }
            2 => {
                let entry = self
                    .depths
                    .get(index)
                    .ok_or(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))?;
                write_u32(&mut output, 32, entry.value)?;
                (entry.references, entry.format, self.depths.len())
            }
            _ => return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER)),
        };
        write_u32(&mut output, 36, references)?;
        write_u32(&mut output, 40, format)?;
        write_u32(&mut output, 44, kind)?;
        write_u32(&mut output, 48, count as u32)?;
        Ok(output)
    }
}

#[derive(Debug)]
pub(super) struct NvHostControlGpuEvents {
    error: (WritableEventObject, ReadableEventObject),
    semaphore: (WritableEventObject, ReadableEventObject),
}

impl NvHostControlGpuEvents {
    pub(super) fn new(kernel: &std::sync::Arc<nixe_scheduler::SchedulerState>) -> Self {
        Self {
            error: EventObject::create_pair(kernel),
            semaphore: EventObject::create_pair(kernel),
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
        let session =
            NvDrvSession::new(crate::test_memory::machine(), &crate::test_memory::kernel());
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
