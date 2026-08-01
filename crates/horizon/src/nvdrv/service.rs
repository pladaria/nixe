use super::diagnostics::NvDrvCallError;
use super::ioctl::{NvDrvIoctlOutcome, NvDrvIoctlRequest};
use super::{NvDrvFileDescriptor, NvDrvSession, UnsupportedNvDrvOperation};

/// Semantic NVIDIA service failure before Horizon response encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NvDrvServiceError {
    DriverResult(u32),
    Unsupported(UnsupportedNvDrvOperation),
}

impl From<NvDrvCallError> for NvDrvServiceError {
    fn from(error: NvDrvCallError) -> Self {
        match error {
            NvDrvCallError::GuestResult(result) => Self::DriverResult(result),
            NvDrvCallError::Unsupported(operation) => Self::Unsupported(operation),
        }
    }
}

/// Semantic service boundary used by Horizon's HIPC/CMIF wire adapter.
///
/// This type accepts decoded values and never reads guest descriptors or
/// encodes a Horizon response.
pub(crate) struct NvDrvService<'a> {
    session: &'a NvDrvSession,
}

impl<'a> NvDrvService<'a> {
    pub(crate) const fn new(session: &'a NvDrvSession) -> Self {
        Self { session }
    }

    pub(crate) fn open(
        &self,
        path: &[u8],
        process_id: u64,
    ) -> Result<NvDrvFileDescriptor, NvDrvServiceError> {
        self.session.open(path, process_id).map_err(Into::into)
    }

    pub(crate) fn ioctl(
        &self,
        request: NvDrvIoctlRequest<'_>,
    ) -> Result<NvDrvIoctlOutcome, UnsupportedNvDrvOperation> {
        self.session.ioctl_outcome(request)
    }

    pub(crate) fn close(&self, fd: NvDrvFileDescriptor) -> u32 {
        self.session.close(fd)
    }

    pub(crate) fn query_event(
        &self,
        fd: NvDrvFileDescriptor,
        event_id: u32,
        process_id: u64,
    ) -> Result<(Option<nixe_runtime::ReadableEventObject>, u32), UnsupportedNvDrvOperation> {
        self.session.query_event(fd, event_id, process_id)
    }

    pub(crate) fn initialize(&self) {
        self.session.initialize();
    }

    pub(crate) fn set_aruid(&self, process_id: u64, applet_resource_user_id: u64) {
        self.session.set_aruid(process_id, applet_resource_user_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvdrv::{NV_NOT_INITIALIZED, NvDrvDeviceKind};

    #[test]
    fn semantic_service_does_not_encode_horizon_wire_results() {
        let session = NvDrvSession::new();
        let service = NvDrvService::new(&session);

        assert_eq!(
            service.open(NvDrvDeviceKind::NvMap.path().as_bytes(), 7),
            Err(NvDrvServiceError::DriverResult(NV_NOT_INITIALIZED))
        );
    }
}
